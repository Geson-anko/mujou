# 🧵 Threads and Devices

This page covers the threads a run computes on, the settings of a node (`device`, `seed`, `dedicated_thread`), CUDA, and the settings of torch that the whole process shares.

## The arguments of `Node`

`super().__init__` takes these keyword arguments, all optional.

| Argument           | Meaning                                                                | Default                       |
| ------------------ | ---------------------------------------------------------------------- | ----------------------------- |
| `period`           | The time between two updates on average, in seconds of the run's timer | `0.01`                        |
| `seed`             | The seed of the node's own random generator `self.generator`           | `None` (different every time) |
| `device`           | Where the node computes: `"cpu"` or a CUDA device                      | `"cpu"`                       |
| `dedicated_thread` | Run the node on a thread of its own instead of the run's workers       | `False`                       |

The node reads them back as the read-only properties `period`, `generator`, `device` and `dedicated_thread`.

## The workers

A run does not start a thread per node. It starts a pool of worker threads, which share the nodes: a worker runs one hook of one node, then takes the next work, the updates due earliest first. With many nodes, the threads stay few, and the OS does not spend its time switching between them.

- `mujou.run(net, workers=n)` sets the number of workers. The default is the number of CPUs the process may use (the logical cores, or fewer when the CPU affinity or a quota limits them), or half of it while the GIL is enabled, and never more than the nodes on the workers
- The hooks of a node never run at the same time, and each sees what the previous one wrote. Which worker runs a node may change from one hook to the next
- A hook that blocks holds its worker meanwhile, and the other nodes' `dt` grows (see the [rules a node keeps](../concepts.md#rules-a-node-keeps))

While the GIL is enabled (on a build of Python with it, or on the free-threaded build with `PYTHON_GIL=1`), one thread runs Python code at a time, so the nodes run in parallel only inside torch operations. More workers than half the CPUs mostly contend for the GIL, which is why the default is half there; for nodes that spend most of their time in Python, even fewer are about as fast.

## A thread of its own

Pass `dedicated_thread=True` when a node needs a thread of its own:

- its hooks block, waiting for a device, a socket or a lock: on a worker, the wait would hold the worker
- it relies on per-thread state set once: its `on_start` runs on its own thread, so make such settings there

```python
class Camera(Node):
    def __init__(self) -> None:
        super().__init__(period=1 / 30, dedicated_thread=True)
```

## Per-thread settings

On the workers, each hook of a node may run on a different thread, and the thread runs other nodes in between. So whether the next hook sees what a hook set per thread depends on the schedule, and the next node on the same worker may see it too:

- `threading.local`, `threading.current_thread()` and `threading.get_ident()` may differ from hook to hook
- Set a per-thread setting of torch, such as `torch.autocast` or gradient recording, back before the hook returns, by keeping it inside a `with` in the hook (`with torch.no_grad():`). A setting left behind leaks into the next node on the worker
- `contextvars` do carry over: each node runs its hooks in a `contextvars.Context` of its own, which starts empty (it does not inherit the host's)

Mujou itself sets only the current CUDA device and stream, before each hook of a CUDA node.

## Seeds and random numbers

Draw random numbers from the node's own generator, `self.generator`, made from `seed` on the node's device. It can be used right after `super().__init__()`, in `__init__` too.

```python
nn.init.normal_(self.model.weight, std=0.1, generator=self.generator)
noise = torch.randn(4, generator=self.generator, device=self.device)
```

`seed` reproduces only `self.generator`. The default initialization of a module such as `nn.Linear` draws from the global generator that every node shares, so initialize it again from `self.generator`. Do not call `torch.manual_seed` in a node: it changes the random numbers of the other nodes too. The state of `self.generator` is in the node's state, so a resumed run goes on with the same random sequence.

## Devices

`device` is the CPU or a CUDA device; any other, such as `"mps"` or `"meta"`, raises `ValueError`, and so does CUDA when it is not available. Received messages arrive on the node's device: allocate with `device=self.device`. `"cuda"` without an index is fixed to the host's current CUDA device when the node is made.

For a node on CUDA, Mujou:

- makes the node's device and a CUDA stream of the node's own current in each of its hooks, so every operation of the node goes on its stream
- waits for the work queued on that stream after each `on_update` and after the state is restored (the worker waits meanwhile, as in a hook that blocks)
- waits for it before and after `state_dict`, on a thread of its own (on the host's thread once the run has ended), so no worker waits for it
- keeps a tensor that the node sends on its own device there: `send` copies it on the node's stream and waits for that stream before the peers can read it. A peer on CUDA copies the message from device to device, without host memory in between (between two GPUs without peer access, the CUDA driver may still stage the copy through the host). A node on the CPU, and the host (`run.read`, `subscribe`, `wait_for`, `state_dict`), get a CPU copy when they read it. A tensor on another device is sent through host memory

So a node writes no synchronization. To synchronize yourself, wait only for your own stream (`torch.cuda.current_stream().record_event().synchronize()`), never `torch.cuda.synchronize()`, which waits for the whole device, the other nodes included. A device has a pool of 32 streams, so past 32 CUDA nodes on one device, some nodes share a stream.

!!! tip "Create CUDA tensors and modules in `on_start`"

    Tensors and modules made on CUDA in `__init__` work: `mujou.run` synchronizes the host's stream before it starts the nodes. Still, create them in `on_start` or later. PyTorch's caching allocator ties memory to the stream current when it was allocated, so memory allocated in `__init__` stays tied to the host's stream; when the node drops such a tensor, the memory may be reused by the host before the node's work on it has finished. A restore loads the state after `on_start`, so state made there is restored too.

A node that reads messages from a CUDA node on another device (a node on the CPU, or on another GPU) copies them on a stream of its own on that device, which Mujou makes current in each of its hooks too. Any other node on the CPU does not touch CUDA: on a worker that ran a CUDA node just before, that node's stream is still current, which a CPU node does not use.

The state tree holds CPU copies, also of the nodes on CUDA, and the default `load_state_dict` copies them back to the node's device. The state of a random generator differs between the CPU and CUDA, though, so a tree saved from nodes on CUDA restores only into nodes on CUDA (on another machine with a GPU too).

## Settings of the whole process

These settings are shared by the whole process. Do not change them in a node; the host sets them once, before `mujou.run`:

- `torch.manual_seed` and the default random generator
- `torch.set_num_threads`, `torch.use_deterministic_algorithms` and the flags of `torch.backends.*`
- operations on a whole device, such as `torch.cuda.synchronize()` and `torch.cuda.empty_cache()`

On the CPU, a torch operation uses further threads of its own for each thread that computes. With too many, the nodes compete for the cores and everything slows down, so the host keeps them few:

```python
--8<-- "examples/online_learning.py:threads"
```

!!! tip "Rule of thumb for the number of threads"

    The number of workers (plus the nodes with `dedicated_thread=True`) times the `n` of `torch.set_num_threads(n)` should not exceed the number of physical cores. With many nodes on small matrices, `n = 1` is enough.
