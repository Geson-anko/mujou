use std::collections::BTreeSet;
use std::sync::mpsc::{self, RecvTimeoutError};

use super::*;

fn threads(log: &Log) -> BTreeSet<String> {
    let log = log.lock().unwrap();
    log.iter().map(|(_, _, thread)| thread.clone()).collect()
}

fn numbered(prefix: &str, k: usize) -> BTreeSet<String> {
    (0..k).map(|i| format!("{prefix}{i}")).collect()
}

/// The updates of one step of the virtual time: receives exactly `k`.
fn receive(ticks: &Receiver<(usize, u64)>, k: usize) -> Vec<(usize, u64)> {
    (0..k).map(|_| ticks.recv_timeout(LONG).unwrap()).collect()
}

#[test]
fn updates_the_earliest_target_first_and_by_index_on_a_tie() {
    let timer = virtual_timer();
    let Probes { mut nodes, .. } = probes(3);
    let (tx, ticks) = mpsc::channel();
    for node in &mut nodes {
        node.ticks = Some(tx.clone());
    }
    let config = with_periods(pool(1, config(nodes)), &[3 * H_NS, 2 * H_NS, H_NS]);
    let (rt, started) = Runtime::start(config, clock(&timer)).unwrap();
    started.wait().unwrap();
    let h = H_NS;
    // The targets are 3, 2 and 1 ms: at 5 ms all are due, earliest first.
    // Then `next_target_ns` gives 6, 7 and 6 ms.
    timer.advance_ns(5 * h).unwrap();
    assert_eq!(receive(&ticks, 3), [(2, 5 * h), (1, 5 * h), (0, 5 * h)]);
    // 6 ms: nodes 0 and 2 tie, by index. Next 9 and 7 ms.
    timer.advance_ns(h).unwrap();
    assert_eq!(receive(&ticks, 2), [(0, h), (2, h)]);
    // 7 ms: nodes 1 and 2. Next 9 and 8 ms.
    timer.advance_ns(h).unwrap();
    assert_eq!(receive(&ticks, 2), [(1, 2 * h), (2, h)]);
    timer.advance_ns(h).unwrap();
    assert_eq!(receive(&ticks, 1), [(2, h)]);
    timer.advance_ns(h).unwrap();
    assert_eq!(receive(&ticks, 3), [(0, 3 * h), (1, 2 * h), (2, h)]);
    assert_eq!(ticks.recv_timeout(QUIET), Err(RecvTimeoutError::Timeout));
    shut_down(&rt).unwrap();
}

/// Updates that take longer than the time between the targets: the node
/// that just updated waits behind the others, so they take turns, and only
/// their Δt grows (the real time since each one's last update).
#[test]
fn an_overloaded_worker_serves_the_nodes_in_turn() {
    let timer = virtual_timer();
    let Probes { mut nodes, .. } = probes(3);
    let (tx, ticks) = mpsc::channel();
    for node in &mut nodes {
        node.ticks = Some(tx.clone());
        // Each of the first two updates takes 1 ms.
        node.advance = Some((Arc::clone(&timer), 2));
    }
    let config = pool(1, config(nodes));
    let (rt, started) = Runtime::start(config, clock(&timer)).unwrap();
    started.wait().unwrap();
    timer.advance_ns(1).unwrap();
    let h = H_NS;
    // The updates start at 1 ns, 1 ms + 1 ns, 2 ms + 1 ns, and so on, each
    // first two moving the time by 1 ms; the last three all at 6 ms + 1 ns.
    let expected = [
        (0, 1),
        (1, h + 1),
        (2, 2 * h + 1),
        (0, 3 * h),
        (1, 3 * h),
        (2, 3 * h),
        (0, 3 * h),
        (1, 2 * h),
        (2, h),
    ];
    assert_eq!(receive(&ticks, 9), expected);
    assert_eq!(ticks.recv_timeout(QUIET), Err(RecvTimeoutError::Timeout));
    shut_down(&rt).unwrap();
}

/// Steps every node of a manual runtime a few times.
fn step_all(timer: &Timer, rt: &Runtime<Probe>) {
    for _ in 0..3 {
        timer.advance_ns(H_NS).unwrap();
        rt.step(None).wait().unwrap();
    }
}

#[test]
fn runs_the_hooks_on_the_given_number_of_workers() {
    let timer = virtual_timer();
    let Probes { log, nodes, .. } = probes(64);
    let (rt, started) = Runtime::start(pool(2, manual(nodes)), clock(&timer)).unwrap();
    started.wait().unwrap();
    step_all(&timer, &rt);
    shut_down(&rt).unwrap();
    assert!(threads(&log).is_subset(&numbered("mujou-worker-", 2)));
}

/// Sees only the threads that ran a hook: an extra idle worker would go
/// unnoticed unless it happened to take some work.
#[test]
fn spawns_no_more_workers_than_nodes_in_the_pool() {
    let timer = virtual_timer();
    let Probes { log, nodes, .. } = probes(2);
    let (rt, started) = Runtime::start(pool(8, manual(nodes)), clock(&timer)).unwrap();
    started.wait().unwrap();
    assert_eq!(rt.workers(), 2);
    step_all(&timer, &rt);
    shut_down(&rt).unwrap();
    assert!(threads(&log).is_subset(&numbered("mujou-worker-", 2)));
}

/// Sees only the threads that ran a hook: a worker would run none here, so
/// this checks that the dedicated threads run every hook.
#[test]
fn the_dedicated_threads_run_every_hook_without_nodes_in_the_pool() {
    let timer = virtual_timer();
    let Probes { log, nodes, .. } = probes(2);
    let mut config = manual(nodes);
    for node in &mut config.nodes {
        node.dedicated_thread = true;
    }
    let (rt, started) = Runtime::start(config, clock(&timer)).unwrap();
    started.wait().unwrap();
    assert_eq!(rt.workers(), 0);
    step_all(&timer, &rt);
    shut_down(&rt).unwrap();
    assert_eq!(threads(&log), numbered("mujou-node-p", 2));
}

/// Waits for the end of its own runtime from an update.
struct Caller {
    runtime: Arc<Mutex<Option<Arc<Runtime<Caller>>>>>,
    result: Arc<Mutex<Option<Result<bool, RuntimeError>>>>,
}

impl NodeHooks for Caller {
    type State = ();

    fn update(&mut self, _dt: f64) -> Result<(), HookError> {
        let rt = self.runtime.lock().unwrap().clone().unwrap();
        *self.result.lock().unwrap() = Some(rt.join(None));
        Ok(())
    }

    fn capture(&mut self) -> Result<(), HookError> {
        Ok(())
    }
}

#[test]
fn a_hook_on_a_worker_cannot_wait_for_its_own_runtime() {
    let timer = virtual_timer();
    let runtime = Arc::new(Mutex::new(None));
    let result = Arc::new(Mutex::new(None));
    let caller = Caller {
        runtime: Arc::clone(&runtime),
        result: Arc::clone(&result),
    };
    let (rt, started) = Runtime::start(pool(1, manual(vec![caller])), clock(&timer)).unwrap();
    started.wait().unwrap();
    let rt = Arc::new(rt);
    *runtime.lock().unwrap() = Some(Arc::clone(&rt));
    timer.advance_ns(H_NS).unwrap();
    rt.step(None).wait().unwrap();
    assert!(matches!(
        *result.lock().unwrap(),
        Some(Err(RuntimeError::Reentrant))
    ));
    // Break the cycle between the node and its runtime.
    *runtime.lock().unwrap() = None;
    shut_down(&rt).unwrap();
}

/// A smoke test on the real clock, after the deterministic ones: free
/// running nodes on several workers update and stop. The hooks of a node
/// cannot overlap here even if the scheduling broke, since a node is
/// called under its lock; a node taken by two workers would trip the
/// `debug_assert` in the core instead and leave the runtime hanging.
#[test]
fn free_running_nodes_on_several_workers_update_and_stop() {
    let timer = Arc::new(Timer::new(Source::Real, Timestamp::from_ns(0)));
    let Probes { log, nodes, .. } = probes(16);
    let config = pool(4, config(nodes));
    let (rt, started) = Runtime::start(config, clock(&timer)).unwrap();
    started.wait().unwrap();
    thread::sleep(Duration::from_millis(200));
    shut_down(&rt).unwrap();
    assert!(threads(&log).is_subset(&numbered("mujou-worker-", 4)));
    for id in 0..16 {
        assert!(events(&log, id).len() > 2);
    }
}

/// While one worker is held in an update, another one takes over the
/// timekeeping and updates the other node on time.
#[test]
fn another_worker_keeps_the_time_while_one_is_held() {
    let timer = virtual_timer();
    let Probes { mut nodes, .. } = probes(2);
    let (entered_tx, entered) = mpsc::channel();
    let (release, release_rx) = mpsc::channel();
    nodes[0].gate = Some((entered_tx, release_rx));
    let (tx, ticks) = mpsc::channel();
    nodes[1].ticks = Some(tx);
    let config = with_periods(pool(2, config(nodes)), &[H_NS; 2]);
    let (rt, started) = Runtime::start(config, clock(&timer)).unwrap();
    started.wait().unwrap();
    timer.advance_ns(H_NS).unwrap();
    entered.recv_timeout(LONG).unwrap();
    assert_eq!(ticks.recv_timeout(LONG), Ok((1, H_NS)));
    for _ in 0..3 {
        timer.advance_ns(H_NS).unwrap();
        assert_eq!(ticks.recv_timeout(LONG), Ok((1, H_NS)));
    }
    // Node 0 returns at once from its later updates.
    drop(release);
    shut_down(&rt).unwrap();
}
