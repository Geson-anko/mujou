# Examples

Programs that show how to use Mujou. The documentation includes them with the
snippets syntax (`--8<-- "examples/<file>.py"`), and `tests/test_examples.py` runs
each of them, so the code on the site keeps working.

- A program is a single file `examples/<name>.py`, or a directory
  `examples/<name>/` whose entry point is `main.py` (other files there are helpers).
- A program must finish by itself, with exit status 0, within the timeout of the
  test (60 seconds), and must not need a GPU or the network.
