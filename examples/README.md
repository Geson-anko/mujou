# Examples

Programs that show how to use Mujou. The documentation includes them with the
snippets syntax (`--8<-- "examples/<file>.py"`), and `tests/test_examples.py` runs
each of them, so the code on the site keeps working.

- To include a part of a program, mark it in the program with
  `# --8<-- [start:<section>]` and `# --8<-- [end:<section>]`, and include it with
  `--8<-- "examples/<file>.py:<section>"`. Do not include line ranges: they break
  when the program changes. The markers do not show on the site, also where a page
  includes the whole file.
- A program is a single file `examples/<name>.py`, or a directory
  `examples/<name>/` whose entry point is `main.py` (other files there are helpers).
- A program must finish by itself, with exit status 0, within the timeout of the
  test (60 seconds), and must not need a GPU or the network.
- A program that runs a network must end its run with `shutdown()` or `join()`
  (or run it in a `with mujou.run(...) as run:` statement), so that an exception
  in a node reaches the program. The test also fails when the
  program prints a traceback to stderr.
