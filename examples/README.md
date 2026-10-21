# Examples

Programs that show how to use Mujou. They are guides for users, not tests: the
documentation includes them with the snippets syntax
(`--8<-- "examples/<file>.py"`), and the test suite does not run them. What the
library guarantees is tested in `tests/`. The list of the programs, with what each
shows, is the Examples page of the documentation (`docs/en/examples.md`).

- To include a part of a program, mark it in the program with
  `# --8<-- [start:<section>]` and `# --8<-- [end:<section>]`, and include it with
  `--8<-- "examples/<file>.py:<section>"`. Do not include line ranges: they break
  when the program changes. The markers do not show on the site, also where a page
  includes the whole file.
- A program is a single file `examples/<name>.py`, or a directory
  `examples/<name>/` whose entry point is `main.py` (other files there are helpers,
  imported by their names, as `import bp`).
- `test_leaky.py` shows how to test a node: it tests the node of `leaky.py`, and a
  reader runs it with `uv run pytest examples/test_leaky.py`. Neither `just test`
  nor CI runs it.
- Write what helps a reader. A program may download data or run for minutes;
  say so, and what it needs (a GPU, the network, the time it takes), in its
  docstring.
- A program that runs a network must end its run with `shutdown()` or `join()`
  (or run it in a `with mujou.run(...) as run:` statement), so that an exception
  in a node reaches the program.
- `just type` checks the programs with pyright (standard mode), so a change of the
  API that breaks one fails there. When you change the library, also run the
  programs it affects by hand.
