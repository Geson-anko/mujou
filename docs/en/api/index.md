# 📚 API Reference

These pages are generated from the docstrings. The API may still change before the first release.

| Page                          | What it covers                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------- |
| [Nodes](node.md)              | `mujou.Node`, the base class of a node, and its hooks                                             |
| [Ports and Channels](port.md) | `Port`, `PrevPort`, `NextPort`, `Channel`, `TensorSpec` and `DictSpec`                            |
| [Networks](network.md)        | `Network`, `Group`, `Graph`, `mujou.run`, `Run` and `StateDict`                                   |
| [Timers](time.md)             | The module `mujou.time`: `Timer`, `VirtualTimer`, `ReadOnlyTimer`, `Timestamp` and `global_timer` |
| [Types](typing.md)            | The module `mujou.typing`: `Single`, `Multi` and `MessageSpec`, for type annotations              |
| [Errors](errors.md)           | `NetworkError`                                                                                    |
| [Testing](testing.md)         | The module `mujou.testing`: manual runs                                                           |
| [PC-ALM](pc.md)               | The module `mujou.pc`: the PC-ALM nodes, their messages and losses, `pc.functional`               |

`import mujou` gives everything on these pages except `mujou.testing` and `mujou.pc`. Import those modules with `import mujou.testing` and `import mujou.pc as pc`.
