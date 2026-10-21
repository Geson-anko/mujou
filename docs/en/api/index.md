# 📚 API Reference

These pages are generated from the docstrings. The API may still change before the first release.

| Page                          | What it covers                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------- |
| [Nodes](node.md)              | `mujou.Node`, the base class of a node, and its hooks                                             |
| [Ports and Channels](port.md) | `Port`, `MultiPort`, `Channel`, `MessageSpec`, `TensorSpec` and `DictSpec`                        |
| [Networks](network.md)        | `Network`, `Group`, `mujou.run` and `Run`                                                         |
| [Timers](time.md)             | The module `mujou.time`: `Timer`, `VirtualTimer`, `ReadOnlyTimer`, `Timestamp` and `global_timer` |
| [Errors](errors.md)           | `NetworkError`                                                                                    |
| [Testing](testing.md)         | The module `mujou.testing`: manual runs                                                           |
| [PC-ALM](pc.md)               | The module `mujou.pc`: the PC-ALM nodes, their messages and losses, `pc.functional`               |

`import mujou` gives everything on these pages except the modules `mujou.time`, `mujou.testing` and `mujou.pc`. Import those by name: `import mujou.time`, `import mujou.testing` and `import mujou.pc as pc`.
