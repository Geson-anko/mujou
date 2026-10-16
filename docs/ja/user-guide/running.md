# ネットワークを走らせる

!!! note "書く内容"

    `h = mujou.run(*networks, timer=None, inputs=..., state=...)` と、それが返すハンドルを書く。ハンドルのメソッドは `write`、`read`、`subscribe`（最初に呼んだ時点の値を返す）、`state_dict`、`shutdown` と `join`（タイムアウト付き）で、with 文でも使える。
