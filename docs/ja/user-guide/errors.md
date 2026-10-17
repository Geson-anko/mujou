# 例外と終了

!!! note "書く内容"

    hook が例外を投げたときの流れを書く。全ノードで `on_exception`、続いて `on_stop` が呼ばれ、ホストには `NetworkError` が届く。`on_start` の例外は `mujou.run` 自身が送出する。停止のタイムアウト（既定は実時間で 30 秒）も書く。
