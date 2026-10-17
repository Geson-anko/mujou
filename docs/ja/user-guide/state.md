# 状態の保存と復元

!!! note "書く内容"

    `h.state_dict()` で保存し、`mujou.run(..., state=...)` で復元する。既定の `load_state_dict` はその場でコピーする。Tensor でもモジュールでもない状態は `state_dict` と `load_state_dict` を override して扱う。例外で終わったときに状態を保存しない理由も書く。
