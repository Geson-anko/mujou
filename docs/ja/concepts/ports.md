# Port と Channel

!!! note "書く内容"

    Port は接続のノード側の端で、Channel は 2 つの Port をつなぐ。メッセージは `torch.Tensor` の値と、書いた時点のグローバル時刻を運ぶ。読む側は常に最新の値を受け取り、待たない。値の shape は `TensorSpec` と `DictSpec` で宣言する。
