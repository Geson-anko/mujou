# pc-g-in-after-relax 実装ログ

- layer.py: on_update で g の合計と relax_ を _predict の前に移した。docstring（クラス、_predict、learn_interval の説明）を計画どおりに直した
- 計画外: `time_constants.py` の docstring 「credit moves one node in each period」を「at most one node」に直した（同じ記述が残っていたため）
- examples/pc_merge.py: relax_ を delta の前に移した。cifar10 は resnet.py と main.py のコメントだけ。S と learn_interval は不変
- 速度: 前は HEAD の layer.py だけを差し替えた src の複製（scratchpad/base2）を PYTHONPATH で当てた。前後を 3 回ずつ交互に回し、差は揺れの内（他の作業が同じ機械で動いていた）
- MNIST: 97.69%（BP 97.53%）。計画は「変わったときだけ直す」なので specs/pc.md 235 行と examples/mnist/main.py の docstring を更新。docs/en,ja の topics/pc.md の 97.56% は未更新（担当外）
- skill pc-alm、memory、CHANGELOG は、この時点で別の担当が編集済み（git status で確認）
- IF 変更通知: なし
