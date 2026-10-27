# h の更新手法を差し替える仕組み（Relaxer）

依頼（2026-10-26、ユーザー）: 「h に momentum を設定できるようにしたなら、h にも θ の optimizer のように更新の仕方を自由に変えられる仕組みが必要。別 PR で、solo-dev-cycle + レビューエージェント」。

## 段階 1: 計画

土台: PR #108（`TimeConstants.momentum`、未 merge）。#108 のブランチから分岐し、PR の base も #108 のブランチにする（積み PR）。#108 が merge されたら base を main に付け替える。

設計（`pc.SGD` / `pc.AdamW` の `Optimizer` / `BoundOptimizer` と対称）:

- `mujou/pc/relax.py`
  - `Relaxer`（ABC、設定。状態を持たない frozen 値）: `bind(h: Tensor) -> BoundRelaxer`。`h` と同じ shape・dtype・device の状態を作る
  - `BoundRelaxer(Stateful)`（ABC）: `step_(h, force, dt, tau_h)`（h を in place で動かす。force は `h - mu + lam + g`）と `reset_()`（以前の力の記憶を捨てる。新しいサンプル・h_in の跳びで呼ぶ）
  - `Euler()`: 既定。`h -= dt/tau_h * force`。状態なし
  - `HeavyBall(tau)`: `v <- exp(-dt/tau) v + force`、`h -= dt/tau_h * v`。dt=0 は何もしない。状態 `{"velocity": v}`
- `functional.relax_(h, lam, mu, g, dt, tau, relaxer=None)`: velocity 引数をやめ、`BoundRelaxer` を取る。None は Euler。λ の更新は今まで通り（差し替えの対象は h だけ）
- `TimeConstants.momentum` は削除（#108 で入れたが未 merge。差し替えの仕組みに載せ替える）
- `Layer(relaxer=Euler())`: `self.relaxer = relaxer.bind(h)`、状態キーは `"relaxer"`（以前の `"velocity"` は `"relaxer"` の中の `"velocity"` になる）。新しいサンプルか h_in の跳びで `reset_()`、θ の跳びでは呼ばない
- `Chain(relaxer=Euler())`: 各 Layer に渡す。Output は h を持たないので受け取らない
- 公開名: `Relaxer`, `BoundRelaxer`, `Euler`, `HeavyBall`
- 例 `examples/cifar10/main.py`: `--momentum`（τ_h の倍数）は残し、`relaxer=pc.HeavyBall(tau=args.momentum * args.tau_h)` にする。数値は変わらない

テスト観点: Euler・HeavyBall の 1 歩が式どおり／dt=0／reset／検証（tau は有限で正）／状態の保存と復元（torch.save 経由、メモリ共有、コピー）／抽象メソッド／Layer（既定は今までと同じ、HeavyBall で credit が速く届き平衡点は同じ、状態は `"relaxer"`、新サンプルで reset、θ の跳びで reset しない、自作の Relaxer が呼ばれる）／Chain が全 Layer に別々の束縛を渡す。

リスク: 状態キーの変更（`"velocity"` → `"relaxer"`）は #108 の未 merge の名前なので互換の問題はない。

## 進捗（段階 3 まで完了、2026-10-26）

- 実装: `src/mujou/pc/relax.py`（Relaxer / BoundRelaxer / Euler / HeavyBall）、`functional.relax_(..., relaxer=None)`、`Layer(relaxer=Euler())`、`Chain(relaxer=)`、`TimeConstants.momentum` を削除、`examples/cifar10/main.py` を HeavyBall に、`tests/test_layers.py` に `mujou.pc.relax` を登録
- テスト: `test_relax.py` を新設、functional / layer / chain / init / time_constants のテストを載せ替え。`just run` は green（1010 passed、pyright 0、docs-build OK）
- docs / specs / memory: en と ja の `topics/pc.md`・`topics/pc-concepts.md`（7 節 relaxer）、`api/pc.md`、`specs/pc.md`（「h の relaxer」）、`specs/open-decisions.md`、`memory/project_pc_relaxer.md` を更新済み
- 未了: (1) ユーザー指示「ドキュメンテーションは /doc-teacher-student /yomiyasu」: 追加した docs（en: pc.md の「Choosing the relaxer」と pc-concepts.md の 7 節、ja: 同）を doc-teacher-student のループと yomiyasu（ja）にかける (2) code-reviewer と algorithm-reviewer の起動と指摘の反映 (3) commit（種別 feat(pc)、docs は別 commit でもよい）、push、PR（base は `feature/2026-10-23/cifar10-resnet-example`、#108 の上に積む）
- 注意: 未コミット。worktree は `/data/geson/mujou/.claude/worktrees/feature+2026-10-26+h-relaxer`（EnterWorktree で入った）。このセッションの git は自分の worktree だけに向ける。Bash は複合コマンド（`;`・`&&`・パイプ併用）を拒否することがある

## レビュー（code-reviewer と algorithm-reviewer、2026-10-26）

どちらも approve（must-fix なし）。反映したもの:

- **丸め（should-fix）**: `_BoundEuler` と `_BoundHeavyBall` の `alpha=` つき `sub_` は FMA になり、#108 の `sub_((dt/tau)*force)` と末尾ビットで違った（algorithm-reviewer の実測で差 3e-16、code-reviewer の実測で 100 万要素中 17 万が不一致）。#108 と同じ式に戻し、`relax_` の素の 1 歩と一致させた。`test_euler_is_the_step_without_a_relaxer` は dt / tau.h が 2 のべきだとたまたま通るので、2 のべきでない値にした
- **reset の理由づけ（should-fix）**: 付け替えの瞬間は g が下流の最後の値のままなので力は連続する。「力が跳ぶから reset」は不正確で、「速度が積んだのは前の入力の力だから reset」が正しい。docstring、コメント、specs、docs（en と ja）、memory を直した
- **不動点（should-fix）**: HeavyBall は F=0 でも v≠0 なら h が動く。不動点は「F=0 かつ v=0」と書き直した
- **持てる状態の条件（should-fix）**: ノードは `step_` の外でも h を動かし（付け替え、θ の跳び）、θ の跳びは relaxer に知らされない。位置の型の状態（先読み、前回の h）は持てない。`BoundRelaxer` の docstring と specs に書いた。`shift_` のような通知は要るようになってから足す（今は実装しない）
- **テスト（should-fix）**: `Layer` を通した復元のテストに HeavyBall を足し（速度が 0 でないことも確かめる）、h_in の跳びで reset するテストを足した
- 小さい直し: memory の description（段階 2）、`specs/scenarios.md` に K32（HeavyBall.tau）

載せなかったもの（nit）:

- `load_state_dict` の食い違いの扱い（Euler の状態を HeavyBall に読むと `KeyError`、逆は黙って無視）: `BoundOptimizer` と同じ作りなので、この PR の範囲外
- `TestBound` と `TestAbstract` の価値が薄い: 害はないので残す

測ったが載せていない所見（algorithm-reviewer）:

- 下流の層ほど、上流の θ の跳び（`jumps` の連鎖）のたびに reset される（4 層で 100 周期、learn_interval 12 周期のとき、reset は l0=2、l1=11、l2=20、l3=29 回）。τ が約 1 周期なら影響は小さいが、τ を大きくすると効きが落ちるおそれがある。#108 から変わっていない挙動で、測っていない
- HeavyBall の利得は周期に依る（周期を半分にすると時間あたりの利得がほぼ 2 倍）。specs に既知の性質として書いてある

ドキュメント推敲（doc-teacher-student、yomiyasu）:

- en: pc.md の「Choosing the relaxer」は R2 で 6/6、保留 3/3。pc-concepts.md の「7. Relaxers」は R2 で 8/8（保留 H2、H3 を評価に移した後）
- 直した主な点: 係数 `exp(-dt/tau)` が実効の指標であること、0.78 は恒等写像の線形な鎖の限界で ResNet はそれより低い係数で崩れたこと、速度は 0 から始まり最初の 1 歩は Euler と同じこと、λ は relaxer が動かさないこと
- ja は yomiyasu で推敲した。太字ラベルとコロンで始まる箇条書きは、yomiyasu の規則で外した（この 2 節だけ。隣の節は太字ラベルのまま）
