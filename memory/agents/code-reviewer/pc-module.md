# mujou.pc（e0b8774・8e5d1f4・003ddf1）レビュー

対象: `git diff feature/2026-10-20/carry...003ddf1` のうち carry 由来でない部分。計画 `memory/agents/implementation-planner/pc-module.md`、実装メモ `memory/agents/plan-implementer/pc-module.md`、設計 `specs/pc-module.md` と照合した。

## verdict: approve

must-fix はない。公開名・シグネチャ・既定値・Port の spec・on_update の手順 1〜7・learn_interval と ΣΔt・mask・時刻 0 の扱いは、どれも計画（orchestrator の決定 K2 と確認事項 1〜3 を含む）と spec に一致している。下の should-fix は docs の段階（計画のステップ 5）の前に片付けたい。

## must-fix

なし

## should-fix

1. `src/mujou/_core/pc/time_constants.py:14`、`functional.py:39,111,178`、`layer.py:27`、`output.py:25` — docstring の数式が RST の `.. math::` で書かれている。mkdocs は `docstring_style: google` と `pymdownx.arithmatex`（`$...$` / `$$...$$`、`mkdocs.yml:68-70`）で描画するので、`::: mujou.pc` を足すと「.. math::」が地の文で残り、式はインデントのせいでコードブロックとして出る。リポジトリにある他の docstring に数式の前例はない。`$$ ... $$` に揃えるか、地の文のコード表記にする。確信度: 中〜高。深刻度: 低（docs の見た目。ビルドは通る）
2. `specs/pc-module.md:3` — 冒頭が「状態: 計画（2026-10-19）。`mujou.pc` はまだ無い。」のまま。003ddf1 で仕様を実装に合わせた際に残った。計画のステップ 5 で「実装済み」に直す予定だが、このブランチの時点でもう事実と合わない。確信度: 高。深刻度: 低
3. マージ前の条件 — `mujou.pc` は公開 API だが、`docs/en/api/pc.md` とユーザーガイド（en/ja）がまだ無い。CLAUDE.md の「docs/ は実装済みの公開 API の使い方を書く」に従い、ステップ 5 を終えてから main へマージする。diff の欠陥ではなく、順番の確認。確信度: 高。深刻度: 中（マージの判断）

## nit

1. `src/mujou/_core/pc/learner.py:56-61`、`layer.py:138`、`functional.py:202` — `learn_interval=True` と `max_step=True` が 1 秒として通る（実行して確認）。`TimeConstants` と `Node(period=...)` は `bool` を `TypeError` で弾くので、検査の方針が揃っていない。確信度: 高
2. `src/mujou/_core/pc/functional.py:138-140` — `loss_vjp(wrt_input=False, params=())` の近道では `loss` の shape 検査を飛ばすので、docstring の `Raises: ValueError` が成り立たない。`vjp` は実装メモのとおり両方の経路で `delta` を検査しており、契約が非対称になっている。ノードからは通らない経路。確信度: 高
3. `tests/mujou/_core/pc/test_carry.py:39-54` `TestMessages` — assert が無く、`DictSpec` が作れることしか見ていない。`Input`（`DictSpec(Sample, ...)`）と `Output`（`DictSpec(Prediction, ...)`）の構築でもう検証済みなので重複。確信度: 中
4. `tests/mujou/_core/pc/test_losses.py:49-56` `test_probability_targets` — `cross_entropy` は `F.cross_entropy` をそのまま呼ぶだけなので、torch の動作の追試になっている（testing-strategy の「書かない」）。確信度: 中
5. `tests/mujou/_core/pc/test_layer.py:737-742` — `Learning.__new__` で `__init__` を飛ばし、属性を手で設定している。`Learning.__init__` に `state=` を受ける口を足すほうが壊れにくい。確信度: 中
6. `tests/mujou/_core/pc/test_output.py:125` — 周期の既定の assert が `test_the_prediction_carries_the_target` に混ざっている。テスト名と中身が一致しない。確信度: 高
7. `src/mujou/_core/pc/input.py:26` — 「ホストが書く前はゼロ（mask False）を送る」は直接のテストが無い（`test_a_first_input_written_at_time_zero_is_learned` から間接に分かるだけ）。確信度: 低
8. `tests/mujou/_core/pc/test_layer.py:64`、`test_output.py:23`、`test_functional.py:49,124` — `nn.Linear` の初期化のために全体の `torch.manual_seed` を使っている。決定性は保たれるが、`torch.random.fork_rng()` で囲めば他のテストの全体の RNG を汚さない。確信度: 低
9. 参考（設計）: `learner.py:74-79` — 時刻 0 の書き込みは、ゼロの付随データと時刻で区別できないので、位相はノードの最初の更新から数える（決定済み）。深い経路では最初のサンプルの緩和が `learn_interval` より短くなり得る。`Input` 側で「書かれたことがあるか」を区別する案も考えられるが、`Prediction.time` を `wait_for` に使う契約と衝突するので、今の決定で問題ない。確信度: 低
10. `learner.py:115-118` — `learn_interval` が許容 0.5 ns 以下だと、入力が切り替わった更新でも学習する（`None` と同じではない）。実用上は起きない。確信度: 低

## 検証結果

- just run: 未実行（依頼により。別エージェントが同じ worktree で作業中）
- `uv run pytest tests/mujou/_core/pc tests/mujou/test_init.py -q -p no:cacheprovider`: 100 passed（22 s。slow の合計は約 20 s、slow でないものは合わせて約 1.5 s）
- `pyright src/mujou/_core/pc src/mujou/pc`: 0 errors
- `ruff check` / `ruff format --check`（pc の src とテスト）: pass
- `</content>` などの混入の grep: なし
- tests/ 配下でテストファイルの basename の衝突: なし

---

# mujou.pc 再レビュー（fc8ddf2..ecadf69: 71fafe6・409b21a・a55dc8a・04c5f46・ecadf69）

h の形への作り直し・`max_dt`・学習の時刻（書いた時刻 + k·learn_interval）・mask の行ごとの cotangent・example のスケール 1 を、`specs/pc-module.md`、実装メモ（「レビューと実験への対応」節）と照合した。HEAD（3a36e66）は ecadf69 に yomiyasu の skill をマージしただけで、src・tests に差は無い。

## verdict: request-changes（コードの欠陥ではなく、ユーザーの承認待ち 1 件。承認されれば approve）

コードは spec の on_update 手順 1〜8 と一致し、テストの参照実装（`Reference`・`layer_update`・`relaxed`）とも 1e-12〜1e-14 で一致する。前回の指摘は should-fix 3（docs）を除き直っている。

## must-fix

1. `src/mujou/_core/pc/learner.py:122-154`、`specs/pc-module.md:173-175,291` — 学習の時刻を「入力が変わってから」から「ホストが書いた時刻から」に変えたのは plan-implementer の判断で、2026-10-19 にユーザーが承認した規則（旧 spec「入力が変わってから learn_interval だけ経った時点で学習する」）を変えている。spec の未決定事項にも「ユーザーの確認が要る」とある。理由（揺らいだ順で発散、書いた時刻から数えると安定）は筋が通っており、コードを直す必要は無い可能性が高い。マージ前に orchestrator からユーザーへ確認を中継する。確信度: 高（承認が無いことについて）

## should-fix

1. `learner.py:132-143` — 書いた時刻から数えるので、ノードが新しい入力を初めて見た更新ですでに `time + learn_interval` を過ぎていると、その更新で学習する。そのときの δ は `h - mu_last + lam`（前の入力の r と λ）で、新しい入力の Jacobian を通した古い credit で θ を動かす。spec が避けたかった「切り替え直後の δ で学習する」状況（spec 205-210 行）にあたる。起きる条件は (a) 経路の深さ × 周期 ≥ learn_interval、(b) 負荷でノードが learn_interval 以上止まった直後に新しい入力を見たとき。MNIST の既定（S=32、深さ 3）では起きない。少なくとも Layer/Output の `learn_interval` の docstring と spec に「経路の遅れより十分長く」と制約を書くか、新しい入力を見た更新では学習しない（次の更新に回す）などの扱いを決める。深いノードほど新しい入力で緩和できる時間が短い（learn_interval − 深さ × 周期）ことも同じ根。確信度: 高（機構）、影響: 中（深い鎖・負荷時）
2. `learner.py:66,82`、`layer.py` Args — `period` が `max_dt` より長くても何も言わない。例えば `period=tau.h` を指定すると、毎回の更新が `tau.h / 2` に打ち切られ、緩和も学習も時間の半分を黙って捨てる。構築時に `period > max_dt` を `ValueError` にするか、`max_dt` の docstring に「period より長くする」と書く。確信度: 中（仕様として許すかの判断）
3. 前回の should-fix 3 の継続 — `docs/` の `mujou.pc` の使い方と API リファレンスが無い（別 worktree で作業中）。main へのマージはそれを待つ。確信度: 高

## nit

1. `src/mujou/_core/pc/output.py:33` — docstring の 1 行が 130 文字（手で直したときの折り返し忘れ）。36-44 行も折り返し幅が不揃い。ruff は E501 を無視しているので検出されない。確信度: 高
2. `tests/mujou/_core/pc/test_layer.py:86-115`（`Chain`）— `max_dt` を Layer にだけ渡し、Output には渡さない。Output の `max_dt`（ΣΔt の打ち切り）と Output の bool の TypeError を見るテストが無い（Learner の共通コードなので実害は小さい）。確信度: 高
3. `TestJacobi`（test_layer.py:327）は `learn_interval=None` で毎回学習するので、2 回目以降の更新はすべて付け替え（`_moved`）の経路になる。付け替えない経路の δ（`h - mu`）と参照実装の一致は、学習ありでは `TestMovesWithMu`・`TestLearnInterval` の `layer_update` が見ている。Jacobi のテストに `learn_interval` を付けた版を足すと、付け替えあり/なしが混ざる経路を 1 本で照合できる。確信度: 中
4. `specs/pc-module.md:39` — 「それ以外の更新では…h_in が動いても付け替えない」。θ を動かした直後の更新の付け替えは `mu(新 θ, 今の h_in) - mu_last(旧 θ, 前の h_in)` なので、その 1 更新の間の h_in の動きも h に吸収される。spec の書き方だと θ の跳びだけを吸収するように読める。algorithm-reviewer の領分。確信度: 中
5. `functional.py` `relax_` の「0 * inf would make NaN of an infinite r」— r が無限大なら h の更新の時点で `inf - inf` で h が NaN になるので、λ を守る意味は薄い（`test_infinite_lam_time_constant_keeps_the_dual` も λ だけを見る）。害は無い。確信度: 中

## 前回の指摘の確認

- should-fix 1（RST の math）: `$$`・`$` と `r"""` に直っている（arithmatex generic と合う）
- should-fix 2（spec の状態行）: 直っている
- should-fix 3（docs）: 未了（上の should-fix 3）
- nit 1（bool）: `learn_interval`・`max_dt` とも TypeError。`max_step` は削除
- nit 2（loss_vjp の近道）: 近道でも `_check_loss` を通る
- nit 3・4: `TestMessages`・`test_probability_targets` を削除
- nit 5: `Learning(state=, steps=)` に置き換え、`__new__` は無い
- nit 6: 周期のテストを分離
- nit 7: `test_sends_zeros_without_a_target_before_the_host_writes` を追加
- nit 8: `torch.random.fork_rng()` で囲んだ
- nit 9・10: 設計の変更で前提が変わった（nit 10 は learn_interval が 1 ns 未満で None と同じ挙動になるだけ）

## 検証結果

- just run: 未実行（依頼により。作業ツリーを変えないため）
- `uv run pytest tests/mujou/_core/pc tests/mujou/test_init.py -q -m "slow or not slow" --no-cov`: 104 passed（37.6 s）
- `pyright src/mujou/_core/pc src/mujou/pc`: 0 errors
- `ruff check` / `ruff format --check`（pc の src・tests、examples/mnist）: pass
- `</content>` などの混入: なし
- `max_step`・`u` の形・`_phase` の残り: src・docs・examples・tests に無し
