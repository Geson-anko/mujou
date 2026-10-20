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
