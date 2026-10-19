# grad-on-by-default レビュー

対象: `git diff refactor/2026-10-19/simplify-codebase...HEAD`（5 コミット、a0f229f まで）

## verdict: request-changes

コード（Rust・pyi・Python・tests・benchmarks）は要求どおりで正しい。文書の削除漏れが 4 か所残っている。どれも 1 文の修正で済む。

## must-fix

1. `specs/native-core.md:71` / 文がまだ「hook の間で保つのは `contextvars` と、フレームワークが戻す勾配モード（hook の前に毎回）・CUDA の device と stream（切り替えのたび）だけ」になっている。同じ diff で直した 69 行目（「勾配の記録には触れない」）や `specs/torch.md:87` と矛盾する / 根拠: 要求「`_before_hook` の削除漏れ（... specs）」 / 確信度: 高
2. `.claude/skills/locality-and-async/SKILL.md:70` / 「フレームワークが戻すのは勾配モード（毎回）と CUDA の device・stream ... だけ」が残っている。同じファイルの 47 行目は直してあり、矛盾する / 根拠: 要求「削除漏れ（... skill）」 / 確信度: 高

## should-fix

3. `.claude/skills/locality-and-async/SKILL.md:104-110`（「ノード内の autograd」の条件 3）/ `# mujou: local-autograd` を付ける条件が無条件のまま。決定では目印は `src/mujou/` の中のコード向けで、利用者のコード（examples/、docs）では使わない。CONTRIBUTING.md と specs/torch.md:149 はもう範囲を `src/mujou/` に絞っている。`specs/nodes.md:201` はこの skill を「ノード内で autograd を使う書き方」として参照している / 根拠: 要求「目印は `src/mujou/` の中のコード向けに残す。利用者向けでは使わない」 / 確信度: 高
4. `memory/MEMORY.md:39` と `memory/project_mn_scheduler.md:14`（Q3）/ どちらもまだ「保つのは contextvars と勾配モード・CUDA の device/stream」になっている。同じ diff で `project_mn_scheduler.md:16` には廃止を書き足したが、Q3 の行と MEMORY.md の索引の行が古いまま / 根拠: memory はセッション開始時に読む。古い記述が残ると、次の作業者が誤った不変条件を信じる / 確信度: 高
5. `docs/en/user-guide/torch.md:53` と `docs/ja/user-guide/torch.md:53` / 「An attribute computed from the parameters ... keeps its graph, which then grows with every update」は一般化しすぎ。グラフが更新ごとに伸びるのは、例の `self.h = f(self.h)` のように前の値を次の計算に戻すときだけ。前の値に依存しない属性（`self.y = self.w(x)`）も 1 回分のグラフをメモリに持ち続けるが、伸びはしない。例は正しいが、文は「パラメータから計算したどの属性も伸びる」と読める。`specs/torch.md:149` は「属性に持って次の更新で使うと伸びる」と正しく書いている / 確信度: 中 / 深刻度: 低

## nit

6. `examples/torch_node.py:57`（`self.next.send(self.model(x))`）と `examples/online_learning.py:75`（`self.loss.send(...)`）/ 送る値を grad モードで計算するので、使われないグラフを毎回作る。新しい docs は「勾配の要らない計算は `no_grad` の中で」と勧めているので、例と助言が少しずれる。正しさには影響しない / 確信度: 中
7. `tests/mujou/_core/test_node.py:146` / `GradSetter(leak=False)` は今は、ポートを 1 つ持つ「ほかのノード」としてしか使っていない。勾配の記録は使わないのに、勾配のクラスを流用している / 確信度: 高
8. `docs/ja/user-guide/torch.md:49` / 1 文に「通常の PyTorch と同じ」が 2 回出る（feedback_japanese_style「繰り返しを削る」）/ 確信度: 中
9. 同じワーカーの上の 2 つのノード間で漏れが起きることを確かめていたテストを消した。今あるテストは 1 つのノードで、フレームワークが設定を戻さないことを確かめている。docs の主張（同じワーカーの次のノードに漏れる）は、これと「設定はスレッドごと」から従うので、別のテストは無くてよい。記録として残す / 確信度: 低

## 確認した点（指摘なし）

- `_before_hook` / `BeforeHook` / `BareBefore`: `memory/agents/` と memory の履歴の記述以外には残っていない。`CACHED_HOOKS` の長さと順序は正しい（参照は名前の position で引く）
- テスト: `Backprop` は `enable_grad` 無しで `backward` し、解析解 `.grad == 2` と比べる（旧い実装では落ちる）。`test_on_start_runs_on_a_worker_recording_gradients` と `test_leaves_what_a_hook_set_to_the_next_hook_on_the_thread` は公開インターフェースだけを使い、ワーカー 1 本の仮想タイマーで決定的
- docs の「`no_grad` の外でパラメータをその場で更新するとエラーになる」: requires_grad の leaf では正しい（実行して確かめた: `a leaf Variable that requires grad is being used in an in-place operation.`）。新しいスレッドは、ホストの設定にかかわらず grad モードがオンで始まる（確認済み）
- 局所性: ライブラリの中でその場の書き込みは `load_state_dict` の `copy_` だけで、もともと `no_grad` の下にある。`send` は detach し、`recv` はコピーを返すので、ノードをまたぐグラフは作れない。グローバルな同期や固定のモードは持ち込んでいない
- a0f229f: GPU だけで走るテストから、残っていた勾配オフの確認を消したもの。ほかに CUDA で skip されるテストのうち勾配を確かめるものは残っていない

## 検証結果

- just run: pass（pytest 587 passed / 20 skipped、pyright 0 errors、docs-build ok、pre-commit 全部 pass）。a0f229f の前に走らせたが、a0f229f は CUDA でだけ走るテストの 3 行を消すだけなので、このマシンでの結果は変わらない
