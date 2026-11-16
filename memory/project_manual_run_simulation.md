---
name: project-manual-run-simulation
description: 手動の実行 ManualRun はテスト用でなくシミュレーション用の正式な実行（2026-11-16 のユーザーの決定）。入口は mujou.run_manually、mujou.testing は廃止。step(dt) は並列の Jacobi、step(dt, *nodes) は逐次。周期は無視。ステップ境界のバリアは ManualRun だけの例外
metadata:
  type: project
---

2026-11-16 のユーザーの決定。実時間で sleep を挟む `mujou.run` は現実世界へのデプロイ用、実時間のタイマーを回さず計算能力いっぱいで進める `ManualRun` はシミュレーション用で、どちらも正式な機能とする。クラス名 `ManualRun` は変えない。

- 入口は `mujou.run_manually(root, *, timer=None, inputs=None, state=None, workers=None) -> ManualRun`。`ManualRun` と一緒に `mujou` 直下から公開する。`mujou.testing` は廃止（0.x の破壊的変更として単に消す）。既定のタイマーは新しい `VirtualTimer`（`state` を渡したなら保存した時刻から）。`VirtualTimer` は `mujou.time`
- `step(dt)`（ノードなし）: タイマーを `dt` 進め、全ノードを worker で並列に 1 回ずつ更新する。読む値はステップ開始時の値（Jacobi）で、送った値はステップの全ノードが終わってから公開される。結果は決定的
- `step(dt, *nodes)`: 渡した順に逐次（Gauss-Seidel）。同じ step に 2 回渡したノードは 2 回目を飛ばす
- 周期（`period`）は無視する
- ステップ境界はグローバルなバリアだが、停止と同じくタイマーが止まっている間だけのものなので、不変原則の例外として `ManualRun` にだけ認める。更新手法の正しさをこれに頼らない。自由実行にはバリアも保留も入れない
- torch の intra-op スレッドとワーカーの CPU の取り合いは docs で注意するだけ
- mujou 自身のテストと利用者のテスト・シミュレーションは同じ口を使う

**Why:** 実時間で sleep を挟む学習はシミュレーションに向かない。シミュレーションでは計算能力をフルに使いたいので、ステップの中の更新を並列にした。テスト用と利用者用で別の口を持つと片方だけが古くなる（2026-10-16 の決定を引き継ぐ）。

**How to apply:** テストは `mujou.run_manually` と `step(dt)` で書き、順が要るときだけノードを渡す。設計の正本は `specs/execution-model.md` の手動の実行の節。
