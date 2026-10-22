# pc-optimizers（orchestrator のメモ）

ブランチ `feature/2026-10-22/pc-optimizers`（worktree `.claude/worktrees/feature-pc-optimizers`、main の 8c20e04 から）。

## 背景（実測、2026-10-22、3.14t、torch 2.14.1）

- `torch.optim.SGD` / `Adam` のコンストラクタが `torch._dynamo`（約 900 モジュール）→ `triton._C` を import し、GIL が有効に戻る（プロセス全体で戻らない）。SGD でも同じ。`torch.optim` の import だけでは戻らない。関数版 `torch.optim.adam.adam(...)` は戻さない
- Adam の β は 1 ステップあたりなので、実効の時定数が学習間隔に依存する。`state["step"]` は float32 テンソルで 2^24 回で増えなくなる

## ユーザーの決定（2026-10-22）

1. τθ を `TimeConstants` から optimizer に移す（`TimeConstants` は h と lam だけ。破壊的変更、互換は残さない）
2. API は「設定オブジェクト + bind」: `pc.AdamW(tau=..., ...)` は frozen dataclass、ノードが `opt.bind(params)` で状態を持つ束縛を作る。束縛は `step_(grads, elapsed)` と `state_dict` / `load_state_dict`
3. 段階 1 の optimizer は SGD と AdamW
4. h と λ の optimizer は段階 2（研究から）。今回は実装しない。specs に方針だけ書く
5. worktree で進め、PR まで。docs は doc-teacher-student + yomiyasu

## 設計の要点（チャットで合意した案）

- 時間で進める: `a = 1 - exp(-Δt/τ)`、`m ← m + a(g - m)`。bias 補正はカウンタの代わりに質量 `c ← c + a(1 - c)`、`m̂ = m / c`
- Δt 一定なら β = e^{-Δt/τ}、c = 1 - β^t で `torch.optim.AdamW(lr=Δt/τθ)` と一致 → オラクル
- 1/B はノードが掛けてから渡す。optimizer はバッチを知らない
- `functional.learn_` は削除して optimizer に置き換え
- torch.optim を import したテストは GIL を戻すので、期待値は手で書いた式、torch.optim との一致は subprocess の 1 本だけ
- 局所性: optimizer は自分のノードの param だけを見る

## レビューの裁定（code-reviewer と algorithm-reviewer、どちらも approve）

採用（simplifier へ）:

- `eps > 0` を必須にする（両者の should-fix 1。K31 が防ぐのは NaN なので揃える）
- `step_` の `elapsed` の説明を「その歩みが重みとする時間（`Tick.elapsed`、捨てた時間を含まない）」に直す。spec も
- chain のテストで moment の shape をノードの param の shape と比べる
- mask が全 False で学ばないテストを SGD と AdamW で回す（先にラベルのある入力でモーメントを 0 から離す）
- AdamW の docstring の例を Adam として妥当な値に（`tau = learn_interval / lr`）
- spec に Adam の正規化の性質（緩和しきらない δ の残差が sign 程度の歩みになりうる、未測定）を短く
- 段階 2 の「不動点は保てる」に条件（勾配の零点だけを停留点に持つ則）を足す
- θ の離散化の説明を「陽的 Euler」から実際の形に
- テストの nit（v と c_v も確かめる、long step の検査）、docstring の折り返し、example のコメント・変数名、Layer の Args に自作の optimizer、feedback_alignment の定数名

却下:

- `load_state_dict` の shape 検査: 先に module の load が断るので YAGNI（G3）
- `elapsed == 0` の早期 return で勾配の数を検査しない: 0 秒の歩みは恒等で害がない
- weight decay の除外の手段: 使い手が出てから

## 進行

- [x] planner
- [x] spec-test-author / plan-implementer
- [x] reviewer（code + algorithm）
- [x] simplifier（`eps > 0`、elapsed の説明、換算を「2 回の学習の間の時間 Δt / lr」に、テストの強化）
- [x] docs（doc-teacher-student + yomiyasu）: tutorial 5（R1 7/7、保留 3/3）、topics/pc（R2 8/8、保留 3/3）、pc-concepts（R2 10/10、保留 3/3）。ja は 3 ページとも yomiyasu で推敲。質量 c の「重み」を θ と紛れない言い方に（en・ja・docstring）
- [x] just run（905 passed）→ commit → PR

## docs の段で見つけたこと

- `Tick.elapsed` は前に学んでからの時間なので、1 入力に 1 回学ぶなら見せる時間で `learn_interval` ではない。最初の docstring の `tau = learn_interval / lr` は誤りだった（teacher が発見）
- 学んだ入力の後で入力が切り替わっても elapsed は捨てないので、1 バッチに何回学んでも elapsed の和は見せる時間になる（clock.py で確認）
