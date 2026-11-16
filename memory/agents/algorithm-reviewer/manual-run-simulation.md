# algorithm-reviewer: ManualRun をシミュレーション用の実行に（Jacobi の step(dt)）

対象: ブランチ feature/2026-11-16/manual-run-simulation の未コミットの変更（計画書 `memory/agents/implementation-planner/manual-run-simulation.md`）。観点: ノード間の局所性・非同期性の不変条件、更新手法を基盤に焼き付けないこと、pc がバリアに頼っていないか、オラクルの呼び方、`specs/execution-model.md` の理論の記述。更新式（`src/mujou/pc/`）そのものは diff に無い。

## verdict

**approve（should-fix 付き）**。確信度: 高。

自由実行（`mujou.run`）の経路に、保留もバリアも入っていない。ノード・Driver・ポートは自分が手動の実行にいることを知らない。pc の実装はバリアに頼っていない。直すべき点は文書だけ: 「Jacobi」という語の使い方が pc のオラクルと新しい `step(dt)` で食い違っている（S1）。pc の「credit は 1 周期に 1 ホップ」は逆順の逐次でしか成り立たず、新しい既定の `step(dt)` では 2 ステップに 1 ホップになる（S2）。`specs/execution-model.md` の「Jacobi 型の陽的 Euler に一致」は、条件を書かないと言い過ぎになる（S3）。どれも数値で確かめた。

## must-fix

なし。

## should-fix

### S1. pc のオラクル（`HandChain.from_next` と `Reference`）を「Jacobi」と呼ぶのは、新しい `step(dt)` の Jacobi と食い違う（確信度: 高）

- `step(h, *chain.from_next())`（出力側から逆順）では、層 i は層 i+1 が**同じステップで**送った `g_in` を読む。前向きの μ は前のステップの値、後ろ向きの `g_in` は同じステップの値なので、前向きは Jacobi、後ろ向きは Gauss-Seidel の混合になる。`_reference.py` の `Reference.step` も、同じステップで計算した `g_next` を次の層がそのまま使っており（`g_next = g_in`）、この混合と同じ
- 実測（scratchpad `jacobi_check.py`。`TestJacobi` と同じ鎖 widths [4, 5, 5]、`learn_interval=4H`、float64）: `step(h)` と `step(h, *from_next())` の r と λ の最大差は、1〜3 ステップでは 0（credit がまだ届いていない）、5 ステップで 2.3e-2、50 ステップで 4.4e-3、2000 ステップで 3e-10。不動点は同じで、途中の軌道は違う
- 実装した `specs/execution-model.md` と `ManualRun.step` の docstring は正しい。「値が逆向きにも流れるなら、どの順とも一致しない」と書いている。ただし、次の箇所はまだ `from_next` を Jacobi と呼んでいる
  - `tests/mujou/pc/test_layer.py:89-90`（`from_next` の docstring "each reads what the others sent in the previous step (a Jacobi update)"）、`:157-170`（`class TestJacobi`、`test_stepping_from_the_output_is_the_jacobi_update`、コメントの "reads only values of the previous step everywhere"。後ろ向きについては誤り）
  - `tests/mujou/pc/_reference.py:189-190`（"every node moves at once from the values of the previous step (a Jacobi update)"）
  - `specs/pc.md:374`（テスト 2「出力側から回す同期の step が…参照実装の Jacobi 更新と 1e-12 で一致」）。`specs/pc.md` の 45・50・67・68・71 行の「Jacobi」と `specs/research/pc-experiments.md:14,73,98` も、測ったのが出力側からの逐次なら同じ
  - 計画書の「理論上の対応」1 項目めの「従来 pc のテストが `step(h, *chain.from_next())` で作っていた Jacobi が既定になる」も誤り（計画書は直さなくてよいが、この前提で書いた箇所が残っていないか確かめる）
- 直し方: これらを「出力側からの逐次（前向きは前のステップの値、credit は同じステップで送られた値）」と呼び直す。たとえば `TestJacobi` → `TestFromTheOutput`。オラクルとしての価値は変わらない（Reference と from_next の一致は今も正しい）。Jacobi の名前を残すなら、`Reference` を `g_in` も 1 ステップ遅らせる形にし、テストを `step(h)` で回す（計画書の「pc: step(h) と step(h, *from_next()) が bit 一致」というテスト観点は、この数値から成り立たないので採らない。実装もそのテストは足していない。足したのは workers 1 と 4 の一致だけで、これは正しい）

### S2. 「credit は 1 周期に 1 ホップ戻る」は出力側からの逐次の性質で、既定の `step(dt)` では 2 ステップに 1 ホップ（確信度: 高）

- `pc.Layer.on_update` は、更新の**前**の δ から `g_in` を計算して送り（`_predict`、`layer.py:246,269-275`）、そのあと r と λ を進める。`step(dt)` の Jacobi では、層 i+1 がステップ k で受け取った g を δ に反映して `g_in` に載せるのはステップ k+1 で、それを層 i が読むのはステップ k+2 になる。出力側からの逐次では、層 i はステップ k+1 の中でそれを読める
- 実測（scratchpad `wavefront.py`、5 層 + Output、重み固定の最初の数ステップで λ が初めて 0 でなくなるステップを、入力側から並べた）: `step(h)` は [16, 14, 12, 10, 8]、`from_next` は [11, 10, 9, 8, 7]。1 ホップに 2 ステップと 1 ステップ。前向きはどちらも 1 ステップに 1 ホップなので、往復は `step(h)` で約 3 × ホップ数 × 周期、逐次で 2 × ホップ数 × 周期。自由実行では位相によって 1 ホップあたり 1〜2 周期の間
- 影響を受ける記述: `specs/pc.md:66`（「credit は 1 周期に 1 ホップ進む」）と `:196`（learn_interval の目安 `2 × ホップ数 × period`）、`src/mujou/pc/layer.py:87-88,176` と `src/mujou/pc/chain.py:86` の docstring、`docs/{en,ja}/topics/pc.md:59,60,68,147`、skill `pc-alm` 7.2 の「credit は 1 周期に 1 ノード戻る」。これらは正しさ（不動点）の主張ではなく目安なので must-fix にはしない。ただし利用者は今後 `run_manually(...).step(h)` でシミュレーションするので、目安を `step(dt)` で守ると credit が戻りきる前に学ぶことになる
- 直し方: 目安を「往復はスケジュールに依る。出力側からの逐次で 2 × ホップ数、`step(dt)` で 3 × ホップ数周期」に書き換える。あるいは更新手法の側で、`g_in` を緩和した後の δ から送るように変える（pc の設計変更で、Reference と既存の数値の取り直しが要る。範囲外なのでユーザーの判断）
- pc がバリアに正しさを頼っていないことは変わらない。どちらのスケジュールでも同じ不動点に向かう（S1 の 2000 ステップの値）

### S3. 「全ノードを固定の Δt で回すと、Jacobi 型の陽的 Euler に一致する」は、メッセージが送り手の更新後の状態だけで決まるときに限る（確信度: 中〜高）

- `specs/execution-model.md` の「オラクルとしての使い方」と計画書の「`step(dt)` は…Jacobi 型の陽的 Euler にそのまま一致する」。`step(dt)` が厳密に一致するのは、「ノードの状態とスロットの中身」を合わせた拡大系の Jacobi 更新である。各ノードはステップ k で自分の状態と、ステップ k−1 に公開された値を読む
- 連続時間系 $\dot x_i = f_i(x_i, x_{\text{隣}})$ の陽的 Euler に一致するのは、各ノードが**更新後の自分の状態の関数**を送るときだけ（`Leaky` の x、`noisy_jacobi`）。送る値が同じ更新で読んだ値を中継する（pc の μ = f(h_in)）か、更新前の状態から計算する（pc の `g_in`）なら、1 ホップごとに 1 ステップの遅れが入る。遅れのある離散化になり、不動点は同じでも軌道は ODE の陽的 Euler と違う。skill `pc-alm` 7.1 の「全ノードを固定の Δt で同期的に回すと、3 節の更新に一致する」も同じ条件付き
- 直し方: `specs/execution-model.md` の当該文を「各ノードがステップ開始時の自分の状態と、前のステップで公開された値だけから更新する同期の離散アルゴリズム（Jacobi）に一致する。各メッセージが送り手の更新後の状態だけで決まるなら、連続時間系の陽的 Euler そのものになる。そうでないなら、メッセージ 1 つにつき 1 ステップの遅れを持つ離散化になる」の趣旨にする

### S4. 逆順の逐次と `step(dt)` が一致する条件の書き方（確信度: 高。内容は正しく、言い方の精度の問題）

- `specs/execution-model.md` の「出力方向にだけ送る chain なら、出力側から逆順に渡すと一致」「値が逆向きにも流れるなら、どの順とも一致しない」と、`ManualRun.step` の docstring の同じ文
- 正確な条件: 「異なるノードの間のメッセージの流れに閉路が無い（DAG）なら、受け手を送り手より先に更新する順（トポロジカル順の逆）が `step(dt)` と一致する」。chain は特別な場合で、前向きにだけ送る分岐・合流（DAG）でも一致する（pc の網は credit が逆向きに流れるので、形によらず DAG にならない）。自己ループ（1 つのポートを自分につないだもの）は条件に影響しない。逐次でも 1 ノードずつのグループで保留するので、自分の送信は自分にも見えない（下の N1）
- 「どの順とも一致しない」は一般の場合の話で、閉路を流れる値がそのステップで変わらなければ一致する。S1 の実測でも、credit が届く前の 1〜3 ステップは bit 一致した。「一般には一致しない」とするのが正確
- `tests/mujou/test_execution.py` の `test_no_order_of_a_loop_is_a_step_of_every_node` は、値が変わる閉路で確かめているので正しい

## nit

### N1. 手動の実行では、ノードは同じ hook の中で自分の送信を読めない（逐次の `step(dt, *nodes)` でも）。自由実行とは違う（確信度: 高）

- `Port.connect` は自分自身への接続を許す（`src/mujou/port.py:70`）。実測（scratchpad `selfloop.py`）: 自分につないだポートに send してすぐ recv すると、`step(dt)` でも `step(dt, node)` でもステップ前の値が返った（`[0.0, 1.0]`）。自由実行と変更前の逐次では、送った値が返る
- 非同期のモデルでは古い値を読むことが許されているので、不変条件には反しない。Jacobi としても筋が通る。ただし逐次の振る舞いが変わったので、`ManualRun.step` の docstring か `specs/execution-model.md` の「送信の保留と公開」に「ノード自身の送信も、そのノードのステップが終わるまで見えない」と 1 行足すとよい

### N2. 自由実行のデータ構造に、手動のための空の部品が載っている（確信度: 高、影響は無い）

- `Slot.staged`（全スロット、`InputGroup` のスロットを含む）、`Outlet.staging: Option`、`Lifecycle.granted: Vec<bool>`。自由実行では `None` か未使用で、`Outlet::send` の分岐は `Option` を 1 回見るだけ。振る舞いは変わらない。概念が漏れているのは型とコメントだけで、計画書が採らなかった `RunClock` に持たせる案よりは閉じている。直す必要は無い
- `Outlet` の `staging` と `RunConfig.manual` が同じ `Staging` であることは型で保証されていない（`execution._start` が同じ値を 2 か所に渡すだけ）。private な口なので nit

## 不変条件の確認（指摘なし）

- **自由実行に漏れていないか**: `execution._start` は `kind.manual` のときだけ `_native.Staging()` を作る。`Outlet::send` は `staging` が `Some` かつ開いているときだけ保留し、自由実行の outlet は `None`。`Controller::step` は手動のときしか呼ばれず（`Runtime::step` が手動でない実行を断る）、`Lifecycle::advance` の自由実行の枝（`Command::Run if !self.manual`）は `granted` の検査より先に返る。バリアは制御スレッドの `step` の中だけにある
- **ノード・Driver・メッセージが手動を知らないか**: `src/mujou/node.py`、`driver.py`、`port.py`、`codec.py` は diff に無い。保留はスロットの公開の時点だけを変え、ノードが読むのは今までどおり自分の状態と隣のスロットだけ。メッセージの型・スケジューラに更新手法固有の量は入っていない（`Staging` が持つのは不透明な `Arc<Message>`）
- **保留の正しさ**: 開閉は制御スレッドだけが行い、開くのは許可と wake の前（同じ lock の下）、公開は全ノードの `end_step` を lock の下で見た後。ノードの送信と commit の前後関係は mutex で順序づく。スロットごとの保留は最後の 1 つで、Target::All は同じメッセージを各スロットに積む。失敗時に開いたまま捨てるのは、失敗後のホストの呼び出しがすべて `NetworkError` になる契約の内側。`Staging → Vec<Arc<Slot>>` に循環は無い
- **非同期のモデルの中にあるか**: `step(dt)` の結果は「各読み出しが 1 ステップ分古い値を返す」実行で、古い値を許す非同期のモデルで起こりうる実行の 1 つ。バリアは時間が止まっている間だけで、例外の条件（停止と同じ理由）と「更新手法の正しさをこれに頼らない」の但し書きが CLAUDE.md・skill・spec にそろっている
- **pc がバリアに頼っていないか**: `src/mujou/pc/` は diff に無い。`Layer.on_update` はそのとき読める最新値で更新する形のまま。pc のテストのうち `step(h)` を使うのは `test_chain.py`（`Chain` と手組みの比較、workers 1 と 4 の比較）で、両辺が同じスケジュールなので、Jacobi に結果を頼っていない。`test_layer.py` のオラクルは今も `from_next` の逐次で回している
- **テストの数理**: `noisy_jacobi` は「前のステップで送られた値で全ノードを更新」を手で書いたもので、`Noisy` が更新後の x を送るなら陽的 Euler に一致する（S3 の条件を満たす）。閉路の期待値 `[(1, 10), (11, 11), (12, 21)]` は $a_k = b_{k-1} + 1$、$b_k = a_{k-1} + 10$ から手で確かめた。許容誤差は `torch.equal`（bit 一致）で、緩すぎない

## 検証結果

- 数値（scratchpad、`/tmp/claude-1000/-Users-geson-Projects-mujou/a3e7d2aa-584c-45ce-818a-3e688fdf9f3f/scratchpad/`）:
  - `jacobi_check.py`: pc の鎖で `step(h)` と `from_next` の差（S1）
  - `wavefront.py`: credit の波面の速さ（S2）
  - `selfloop.py`: 自己ループの読み出し（N1）
- `just type`: 環境の問題で走らなかった（pyright の nodeenv の取得に失敗）
- `just format` / `just docs-build`: 走らせていない。format はファイルを書き換え、docs はほかの agent が並行して編集中のため
- `just test`: 通過（cargo test 146 passed / 0 failed、pytest 951 passed / 88 skipped。skip は CUDA など）

## 照合メモ

- Layer のステップ k（Jacobi）: 読む h_in = μ_{i−1}(k−1)、g = g_in^{(i+1)}(k−1) = F(μ_i(k−2), δ_{i+1}(k−1 の開始), θ_{i+1}(k−1))。送る g_in^{(i)}(k) は δ_i(k の開始) から
- 出力側からの逐次: g = g_in^{(i+1)}(k) = F(μ_i(k−1), δ_{i+1}(k の開始), θ_{i+1}(k))。`Reference.step` の `g_next` と同じ。違いは g の 1 ステップの遅れだけ
- 波面: Output が k=1 で g を出す → L−1 は Jacobi なら k=2 に読んで δ を動かし、k=3 にそれを `g_in` に載せる → L−2 は k=4 に読む。逐次なら L−1 は k=1 に読み、k=2 に載せ、L−2 は k=2 に読む。実測の 2 ステップと 1 ステップの間隔に合う
- 逐次 = Jacobi の条件: 辺（送り手 → 受け手）ごとに、受け手が送り手より先に更新されれば、受け手は前のステップの値を読む。全辺で成り立つ順があるのは、異なるノード間の辺が DAG のときだけ
