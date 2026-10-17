# node-period 2 巡目（目標を前回の目標 + P で数える・`period` の予約名）レビュー

対象: worktree `.claude/worktrees/node-period`、ブランチ `feature/2026-10-17/node-period`。`git diff ded660b..HEAD`（b30e499..2ac7ecd）と、merge ded660b の衝突の解消（`session.py`・`node.py`・`specs/execution-model.md`・`specs/nodes.md`・`test_node.py`）。

## verdict: approve

must-fix はない。`node_loop` の目標の更新は正しい。境界（ちょうど P 遅れ）、停止と capture、手動モード、周期なしのときに従来と同じになること、のどれにも問題はない。ただし、復元すると目標の位相が失われ、止めずに続けた場合と次の更新の時刻がずれる。これは仕様の書き方どおりの動作だが、ユーザーに伝えて判断を仰ぐべき点として should-fix 1 に挙げる。

## must-fix

なし。

## should-fix

1. **復元すると目標の位相が失われ、止めずに続けた場合より次の更新が遅れる**（確信度: 高（挙動）／中（問題かどうか）、深刻度: 中）
   - 対象: `crates/mujou-core/src/runtime.rs` の `node_loop` 冒頭の `let mut target_ns = last_ns.saturating_add(period_ns);`、`specs/nodes.md:132`（「復元したときは、状態を取った時点の最後の更新から P 後」）
   - 問題: 走っている間の目標は「前回の目標 + P」なので、(last, last + P] のどこかにある。ところが状態の木には `last_update_ns` しか入らず、目標は入らない。そのため復元した実行は目標を last + P に置き直し、少し遅れて更新したノード（last > 前回の目標）の次の更新は、続けた場合より「最後の更新の遅れ」の分だけ後ろにずれる。capture（`state_dict()` の後にそのまま続ける）では目標を保つので、「取得して続ける」と「取得して復元する」で振る舞いが分かれる
   - 根拠（実測、仮想時計、P = 10 ms）: 15 ms に 1 回更新（目標 10 ms から 5 ms 遅れ、次の目標 20 ms）した後に `run.state_dict()` を取った。そのまま続けると +5 ms（20 ms）で Δt = 5 ms の更新が来る。同じ木から別の `VirtualTimer` へ復元すると、更新は +10 ms（25 ms）で Δt = 10 ms になった（scratchpad の `r2/phase.py`）。`specs/execution-model.md` の「状態の取得」は、最後の更新から T までの時間が「止めずに続けた場合と同じく」復元後の最初の Δt に入ることを、木の目的として書いている。周期の位相はこの「同じ」に含まれていない
   - 方向: どちらかに決めて仕様に書く。(a) 現状のまま、`specs/nodes.md` の項目 1 に「目標は木に入らないので、復元すると位相は最後の更新から数え直しになり、止めずに続けた場合より最大で P 近く遅れることがある」と明記する。(b) 各ノードの次の目標を木に入れ（「各ノードの最後の更新の時刻」の行の隣）、`RuntimeConfig` に渡す。平均 1/P を守るという今回の決定の趣旨からは (b) が一貫するが、木の形が変わる。ユーザー判断

2. **`memory/project_node_period.md` の 10 行目が旧方式のまま**（確信度: 高、深刻度: 低）
   - 対象: `memory/project_node_period.md:10`「前回の更新の直前に読んだ時刻 + period まで `Timer::wait_until_ns_or` で待つ」
   - 問題: 12 行目（数え方の決定）や specs と矛盾する。待つ先は「目標の時刻（最初は起点 + P、その後は前回の目標 + P）」になった
   - 根拠: 2ac7ecd でこのファイルの他の行は更新したが、この行は diff の文脈行として残っている

## nit

1. **「平均で P に 1 回」は、更新と起床の遅れが P に収まる間だけ成り立つ**（確信度: 中、深刻度: 低）。対象: `Node` の docstring（"the node updates once every ``period`` on average"）、`RuntimeConfig::period_ns` の doc、`specs/execution-model.md:40`、`specs/nodes.md` の表と「pamiq-core との違い」（「平均の頻度は 1/P に保たれる」）。`on_update` の時間 C と起床の遅れ L の和が P 以上だと、遅れが毎回 C + L − P ずつ積み上がって P に達したところで数え直すので、頻度はおよそ 1/(C + L) に下がる。「間に合っている限り」の一言を足すと正確になる
2. **`specs/nodes.md:135` の「すでに過ぎていたら」**: 実装は `now_ns >= target_ns`（達していたら）。括弧の「P 以上遅れたら」は正しいが、本文の「過ぎていた」は `>` と読める。`specs/execution-model.md:240` の「達していたら」に揃える（確信度: 中）
3. **Δt の下限**: 遅れ d（0 < d < P）の次の Δt は P − d（+ 次の起床の遅れ）なので、d が P に近いと Δt は 1 ns まで小さくなる。連続時間の定式化とは整合する（Δt の和は経過時間に一致し、常に Δt > 0。目標は常に last より後にあるので Δt = 0 の更新は起きない）。ただ `specs/nodes.md:143` の「少し遅れた次の更新では P より短い」は「遅れが P に近いほど短く、0 に近づく」と書いた方が、Δt で割るノードの作者に親切（確信度: 高、深刻度: 低）
4. **`crates/mujou-core/tests/runtime.rs` の `exactly_a_period_late_counts_again_from_the_late_update` のコメント**「the target after it is 4 P either way」: このテストは `>=` と `>` を区別している（`>` なら目標が 3P に残り、3P で Δt = 0 の更新が来て `assert_quiet` で落ちる）。「either way」は、区別していないように読める。「数え直さなければ目標が 3P のままになり、Δt = 0 で続けて回る」と書くと、テストの意図が伝わる（確信度: 中）
5. **`tests/mujou/_core/test_node.py` の `test_a_period_below_a_nanosecond_is_one_nanosecond`**: 1 ns に丸めた場合と、周期なし（`None`）として渡した場合を区別できない（どちらも 1 ns 進めると Δt = 1e-9 で更新する）。区別できないのは 0 ns に丸めてコアに拒否される退行だけ。仕様上どちらでも振る舞いは同じなので、このままでよい（確信度: 高、深刻度: 極小）
6. **`src/mujou/_core/node.py:39-40` の docstring の折り返し**: "once every ``period`` on average (see\n      below). ``dt`` is the\n      time in seconds ..." の行の長さが不揃い（編集の名残り）（確信度: 高、深刻度: 極小）
7. **同じ P のノードは位相がそろい続ける**: 旧方式では起床の遅れのばらつきで位相が自然にずれていったが、新方式では同じ P・同じ起点のノードは毎周期まったく同じタイマーの ns に起き、タイマーの lock を取り合う。正しさには影響しない。性能だけの話なので、M:N の検討の材料として記録する（確信度: 中、深刻度: 低）
8. **巨大な `period` の `OverflowError`（1 巡目の nit 1 のまま）**: `Node(period=1e12)` は `__init__` を通り、`runtime.start` で `OverflowError: int too big to convert` になる。周期と結びつかないメッセージだが、何も起動せずに戻る経路（`except Exception` の片付け）に乗るので実害はない（確信度: 高、深刻度: 低）
9. **予約名の検査の時点**: `period_ns` は実行の開始で検査するので、走らせないクラスや、別の実行経路を足したときに漏れうる。`Node.__init_subclass__` で検査すれば定義の時点で落とせ、session 側の呼び出しも要らなくなる。今の位置でも `restore`・入力の書き込み・`Runtime` の生成より前で、何も始まらずに `TypeError` になり、ネットワークは走らせ直せる（`start_once` が `_ran` を立てない）ので、現状で問題はない（確信度: 中、深刻度: 低）

## 確認して問題がなかった点（依頼された観点のみ）

- `target_ns` の更新: 更新した時刻 now は常に now ≥ 旧目標。新しい目標は「旧目標 + P」か「now + P」で、どちらも now より後になる。そのため次の Δt ≥ 1 ns、Δt = 0 の更新は起きない。P = 1 ns（周期なし）では、どちらの分岐でも目標は now + 1 になり、従来の `now <= last` で待つのと同値になる。手動モードは `last_ns + 1` を使い、`target_ns` は読まない
- 停止・capture・再開: 停止中はタイマーが凍結され、`target_ns` はループのローカル変数なので保たれる。再開後は残りだけ待つ（`a_capture_does_not_move_the_target` で確認）
- オーバーフロー: `saturating_add` の端（約 584 年）だけで、1 巡目の nit 7 と同じ
- `period_ns(node)`: `inspect.getattr_static(type(node), "period")` は、MRO のどこかでクラス属性・property・mixin による置き換えがあれば `vars(Node)["period"]` と食い違うので検出できる。インスタンス属性への代入は property（setter なし）で `AttributeError` になる。session が読むのは `node._period` なので、検査を迂回して値が入ることはない。丸めの `floor + (frac >= 0.5)` は、正の値で Rust の `f64::round`（0 から遠い側）と一致する（`ns - whole` の引き算は誤差なく計算できる）
- session での検査の時点: `Session.start` の先頭、`_restore`・入力の書き込み・`Runtime` の生成・`_RUNNING` への追加より前。失敗しても何も始まらない。`Session.__init__` のポートの bind は残るが、復元の失敗と同じ既存の経路で、走らせ直すときに上書きされる
- merge の解消: `period_ns=periods` は `**(restored or {})` と別の引数で渡しているので、復元しない実行にも周期が渡る。`node.py` の `on_update` の行と予約名の一覧、`specs/execution-model.md` の手動モードの行（「周期は使わない」）、`test_node.py`（`virtual`・`start` の fixture）の解消はどれも妥当
- テストの決定性: 仮想時計のテストは時刻が `advance` でしか進まず、目標の計算は区切りで読んだ時刻だけに依存するので、スレッドの起き方によらず決まる。実時計の 2 本の上限（更新数 ≤ 経過時間 × scale / P）は、更新が P ずつ離れた目標でしか起きず、タイマーの経過が実時間の経過を超えないので、負荷があっても健全。下限は期待値の 1/2（scale 1）と約 1/4（scale 4）で、遅れが P 未満なら頻度が落ちない新方式では余裕が大きい
- 成果物の汚染（`</content>` など）: diff の対象ファイルに無し

## 検証結果

- just run: pass（pre-commit 全通過、cargo test の `period::*` 14 本を含め全通過、pytest 462 passed / 15 skipped、pyright 0 errors、mkdocs build --strict 通過）
- 実測は仮想時計のスクリプト 1 本だけ（scratchpad の `r2/phase.py`。should-fix 1 の数値）。負荷試験はしていない
