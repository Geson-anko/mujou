# core-runtime WP-A（スロット・バッファプール・DLPack）の監査

対象: worktree `.claude/worktrees/core-slots`、ブランチ `feature/2026-10-16/core-slots`（`aa38fe9`）、`chore/2026-10-15/core-scaffold..HEAD` の差分。基準は skill `locality-and-async`、`docs/execution-model.md`、`docs/messages.md`、`docs/network.md`、計画書の WP-A。

## verdict

**approve**（確信度: 高）。must-fix は無い。should-fix 3 件はいずれも WP-A のコードの誤りではなく、仕様文の約束の範囲と、WP-D に渡す前提条件の明文化。

- 書き手 1 つ・読み手 N の最新値: 満たす。値は不変の `Arc<Message>`、書き込みはプールから取ったバッファに詰めてから `ArcSwap::store` で差し替える。読み手は lock を取らない
- 書き手は読み手を待たない: 満たす。書き手が取りうる lock は (1) プールの空きリスト（push/pop 1 回の臨界区間。読み手がバッファを手放すときも同じ lock）、(2) 待ち手がいるときだけ `Waiters.lock`（待ち手の臨界区間は `ptr_eq` の確認だけ）。待ち手がいなければ SeqCst fence と atomic の読み出し 1 回
- 版番号の一斉切り替え: 1 回の書き込みのうち、どのスロットの切り替わりも `version.store(V+1, Release)` の 1 点に線形化される（照合メモ 1）。ただし 1 つのノードが 2 つのスロットを順に読むと、その間に書き込みが 1 回終わることはありうる（should-fix 1）
- 時刻: `Outlet` は fill の後に 1 回だけ時刻を取って全スロットに同じ `Arc` を置く。`InputGroup::write` も全 fill の後に 1 回取り、全スロットで同じ時刻になる。時刻の源は常にグローバルタイマー（bindings は `time::global()`）。停止中の書き込みは凍結時刻になる
- 学習則の焼き付け: 無い。`Schema` はフィールド名・shape・DLPack dtype だけで、`Target` も All/Channel だけ。PC-ALM の量（λ、$W^\top\varepsilon$）も教師の概念も入っていない
- 局所性: 境界では読みも書きも必ずコピーする（書くときは `copy_from` でプールへ、読むときは `BufferView` を貸すので受け手が `.clone()` する）。計算グラフは構造上つながらない

## must-fix

なし。

## should-fix

1. **`docs/network.md` 160 行の約束が実装（と 164〜169 行の仕組み）より強い**（確信度: 高）
    - 文面は「新しい入力と古い教師信号の組を読むノードが、一瞬も存在しない」。実装が保証するのは「どの瞬間を切り取っても、外から書けるスロットは同じ write の値で揃っている」こと（各 `load` は `version` の store を切り替え点とする 1 点に線形化される）
    - 1 つのノードが `on_update` の中で教師 → 入力の順に読み、その間に `write` が 1 回終わると、(教師 k, 入力 k+1) を読む。入力 → 教師の順なら (入力 k, 教師 k+1)。どちらの順で読んでも、2 回の読みの間に終わった書き込みは混ざる。Rust のテスト `input_group_switches_all_slots_at_once` が確かめている `b >= a`（後に読んだ方が古くない）もこの性質であり、計画書のテスト観点と一致している
    - skill `locality-and-async` の「複数のスロットをまとめて読んでも、一貫したスナップショットにはならない」とは整合する。直すのは docs の文面の方がよい。案: 「どの瞬間にも、新しい入力と古い教師信号が並んで見えることはない。1 回の更新で複数の Port を順に読むと、その間に次の write が終わることはある。読んだ値の `time` を比べれば同じ write か判別できる」
    - スナップショットの読みが本当に要るなら `InputGroup` に、V を読む → 全 Pair を読む → V を読み直し、一致しなければやり直す、という読みを足す手はある（書き手は待たない。読み手はリトライする）。ただし、それで揃うのは外から書けるスロットだけで、ノードのスロットとは揃わない。要求されるまで足さないほうがよい
    - WP-D への注意: 計画のテスト「1 回の write で書いた複数の Port の値を、ノードは同じ時刻で読む」は、手動モード（書き込みとステップが交互）でだけ決定的に成り立つ。自由実行で書くと flaky になる

2. **`Outlet` の「書き手 1 つ」は型で強制されておらず、並行に `send` すると Target::All の不変条件と時刻の単調性が壊れる**（確信度: 挙動は高、重大度は中）
    - `send_with(&self, ..)` で、`PyOutlet` は `frozen`（共有参照で複数スレッドから呼べる）。2 スレッドが同時に `send` すると、時刻 t1 < t2 の 2 通が逆順に置かれることがある。さらに Channel 群と outbox への置き方が交錯すると、outbox と Channel が別の Message を持ったまま止まる（「全員に送ると全スロットが同じ Arc」が崩れる）
    - 計画では WP-D の Port が所有ノードのスレッドとの照合で防ぐことになっているので、WP-A のコードの誤りではない。ただし前提条件なので `Outlet` / `send_with` の doc comment に「呼び出しは 1 スレッドから直列に。並行に呼ぶと時刻の順と Target::All の一致が崩れる」と書いておく。コストの低い保険として、debug ビルドだけ `AtomicBool` で再入を検出する方法もある
    - 同じく `Slot::restore` / `InputSlot::restore` も、読み手側のハンドル（`channel(k)` で得た `Arc<Slot>`、Python の `Slot`）から呼べる、2 本目の書き込み経路である。停止中（タイマーが凍結され全ノードが区切りで止まっている間）にだけ呼ぶ前提を doc comment に書く

3. **`wait_newer` を全スロットに持たせた逸脱で、ノードが隣を待つ API が手の届くところに出た**（確信度: 中）
    - 判断ログの理由（Python の `Slot` 型が外向きと Channel で共通）は妥当で、書き手のコスト（待ち手がいなければ fence と atomic 読み 1 回）も許容範囲
    - ただし Channel のスロットで `wait_newer` を使えば、ノードが隣のノードの書き込みを待てる。これは「他のノードを待たない」に反する使い方になる。WP-D の Port（ノードに見せる面）では `wait_newer` を公開せず、ホストの `subscribe` だけが使うことを、判断ログか `docs/messages.md` に一文残す

## nit

1. `Outlet::send_with` が付ける時刻は「fill（コピー）が終わった時刻」で、ノードが `on_update` で値を計算した時刻より少し後になる（自由実行時。停止中や手動モードでは一致する）。`docs/messages.md` の「書いた時点の時刻」とは合っている。読み手が鮮度を判断する用途では差は小さいが、更新の時刻で印を付けたいという要求が出たら `send_with` に時刻を渡す口を足す（確信度: 中、優先度: 低）
2. `BufferView` は READ_ONLY を立てずに共有バッファを貸すので、`torch.from_dlpack(view)` を clone せずにその場で書き換えると、同じ Message を持つ他の読み手の値が変わる（「状態の所有」の不変条件 4）。計画どおりの判断で、`_core` は private なので許容する。WP-D の Port の recv で必ず clone していることを、D のテストでも確かめる（`test_reader_changing_its_tensor_does_not_change_other_reads` 相当を Port の面で）（確信度: 高）
3. GIL ありのビルドでは `Outlet.send` / `InputGroup.write` のコピーが GIL を持ったまま走るので、大きなメッセージでは他の Python ノードのスレッドが待つ。`Borrowed` が `Bound<'py>` を持つため、`py.detach` で囲むには生ポインタだけを取り出す形に組み替える必要がある。測ってから判断すればよい（確信度: 中、優先度: 低）
4. 1 つのネットワークに `InputGroup` が複数できると、版番号による一斉切り替えはグループ内でしか効かない。WP-D の `Session` ではネットワークに 1 つだけ作ることを確認する（`docs/network.md` 164 行「ネットワークは版番号 V を 1 つ持つ」）（確信度: 高）

## 検証結果

- `cargo test --workspace`: pass（slot/dlpack を含む mujou-core の単体 25 件、time 23 件）
- `uv run pytest -q`: 80 passed
- `uv run pyright`: 0 errors
- `cargo fmt --all --check`: ok
- `just format`（pre-commit run -a）は worktree のファイルを書き換えうるため実行しなかった（「コードは修正しない」を優先）。`ruff format --check .` は docs/*.md の Python コードブロックを差分として出したが、これは今回の差分に含まれない既存ファイルの話で、pre-commit の設定ではその対象になっていない可能性が高い
- ミューテーション確認（書き込みで version を先に進めると `input_group_switches_all_slots_at_once` が落ちるか）は、scratchpad でのコピーとビルドの許可が下りず実行できなかった。照合メモ 2 の手計算では検出できる

## 照合メモ

1. **版番号の線形化**（`InputGroup::write` / `InputSlot::load`）
    - write j（lock 下）: `V = j-1` を読む → 各スロットに `Pair{current: resolve(old, j-1), next: (j, msg_j)}` を置く → `version.store(j, Release)`
    - read: `pair = load()`（時刻 tp）→ `v = version.load(Acquire)`（時刻 tv > tp）→ `resolve(pair, v)`
    - tp で見えた pair が write j のもの（write j+1 の store より前）なら、tp での version は j-1 か j。`v >= j` なら msg_j を返し、[tp, tv] の間に version = j の瞬間がある。`v = j-1` なら msg_{j-1} で、tv の瞬間の値と一致する。よって各読みは [tp, tv] 内の 1 点での「version に対応する値」を返し、切り替え点はスロットによらず version の store 1 点になる
    - A → B の順に読んで `b >= a` になること: a = k を返したなら、write k の B への store は version k の Release（または write k+1 の Pair の Acquire 読み）より前に起きている。そのため B の Pair の読みは write k 以降のものを見て、b >= k になる
    - write k+1 の `current` は lock 下で `resolve(pair, k)` から取るので、前の次の値を正しく繰り上げる。同じスロットが 1 回の write に 2 度現れても、後の値が勝つだけで不整合にはならない
2. **version を先に進める変異の検出**: version を k にしてから Pair を置くと、A の Pair（write k、次 k）と `v = k` から k を読み、続く B の Pair がまだ write k-1 のもの（次 k-1）なら k-1 を読む。`b < a` になるので、10^5 回のループで捕まる見込みが高い
3. **通知の取りこぼしが無いこと**（`Waiters`）: 書き手は store → SeqCst fence → count を読む。待ち手は count++ → SeqCst fence → lock → `unchanged()` → 寝る。2 つの fence の SeqCst 順により、「書き手が count > 0 を見る」か「待ち手が新しい値を見る」の少なくとも一方が成り立つ。前者の場合、書き手が lock を取ってから notify するので、確認から寝るまでの間（lock を持っている）に通知がすり抜けることはない
4. **学習則の焼き付けの走査**: `slot.rs`・`dlpack.rs`・bindings に、λ・ε・teacher・phase・epoch・train/eval に当たる識別子や分岐は無い
