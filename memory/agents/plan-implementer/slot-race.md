# slot-race（スロットが別のスロットのメッセージを返す不具合）の調査記録

ブランチ `fix/2026-10-16/slot-race`（`feature/2026-10-16/timers` の上）。skill `solo-dev-cycle` で 1 人で回した。

## 症状

- `slot::tests::readers_see_whole_messages_in_order` が約 2% で落ちる。`index out of bounds: the len is 1 but the index is 1`、まれに SIGSEGV
- 単独では落ちず、ほかの slot テストと並行に走るときだけ落ちる。PR #8（スロット）の時点からある

## 原因

`arc-swap` 1.9.2（最新版）の既定の strategy（`HybridStrategy`）にある、別の `ArcSwap` をまたぐ ABA。mujou のコードの誤りでもテストの誤りでもない。

arc-swap の読み手は、読んだポインタを「借り」（debt）としてスレッドごとの debt slot に記録する。この debt slot の一覧はプロセスに 1 つで、型も storage も区別しない。書き手は値を差し替えると、外した古いポインタと同じ値の debt を一覧から探し、参照数を 1 増やして「支払う」。

壊れる順序（`HybridProtection::attempt`、`src/strategy/hybrid.rs`）:

1. スロット A の読み手 R が、A から `ptr = P` を読む
2. A の書き手が P を外して全 debt を払う。R はまだ debt を置いていないので、何も払われない。P の最後の参照が落ちて解放される
3. 別のスロット B（別のテストの `Outlet`、または `InputSlot` の `Pair`）の書き手が新しい値を確保し、同じアドレス P に置かれる。B に store する
4. R が debt slot に P を記録する（`new_fast(P)`）
5. B の書き手が P を外し、全 debt から値 P を探して、R の debt を払う（B の値の参照数を 1 増やす）
6. R が A を読み直すと P ではない。debt を自分で返そうとして失敗するので、「誰かが払ってくれた」と判断し、P を自分の値として返す。これは B の値である

結果:

- B の値が 1 フィールドのメッセージなら、2 フィールドのスロットから 1 フィールドのメッセージを読む（`index out of bounds`）
- B の値が `Pair`（`InputSlot` の中身）なら、`Pair` を `Message` として読む（逆も起きる）。型の取り違えで SIGSEGV になる。`Arc<Message>` と `Arc<Pair>` の確保は macOS の malloc で同じ 48 バイトの size class に入るので、アドレスが互いに再利用される
- `fallback`（helping の経路）でも、同じ形で支払われた debt を `T::dec` で返すので、別の型の値を `Message` として drop しうる

同じテストの中では、外向きスロットとチャンネルが同じスキーマ・同じ型なので症状が見えない。ほかの slot テストと並行に走ったときだけ見えるのはこのため。

### 根拠

- `lldb` で落ちた瞬間のスタック（3 件）: 2 件は `readers_see_whole_messages_in_order` の読み手が 1 バッファの「メッセージ」の `Buffer::field` を読み、`pool` が `0x1e97` などの小さな整数で不正アクセス（中身がメッセージでない）。1 件は `input_group_switches_all_slots_at_once` の読み手が `InputSlot::load` の `Arc::clone` でアドレス `0x2` に書き込み（`Pair` のつもりで `Message` を読んだ）
- arc-swap だけの再現（mujou を使わない safe Rust、scratchpad の `abarepro`）: 同じ型の `ArcSwap` を 2 つ作り、1 本の書き手が a と b に交互に store、8 本の読み手が a を `load_full` して b の値が返らないか調べる。debug ビルドで 50 万回の書き込みあたり 5〜10 回、a が b の値を返した（3 回中 3 回再現。code-reviewer の追試では 3 回中 2 回、5 / 0 / 2 回）。release では 200 万回で 0 回（窓が狭い）
- arc-swap の Changelog と GitHub の検索では、この不具合の報告は見つからなかった（未報告。upstream に報告するかはユーザー判断）

## 修正

`crates/mujou-core/src/slot.rs` から `arc-swap` を外し、依存からも削除した。

- `Slot::value` を `Mutex<Arc<Message>>` にした。lock の下では `Arc` の複製と差し替えだけを行う。バッファを埋めるのも古いメッセージを drop する（バッファをプールへ返す）のも lock の外
- `wait_newer` は同じ mutex と `Condvar` で待つ。待ち手は数を増やしてから lock を取り、値を確かめ、`wait_timeout` で lock を手放して眠る。書き手は lock を手放したあとに数を読み、0 でなければ `notify_all` する。書き手が lock を取るのは待ち手が値を確かめたあとなので、増えた数が必ず見え、通知を取りこぼさない。以前の `Waiters`（別の lock と SeqCst fence の組）はやめた。fence の論法は、値が atomic でなく mutex の下にあると成り立たないため
- `InputSlot::pair` を `Mutex<Pair>` にした（`Arc<Pair>` は不要になった）。「pair を先に、版を後に読む」順序と、版の Release / Acquire はそのまま

### 仕様の変更

mutex 方式は 2026-10-17 にユーザーが承認した（algorithm-reviewer の must-fix「設計の決定の変更にはユーザーの確認が要る」への対応）。決定は `memory/project_native_core_rust.md`、残る競合の問題は `docs/open-decisions.md` に書いた。

- スロットの読み書きは lock-free ではなくなった。`load` は他の読み手か書き手が `Arc` を複製・差し替える間だけ待つ。設計文書はもともと「lock 付き共有スロット」「OS の mutex で長く待たせない」なので、その範囲にある。`docs/native-core.md` と `memory/project_native_core_rust.md` の `arc-swap` の記述を直した
- 既存テスト `write_without_waiters_takes_no_lock` を削除した。書き手が待ち手用の lock を取らないことを確かめるテストで、待ち手用の lock がなくなった。待っている読み手が書き手を止めないことは、`wait_newer_returns_on_write_or_timeout`（待ち手が 10 秒待つ間に書き、書き込みで起きることを確かめる）が確かめる

## 回帰テスト

`slot::tests::readers_never_see_a_message_of_another_slot`。確率的な smoke test で、通っても安全の証明にはならない。修正は構造的（arc-swap の削除）なので、このテストの役割は lock-free な方式に戻したときの網である。

1 本のスレッドから、2 フィールドの `Outlet`、1 フィールドの `Outlet`、2 フィールドの `InputSlot`（`InputGroup::write`）に順に 10 万回書く。読み手は `Outlet` の外向きスロットと `InputSlot` を 4 本ずつ読み続け、フィールド名が `["a", "b"]` 以外のメッセージが来ないことを確かめる。1 本のスレッドで書くので、解放したメッセージや pair のアドレスが、次に書く別スロットの値に再利用されやすい。`InputSlot` を混ぜたのは、SIGSEGV の原因になった `Pair` と `Message` の取り違えも捉えるため（2026-10-17 追加）。

決定的には捉えられない（arc-swap の窓は読み手が 2 命令の間で横取りされたときだけ開く）。修正前のコードで、このテストが落ちた割合:

| 版                            | 条件                                   | 落ちた回数                         | 測った人      |
| ----------------------------- | -------------------------------------- | ---------------------------------- | ------------- |
| 初版（`Outlet` のみ）         | 単独、読み手 4・書き込み 10 万         | 10 / 100                           | 実装者        |
| 初版（`Outlet` のみ）         | 単独、読み手 8・書き込み 10 万         | 15 / 40                            | 実装者        |
| 初版（`Outlet` のみ）         | 単独、読み手 8・書き込み 30 万         | 21 / 40                            | 実装者        |
| 初版（`Outlet` のみ）         | `slot::` 全体と並行、読み手 8・10 万   | 17 / 100                           | 実装者        |
| 初版（`Outlet` のみ）         | 単独、読み手 8・10 万                  | 8 / 100                            | code-reviewer |
| 初版（`Outlet` のみ）         | `slot::` 全体と並行、読み手 8・10 万   | 13 / 40                            | code-reviewer |
| 現行（`InputSlot` を追加）    | 単独、読み手 4 + 4・10 万              | 47 / 100（うちシグナル 33）        | 実装者        |
| 現行（`InputSlot` を追加）    | `slot::` 全体と並行                    | 21 / 40（うちシグナル 12）         | 実装者        |

負荷によって割合は大きく変わる。現行版は debug ビルドで 1 回約 1.2 秒。

## 再現の割合（`target/debug` の lib テストを `slot::` で直接回す）

| コード   | 回数 | 落ちた回数                                                        |
| -------- | ---- | ----------------------------------------------------------------- |
| 修正前   | 300  | 2（どちらも SIGSEGV）                                             |
| 修正前   | 400  | 5（SIGSEGV 4、abort 1）                                           |
| 修正前＋回帰テスト | 100 | 19（回帰テスト 17、既存の順序テスト 1、シグナル 2。重複あり） |
| 修正後＋回帰テスト | 3000（4 本のループを並行に 750 回ずつ） | 0                                  |
| 修正後＋現行の回帰テスト（`InputSlot` 入り） | 200 | 0 |

## 確かめたこと

- Miri（nightly、`-Zmiri-many-seeds=0..4`）: slot の 13 テストと、書き込みを 100 回・読み手 3 本に減らした 3 本の並行テスト（一時的に定数を変えて実行、コミットには含めない）で UB なし
- ThreadSanitizer / AddressSanitizer（nightly、`-Zbuild-std`）: 修正後の slot の 16 テストで報告なし
- Python 側: `tests/mujou/test_core_slot.py` などは Rust の公開 IF（`load` / `wait_newer` / `restore` / `send_with` / `write`）が変わらないので影響なし。`just run` で確認
