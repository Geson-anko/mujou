# examples: ユーザー向けドキュメントのサンプルプログラム

2026-10-17。ブランチ `docs/2026-10-17/docs-content`。`examples/` に 10 本のプログラムを足した。`tests/test_examples.py` は変更していない（ディレクトリ形式の `leaky_chain/main.py` も既存の glob で拾える）。

## サンプルと載せるページの対応

| ファイル                        | 示すこと                                                                                                    | 載せるページ                                                 |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `quickstart.py`                 | 2 ノードのネットワーク、`with mujou.run(...)`、`write` → `mujou.time.sleep` → `read`                         | getting-started/quickstart、user-guide/running               |
| `leaky_chain/main.py`, `leaky.py` | 連続時間のノード（リーク積分器）、`dt` に比例する更新と打ち切り、入力はグローバル時間上の信号、厳密解と平衡 | user-guide/writing-a-node、concepts/time、user-guide/time    |
| `branching.py`                  | `NextPort` の分岐、`PrevPort(multi=True)` の合流（`recv()` が dict）、`send_to`、`build()` のグラフ          | user-guide/building-a-network、concepts/ports                |
| `compose.py`                    | `Compose(enc=..., dec=...)` と `mujou.run(a, b)`、パス `"enc.scale"` / `"0.scale"`                          | user-guide/compose、concepts/networks                        |
| `save_and_restore.py`           | 状態の自動収集（Tensor と nn.Module）、終了後の `state_dict`、pickle、`state=` で復元して続ける             | user-guide/state、concepts/state                             |
| `subscribe.py`                  | `subscribe` の反復、別スレッドからの `shutdown` で反復が終わる、`Node.timer`                                 | user-guide/running                                           |
| `torch_node.py`                 | ノード内の `nn.Module`、`seed` と `self.generator`、`# mujou: local-autograd` の BP、勾配流、`set_num_threads` | user-guide/torch（将来 user-guide/update-rules にも）        |
| `testing_a_node.py`             | `VirtualTimer` と `ManualRun.step` で離散アルゴリズムとビット単位で一致                                      | user-guide/testing                                           |
| `timers.py`                     | `Timestamp`、実行ごとの `Timer`、タイムスケール（実行のタイマーとグローバル）                               | user-guide/time、concepts/time                               |
| `errors.py`                     | `NetworkError` と `__cause__`、`on_exception` → `on_stop`、`on_start` の例外は `mujou.run` が送出、ログ      | user-guide/errors                                            |

案からの変更: 案の 10 本をそのまま採った。`leaky_chain` はディレクトリ形式の例を兼ねる（ノードの定義を `leaky.py` に分けた）。

## 判断

- 実時間で待つサンプルは 1 本あたり実行時間 1 秒前後に収めた。ノードは周期が無いと待たずに回り続けるので、走らせる時間を短くした
- 実時間に依存する値の assert は緩くした（リーク積分器は厳密解との差 0.05、学習は損失が 1/100 以下）。決定的にできるもの（quickstart・branching・compose の出力、testing_a_node のビット一致）は等号で確かめる
- `subscribe` の最初の値が zeros であることは assert しない。呼んだ時点ですでに送られていることがある（docstring どおり「呼んだ時点の最新」）
- `errors.py` は `logging.basicConfig(stream=sys.stdout)` でログを stdout に出す。mujou はノードの例外を traceback 付きで logger `"mujou"` に出し、handler が無いと Python は stderr に出す。テストは stderr の Traceback で落とすので、ログを stdout に向けた
- `torch_node.py` は `p.grad`（`Tensor | None`）を避けて `torch.autograd.grad` を使った。pyright（basic でもプロジェクトの設定で examples が対象）が `p.grad` の None を指摘するため
- 表示は `.tolist()` を使わず Tensor の repr にした。torch の stub で `tolist` の戻り値が不明になり、pyright strict が落ちる
- pyright: プロジェクトの設定（strict は `src/` のみ、`examples/` は basic で対象）で 0 errors。加えて scratchpad の設定で `examples/` を strict で流して 0 errors

## API の気づき（直していない）

1. ノードの例外は logger `"mujou"` に traceback 付きで出る。handler を設定しないと stderr に出る。ドキュメントの errors ページに書くべき。サンプルのテストが stderr の Traceback で落ちる仕組みとも関係する
2. `mujou.run` は `Network` しか受け取らない。1 ノードを走らせるだけでも、ポートを公開する `Network` を書く必要がある（どのサンプルにも同じ定型がある）
3. `send_to` で枝ごとに違う値を送るとき、ノードは `self.next.channels` の並び（ネットワークが connect した順）に頼るしかない。チャネルに名前や相手のパスが無いので、つなぐ順が変わると黙って入れ替わる
4. グローバルタイマーの上で `state=` から復元すると、グローバル時刻が保存時の時刻へ戻る（`mujou.time.time()` が巻き戻る）。docstring には「タイマーは保存時の時刻から続く」とあるが、ホストの `mujou.time` も巻き戻ることは明記されていない
5. `on_exception()` は引数を取らないので、ノードはどの例外で止まったかを知れない
6. 周期が無いノードは待たずに回り続け、ノード 1 つで CPU コア 1 つを使い切る。周期（`Node(period=...)`）が入ったらサンプルにも足す
7. `Run.read` の戻りの時刻と `mujou.time.time()` を比べる例（leaky_chain）は、`write` が時刻を返さないため、`write` の直後に `mujou.time.time()` を読んで代用した。`write` が書いた時刻を返すと、入力の切り替え時刻を正確に取れる

## 確かめたこと

- 全プログラムを free-threaded（3.14t）と `-X gil=1` でそれぞれ 3 回ずつ実行し、全部終了コード 0、stderr に Traceback なし
- `uv run pytest tests/test_examples.py`（10 passed）、`just test`（cargo test と pytest、462 passed）、`just type`（0 errors）、`just docs-build`
- `just format`（`pre-commit run -a`）は同じ worktree で別の担当が `docs/ja/` を編集中のため実行せず、`examples/` と本ファイルに対して pre-commit を流した
