# ネットワークのグラフを HTML で可視化する（mujou.viz.to_html）レビュー

対象: `git diff main...HEAD`（1fbbb39）、worktree `feature-graph-html`

## verdict: request-changes

## must-fix

1. `src/mujou/viz.py:72`（`ports.append({"name": attribute, "path": tree.path(value)})`）
   - 問題: ネットワークが木の外のノードの Port を属性に持つと（`l = Hidden(); self.prev = l.prev` で `self.l = l` を忘れた場合など）、`tree.path` が素の `KeyError` を投げる。`walk` も `mujou.run` もこの木を受け付ける
   - 根拠: 計画は「run と同じ検査・例外」。docstring も「`TypeError, ValueError`: run が投げるとおり」。再現: `class Exp(Network): __init__ で self.l = Hidden(); self.prev = outside.prev` として `to_html(Exp(Hidden()))` が KeyError、`mujou.run(Exp(...))` は通る
   - 確信度: 高（挙動は再現済み）。直し方（その行を飛ばす／run 側でも拒む）は orchestrator が決める
   - 深刻度: 中

## should-fix

1. `src/mujou/viz.html:282`（`const innerH = Math.max(...colH)`）。子のない開いたネットワークで `Math.max()` が `-Infinity` になり、`c.h = -Infinity` になる。`to_html(Group())`（ルートが空）は `fit` で `k = -0` になって何も描かれない。空の子ネットワーク（`Group()` や Port の参照しか持たないネットワーク）は開くと親の `dy` が Infinity になり、レイアウトが壊れる。`walk` も `run` もどちらの木も受け付ける（確認済み）。確信度: 高（コードを読んで確認。ブラウザでは見ていない）。深刻度: 低〜中
2. `src/mujou/viz.html:476`（`<span class="kind">${p.class}</span>`）。innerHTML の中でここだけ `esc` を通していない。`type(port).__name__` はどんな文字列にもできる（`type("<img src=x onerror=alert(1)>", (Port,), {})` がそのまま JSON に入ることを確認）。自分のコードから起きる self-XSS だが、エスケープの方針として抜けている。確信度: 高。深刻度: 低
3. `src/mujou/viz.py:88-90`。host の判定が `wiring._written_by_the_host` の写しになっている。いまは一致しているが、同じ事実を 2 か所に置いており、片方だけ変わるとずれる。refactor-conventions の「複数のモジュールが使う内部の名前には `_` を付けない」に従い、`wiring.py` か `port.py` に `_` なしで置いて共有するのが筋。`_is_port` / `_join` も `tree.py` の写しで、同じことが言える。確信度: 中。深刻度: 低
4. `src/mujou/viz.html:381`（`key = a|ra|b|rb`）。集約のキーに向きが入っている。畳んだ箱 A と B の間に A→B と B→A の channel があると、同じ曲線が 2 本重なって描かれ、`×n` のラベルも同じ位置に重なる。計画の「同じ組は本数付きで 1 本に」を順序のない組と読むなら仕様と違う。確信度: 中（「組」の解釈次第）。深刻度: 低
5. `src/mujou/viz.py:49`（`name = path.rpartition(".")[2] or tree.name(path)`）。名前をパスから逆算しているので、`Group(**{"a.b": node})` の名前が `"b"` になる（確認済み。path は `"a.b"`）。呼び出し側は attribute をすでに持っているので、引数で渡せば分岐も逆算も要らない（feedback_good_taste）。確信度: 高。深刻度: 低
6. `tests/mujou/test_viz.py` 全体。テストが埋め込みの JSON の形をまるごと固定している。この JSON は viz.py と viz.html の間の内部の契約で、公開 IF は「HTML を返す」こと。testing-strategy の「内部実装の詳細は見ない」と緊張関係がある。JS をテストしない以上、観測できるのはこの JSON だけなのでやむを得ない面もある。それなら `#mujou-graph` の JSON を公開の契約として docstring か specs に書くか、テストを木・エッジ・エスケープの要点に絞るか、どちらかに決めるのがよい。確信度: 中。深刻度: 低

## nit

- `viz.py:57` の `f"{period:g}"` は有効数字 6 桁に丸める（`0.0123456789` が `"0.0123457"` になる）。`inf` を JSON に入れないために文字列にするなら `repr` のほうが値を失わない
- `viz.py:54-62` は、ノードの Port を `vars()` から読み直している。`tree.members` の `Member.ports` に同じものが同じ順で入っているので、パスで引けば読み直さなくて済む
- `viz.html:207` の `isHost` にある `!peers.has(p.path)` は余分。`p.host` は channel がないときだけ true になる
- `viz.html:422-426` の `toggle` は `layout` と `place` を呼んだあと、`render()` の中でもう一度レイアウトしている（差分を取るためと分かるが、2 回走る）
- `viz.html:413-421` と `viz.html:523`。選んだノードを含むネットワークを畳んでも（Collapse all でも）詳細パネルは開いたままで、`selected` も消えない。そのノードの箱はもう見えていない
- `viz.html:527` は `fit` に MouseEvent を `animate` として渡しており、結果としてアニメーションする。意図どおりなら `() => fit(true)` と書いたほうが明確
- `viz.html:507` は `pointercancel` を扱っていない（タッチで drag が残り得る）
- ネットワークが自分の部分木の外の Port を公開すると、畳んだ箱にその行は出るが、エッジは決してその行につながらない（見た目だけの問題）
- `tests/mujou/test_viz.py` の `hidden()` は分岐 `path == "body.l1"` を持っている。期待値に条件が入っている（軽微）
- パッケージング: `viz.html` は `py.typed` / `_native.pyi` と同じ置き方なので、maturin の wheel に入るはず。ただし wheel を作って確かめてはいない（確信度: 低）

## ドキュメント（段階 5 への提案）

- `docs/{en,ja}/`: 「Network を見る」ページ（`to_html` と `Path.write_text`、畳む・開く UI、host の印、エッジの向きの規則）。API リファレンスに `mujou.viz` を加える
- `examples/`: 小さなネットワークを `net.html` に書き出す example を置き、ページへ snippet で取り込む
- `specs/`: `specs/network.md` に節を足すか、`specs/viz.md` を新しく作る。書く内容は、エッジの向き（片方だけが送るなら送る側から、両方なら木の順で先の側から）、JSON の形（公開の契約にするかどうか）、自己完結の方針（CDN を却下した理由）

## 検証結果

- just run: fail（docs-build だけ）。format・test-rs・test-py（767 passed / 1 skipped）・type（pyright 0 errors）は pass
  - docs-build の失敗は、レビュー対象のコミットにない未コミットの段階 5 の作業による。`docs/{en,ja}/topics/visualize.md` が `assets/viz/{closed,open,node}.png` を参照しているが、その画像がない（strict で警告 3 件）
- 手で試したこと（scratchpad の exp.py / exp2.py）: 木の外の Port を参照すると KeyError、空の Group を run は受け付ける、名前にドットがあると名前がずれる、Port のクラス名がエスケープされずに JSON に入る、period を丸める
