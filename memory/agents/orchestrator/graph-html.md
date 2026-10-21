# ネットワークのグラフを HTML で可視化する（2026-10-21）

ユーザーの依頼: ネットワークのグラフを HTML で出力し、トップ層からクリックで展開していく UI。solo-dev-cycle + レビューエージェント、worktree、PR まで。出力は必ず目視で確認する。

## 段階 1 計画

- 公開 IF: 新しい公開モジュール `mujou.viz` に `to_html(root: Node | Network) -> str` だけを置く。`import mujou` では読み込まない（`mujou.testing` などと同じ）。ファイルへの書き出しは利用者が `Path.write_text` で行う（関数を増やさない）
- 木の読み取りは `mujou.tree.walk` を使う（run と同じ検査・同じ例外。パスも run と同じ）。ネットワークの入れ子は `vars()` を代入の順に読む
- HTML は自己完結（外部の CDN を使わない。オフラインのロボットでも開ける）。テンプレートは `src/mujou/viz.html`、グラフのデータは `<script type="application/json" id="mujou-graph">` に埋める（`<` は `<` に逃がす）
- データ: ネットワーク（name / path / class / children / 公開した Port の参照）、ノード（class / period / device / ports）、Port（name / path / Port か MultiPort / send・recv の spec の repr / ホストが書くか）、エッジ（channel ごとに 1 本、from / to の Port のパス）
- エッジの向き: 片方だけが送るなら送る側から。両方が送るなら木の順で先の側から（`Channel` の両端は private なので接続の呼び出し側は使わない）
- UI: ルートは展開済み、その子のネットワークは畳んだ箱。箱をクリックで展開/折り畳み。畳んだネットワークに入るエッジは箱へ持ち上げ、同じ組の複数のエッジは本数を付けて 1 本に。ノードのクリックで詳細パネル。Expand all / Collapse all / Fit、ドラッグでパン、ホイールでズーム、ダーク対応。レイアウトは各ネットワークの中で層別（最長路、閉路は DFS で切る、重心で並べ替え）
- テスト観点（`tests/mujou/test_viz.py`）: 埋め込んだ JSON の木（順序・パス・クラス）、Port（spec・種類・ホストが書くか）、エッジ（channel ごとに 1 本・向き）、公開した Port の参照、ルートがノード、不正な木で run と同じ例外、`</script>` を含む名前の逃がし、外部の資源を読まない、`import mujou` で読み込まれない
- 目視確認: playwright の chromium（ライブラリは scratchpad に展開）でスクリーンショットを撮り、畳んだ状態・展開した状態・詳細パネル・ダークを見る
- リスク: レイアウトの見た目。JS はテストしないので目視で担保する

## 却下した案

- elkjs / cytoscape を CDN から読む: レイアウトの質は高いが、オフラインで開けない
- `Tree` にネットワークの構造を足す: run の経路を変えることになる。viz の中で読めば足りる

## 段階 4 レビュー（code-reviewer、request-changes。記録は memory/agents/code-reviewer/graph-html.md）の裁定

- must-fix（木の外のノードの Port を参照するネットワークで素の KeyError）: 採用。run も読まない参照なので飛ばす（walk は変えない）。テストを追加
- should-fix 1（子のないネットワークで高さが -Infinity）、2（`p.class` の esc 漏れ）、4（向きの違う Channel が重なる。順序のない組で集約し、矢じりは受け取る端ごと）、5（名前をパスから逆算。属性名を渡す）: 採用
- should-fix 3（host 判定・is_port・join の写し）: 採用。`wiring.written_by_the_host`、`tree.is_port`、`tree.join` を `_` なしにして共有
- should-fix 6（埋め込みの JSON を丸ごと固定するテスト）: 却下。JSON は `to_html` の出力の一部で、ブラウザなしに決定的に検査できる唯一の口。viz.py の docstring に「ページが読むのはこの JSON だけ」と書いた
- nit: period を repr に、ノードの Port を Member.ports から、isHost の余分を削除、toggle のレイアウト 2 回、畳んだときにパネルを閉じる、Fit の引数、pointercancel を採用。部分木の外の Port を公開した行にエッジが繋がらない件とテストの分岐は見送り（提供シナリオの外、読みやすさの問題ではない）
- 修正後に目視で再確認: 空の Group を開く、畳んだ箱の間の逆向きの 2 本（×2、両矢じり）、選択中のノードを Collapse all で畳む

## 段階 5

- docs/{en,ja}/topics/visualize.md（スクリーンショット docs/assets/viz/*.png は examples/visualize.py のページをヘッドレス Chromium で撮った）、docs/en/api/viz.md、examples/visualize.py、docs/{en,ja}/examples.md、mkdocs.yml の nav、specs/architecture.md の層の表
