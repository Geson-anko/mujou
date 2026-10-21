---
name: project-viz
description: ネットワークのグラフは mujou.viz.to_html(root) -> str で自己完結の HTML に描く。CDN なし、JS はテストせず目視で確かめる（2026-10-21）
metadata:
  type: project
---

2026-10-21、ユーザーの依頼「ネットワークのグラフを HTML で出力し、トップ層からクリックで展開していく UI」で追加した。

- 公開名は `mujou.viz.to_html(root)` だけ。保存は利用者が `Path.write_text` で行う。`import mujou` では読み込まない
- 木は `tree.walk` で読む（run と同じ検査・パス）。テンプレートは `src/mujou/viz.html`、データは `#mujou-graph` の JSON。テストはその JSON を見る
- 外部の資源を読まない（オフラインのロボットでも開ける）。elkjs などを CDN から読む案は却下

**Why:** 出力は利用者が目で見るもので、JS のレイアウトは単体テストで担保できない。

**How to apply:** viz.html を変えたら、ヘッドレス Chromium（playwright。足りない共有ライブラリは apt-get download で scratchpad に展開し LD_LIBRARY_PATH で渡す）でスクリーンショットを撮って目視する。docs/assets/viz/*.png は examples/visualize.py のページを撮ったもので、UI を変えたら撮り直す。関連: [[project-docs-site]]
