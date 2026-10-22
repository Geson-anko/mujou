# code-simplifier: pc-optimizers

2026-10-22。orchestrator の「レビューの裁定」で採用された指摘と、docs の teacher からの追加の指摘（換算は「2 回の学習の間の時間 Δt / lr」で書く）を反映した。却下された指摘（`load_state_dict` の shape 検査、`elapsed == 0` の早期 return の順、weight decay の除外）は扱っていない。`docs/` の本文は編集していない。format の hook が `docs/` の md を整形したが、対象は並行して進む docs 作業のファイルである。

## 変更

- `src/mujou/pc/optim.py`
  - `AdamW.eps` を有限で > 0 にした（`_check_positive` の対象に加えた）。使う場所が 1 か所だけになった `_check_non_negative` は、weight_decay の検査に inline して削除した
  - `BoundOptimizer.step_` の `elapsed` の説明を「その歩みが重みとする時間（`Tick.elapsed`。打ち切った dt の和で、捨てた時間を含まない）」に直した
  - `AdamW` の docstring
    - 「陽的 Euler」を、torch と同じ順の説明に替えた（decay は歩みの前の θ に掛け、Adam の項は更新後の m・v・c で取る）
    - eps の Args と Raises を更新した
    - Adam の歩みは勾配の大きさに依らず約 η なので、`tau = Δt / lr`（Δt は 2 回の学習の間の時間）と書いた。例は `tau=32e-3 / 1e-3`（32 ms 見せる入力から 1 回学ぶ場合）にした
  - `SGD` の docstring: `learn_interval / tau` の説明を「2 回の学習の間の Δt で、学習率 Δt / tau」に直した
- `src/mujou/pc/layer.py`: docstring の折り返しを 2 か所直した。Args の `optimizer` に、自作の `Optimizer` も渡せることを書き足した
- `tests/mujou/pc/test_optim.py`
  - TestChecks を、eps=0 を断る形にした。tau・tau_m・tau_v・eps を同じ parametrize で検査し、weight_decay は別に検査する。0 を受け付けるのは weight_decay だけ
  - 長い 1 歩のテストで、c_m と c_v の両方が 1.0 になることを確かめる。θ が有限かどうかを見る弱い検査は削除した
  - 状態の読み込みのテストで、m と v の両方を確かめる
- `tests/mujou/pc/test_chain.py`: moment の shape を、そのノードの param の shape と比べる
- `tests/mujou/pc/test_layer.py`: `test_no_theta_moves_while_every_mask_is_false` を SGD / AdamW で parametrize した
  - 先にラベルのある 2 入力で学ばせる（AdamW では c_m > 0 を確かめる）
  - 全ノードが全 False の入力を見てからは、θ がビット単位で変わらないことを確かめる
  - ゲート `mask.any()` を外す変異を入れると AdamW の 2 本が落ちることを確認した。pyright の possibly-unbound を避けるため、ループは 2 段に分けた
- examples
  - `examples/pc_regression.py`: 式の途中で折れていたコメントを直した
  - `examples/pc_merge.py`: 内包表記の `g` を `g_p` にした
  - `examples/feedback_alignment.py`: 定数 `SGD` を `OPTIMIZER` にした
- `specs/pc.md`
  - `elapsed` の意味を書いた（捨てた時間はモーメントも c も動かさず、decay はラベルのある時間にだけ掛かる）
  - θ の離散化の説明を実際の形に直した
  - K31 の eps > 0 とその理由を書いた
  - Adam の正規化の性質を書いた（未測定）
  - 段階 2 の不動点が保てる条件を足した
- `specs/scenarios.md` の K31 と `memory/project_pc_optimizers.md` を、上の変更に合わせた

## just run

pass。

- format: pass
- cargo test: pass
- pytest: 905 passed, 1 skipped
- pyright: 0 errors
- docs-build（strict）: OK
