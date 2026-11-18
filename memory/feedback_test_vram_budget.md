---
name: feedback-test-vram-budget
description: テストの VRAM は 2GB 以内（上限 4GB）。conftest がアロケータを device ごとに 1 GiB に絞り、CUDA テストの後に empty_cache する
metadata:
  type: feedback
---

テストスイートが使う VRAM は 2GB 以内に収める。ユーザーは 4GB の上限をかけて走らせる（2026-11-18）。

**Why:** テストが重く、GPU のテストが VRAM を食いすぎていた。開発は macOS の Docker の中で行い、資源が限られる（RAM 7GB の VM ではフルスイートの途中でセッションごと落ちた）。

**How to apply:**

- `tests/mujou/conftest.py` が `cuda` マーカーのテストの前に `torch.cuda.set_per_process_memory_fraction` でキャッシュアロケータを device ごとに `CUDA_BUDGET`（1 GiB）に絞る。CUDA のコンテキストと合わせて 2GB に収まる。超えるテストは OOM で落ちる
- autouse の `release_cuda` が CUDA テストの後に `gc.collect()` → `synchronize()` → `empty_cache()` し、実行や CUDA graph のメモリプールを次のテストに持ち越さない
- CUDA テストは小さなテンソルで、run の数を減らし、同じ device のセットアップを共有するものは 1 本にまとめる
- 収束を待つ数値テストは、許容誤差を緩めずにステップ数・深さを減らす（skill testing-strategy）
