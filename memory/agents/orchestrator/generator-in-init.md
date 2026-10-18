# generator を super().__init__ で作る（solo-dev-cycle）

2026-10-18 のユーザーの指示:「generator は super().__init__ で生成し、__init__ の中からすでに扱えるべき」。

## 段階 1: 計画

- 公開 IF: `Node.__init__(*, device="cpu", seed=None, period=0.01)` は変えない。`super().__init__` の中で `torch.Generator(device)` を作り、seed を入れる（None なら `seed()`）。`self.generator` は `super().__init__()` の後から使える
- `_on_thread_start` では作り直さない。`__init__` で引いた乱数の続きを `on_thread_start` / `on_start` 以降で引ける
- `super().__init__()` を呼ばないノードも動く（既存のテスト `test_works_without_calling_the_base_init`）。その場合だけ `_on_thread_start` で作る
- テスト観点: `__init__` で引いた乱数が seed どおり／`on_start` がその続きを引く／基底の `__init__` を呼ばないノード／`on_thread_start` で引く既存のテストは不変
- ドキュメント: node.py の docstring と generator の property、specs/torch.md、docs/{en,ja} の torch.md・writing-a-node.md・state.md。`nn.Module` を `on_start` で作る理由が無くなるので「`__init__` でも `on_start` でも作れる。乱数で初期化するなら `self.generator`」に変える
- リスク: CUDA の generator をホストのスレッドで作る（`torch.Generator("cuda:N")` はどのスレッドでも作れる）。M:N の実装（別ブランチ）は generator がスレッドに依存しなくなることを前提にできる
- PR #71 が writing-a-node の同じ段落を直すので、そのブランチを取り込んでから始めた

## 段階 2: テスト
- `test_the_generator_is_made_on_the_thread` を `test_the_generator_is_ready_in_init` に置き換え、`test_on_start_continues_the_sequence_from_init` を足した。実装前に 2 件が期待どおり失敗（red）

## 段階 3: 実装
- `Node.__init__` の最後で `_make_generator()`。`_on_thread_start` は `_generator` が無いときだけ作る（基底の `__init__` を呼ばないサブクラス）。`just run` green（559 passed）

## 段階 4: 自己レビュー
- diff を読み直した。計画外の行なし、整理点なし

## 段階 5: ドキュメント
- node.py の docstring、specs/torch.md、docs/{en,ja} の writing-a-node・torch・state、memory 3 件（project_torch_module、MEMORY.md、project_runtime_api_revision）
