# generator を super().__init__ で作る レビュー

対象: `git diff 0244ccc..HEAD`（f2cb7ee、ブランチ feature/2026-10-18/generator-in-init）

## verdict: request-changes

実装とテストは要求どおり。spec の表と memory の本文に、生成器を `_on_thread_start` で作るという古い記述が残っている。直すのは文書だけで済む。

## must-fix

1. `specs/torch.md:82`（「実行スレッドの設定」の表）
   - 問題: 表の前の文は「`mujou.Node` は `_on_thread_start()` で次を設定する」「どれもスレッドごとの設定」。その表に「乱数生成器 | ノードごとの `torch.Generator` を、渡された seed で作る」の行が残っている。同じ節の 86 行目（`super().__init__()` の中で作る、スレッドに結び付かない）と矛盾する
   - 根拠: `src/mujou/_core/node.py:173` で `__init__` が作り、`:359-361` は `_generator is None` のとき（基底の `__init__` を呼ばないサブクラス）だけ作る
   - 確信度: 高 / 深刻度: 中（spec の自己矛盾。コードの動作には影響しない）

## should-fix

1. `memory/project_torch_module.md` の本文の箇条書き（「`mujou.Node` は ~~`on_start`~~ 内部の hook `_on_thread_start` … で、スレッドごとの設定を行う: … ノードごとの `torch.Generator`」）
   - 問題: description は直したが、本文はまだ生成器を `_on_thread_start` で作ると書いている
   - 根拠: 上と同じ（node.py:173）
   - 確信度: 高 / 深刻度: 低

2. `docs/{en,ja}/user-guide/writing-a-node.md` の「2. Set up the state in `on_start`」の節（en 105 行・127 行、ja 同じ行）
   - 問題: 節の冒頭は「Create the state tensors in `on_start`」と言い切るのに、127 行目で「`nn.Module` は `__init__` でも `on_start` でも作れる」に変わった。モジュールも状態なので、テンソルだけ `on_start` に置く理由を読者が探してしまう。これまで on_start を勧める理由だった生成器（187d81b で足した説明）は、今回の変更で理由にならなくなった
   - 根拠: `Node` の docstring（node.py:71）は「Allocate the tensors of the state by `on_start`」で、`__init__` での確保も許している。テンソルを `on_start` に置くべき理由が残っているなら書き、無いなら節の言い切りを緩める。どちらにするかは orchestrator / ユーザーの判断
   - 確信度: 中 / 深刻度: 低

3. `tests/mujou/_core/test_node.py`（`TestThreadSetup`）
   - 問題: 古い `test_the_generator_is_made_on_the_thread` を消したので、docstring の `Raises: RuntimeError`（`super().__init__()` より前に読む）を確かめるテストが無くなった。`super().__init__()` より前に `self.generator` を読むと `RuntimeError` になることを、公開の property を通して確かめる 1 件があるとよい
   - 根拠: node.py:203-209 の公開の契約（Raises）。手で確かめたところ、メッセージは "the generator is created by super().__init__(); call it first" で、期待どおりに動く
   - 確信度: 中 / 深刻度: 低

4. `docs/ja/user-guide/state.md:73`
   - 問題: 「復元した実行は新しいインスタンスで走り、生成器も `seed` から作り直されるので、生成器の状態を足さないと、復元後の乱数列が止めずに続けたときと違う列になる。」は読点が 4 つある 1 文で、「走り」は英語の run の直訳。memory/feedback_japanese_style.md（読点で細かく区切らない、英語の直訳を避ける）に合わない。`docs/ja/user-guide/torch.md:45` の「新しいインスタンスで走り」も同じ
   - 根拠: memory/feedback_japanese_style.md
   - 確信度: 中 / 深刻度: 低

## nit

1. `docs/ja/user-guide/writing-a-node.md:127`「（`__init__` の中から使える）」と `:220`「（`__init__` の中でもよい）」は、文体の基準が削るよう求める補足のカッコ。本文に入れるか削る
2. `src/mujou/_core/node.py:203-204` の `Raises: RuntimeError: super().__init__() has not been called yet.` は厳密には「基底の `__init__` が呼ばれず、スレッドもまだ始まっていない」とき。基底の `__init__` を呼ばないサブクラスでも、`on_thread_start` 以降は raise しない。実害は無い（確信度: 高 / 深刻度: 極小）
3. `copy.copy(node)` で作った浅いコピーが元と同じ `torch.Generator` を共有するようになった（手元で `t.generator is n.generator` → True）。これまでは実行前の `_generator` が None だったので、それぞれが起動時に別の生成器を作っていた。ノードの浅いコピーを実行する使い方は想定外で、Port も共有されるので、記録だけ（確信度: 高 / 深刻度: 極小）。`deepcopy` は別の生成器になり、問題ない

## 確認した点（指摘なし。依頼の観点への回答）

- `self.generator` は `super().__init__()` の直後から使える。`_on_thread_start` は `_generator is None` のときだけ作るので、`__init__` で引いた続きを `on_thread_start` / `on_start` が引く。テスト `test_on_start_continues_the_sequence_from_init` が確かめている
- `super().__init__()` を呼ばないサブクラス: クラス属性の既定 `_generator = None` によって `_on_thread_start` で作られる（CPU、seed なし）。既存の `test_works_without_calling_the_base_init` が通る
- seed が None: `_make_generator` が `generator.seed()` を呼ぶ。以前と同じ動作で、作る時刻が構築時になっただけ
- CUDA: `_check_device` が index を固定した後に `torch.Generator(self._device)` を作る。`cuda:N` の生成器はホストのスレッドで作っても、そのデバイスに結び付く。`__init__` で CUDA に積んだ仕事は、`session.py` の起動の前の `_synchronize_current_streams()` で待つ（既存）。問題は見当たらない（GPU が無いので実行はしていない）
- インスタンスが 1 回しか走らないこと: `graph.start_once` が `"... has already run; create a new instance to run again"` で強制している。どのノードも起動する前に断られた起動は数えないが、hook が 1 つも走らないので生成器は進まない。「復元した実行は新しいインスタンスで走り、生成器は seed から始まる」という説明は正しい
- テストは公開の `generator` property と `start_manual` を通して確かめており、seed も固定している。内部のモックや private の直接テストは無い
- 成果物の汚染（`</content>` など）は無い

## 検証結果

- just run: pass（559 passed, 18 skipped、pyright 0 errors、mkdocs build --strict 成功、format 後に作業ツリーの差分なし）
