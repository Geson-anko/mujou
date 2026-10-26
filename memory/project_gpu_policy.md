---
name: project-gpu-policy
description: CUDA のノードが自分の device の上で send したテンソルは、CUDA のノードが読むポートなら（2026-10-25）、コアが不透明な値として device に置いたまま持ち、読み手は device の上でコピーする（2026-10-24 にユーザーが「コアは GPU をサポートしない」を覆した）。書き手が send の中で自分の stream を待ってから公開（S2）、読み手は record_stream、別の device を読むノードには副 stream。ホストの write は、ポートを持つノードの CUDA device のテンソルだけ device に置く（2026-10-25）。それ以外の値はホストのメモリ。ノードのデバイスは CPU と CUDA だけ
metadata:
  type: project
---

2026-10-15 決定、2026-10-24 にユーザーが覆して書き換えた。

- **CUDA のノードの間のメッセージは device に置いたまま渡す（2026-10-24、ユーザーの依頼）。** 依頼の原文は「ノードが GPU デバイス上で計算した時、そのテンソルは device -> host -> device という無駄な経路を踏む。デバイス転送を無くしたい。ノードはそれぞれ別々の cuda stream になるので同期が必要」。それまでの「コアとメッセージは GPU をサポートせず、境界で device → host → device とコピーする」（2026-10-15）を覆した
- **device に置くのは次の密な（strided）テンソル。** (1) ノードの hook の `send` / `send_to` で、ノード自身の CUDA device の上のもの。ただしそのポートの接続先に CUDA のノードが 1 つ以上あるときだけ（2026-10-25、ユーザーの決定。CPU のノードだけ・ホストだけが読むポートは send でホストへ移す。配線時に `Wire.cuda` で静的に決め、副 stream もその分だけ作る）。(2) ホストの `write`（と `inputs=`）で渡した、そのポートを持つノードの CUDA device の上のテンソル（2026-10-25、ユーザーの決定。最初は「渡したテンソルの device に置く」と決めたが、同じ日の監査を受けて持ち主の device に狭めた。持ち主が CPU のノード、持ち主と別の device のテンソルはホストのメモリを通す。配線時に `Wiring.input_devices` で決める。`device=` 引数は足さない）。`initial`・復元はホストのメモリのまま。ほかの device のテンソル、sparse、スカラーも今どおり
- **ホストの write の同期も S2。** 呼んだスレッドのその device の current stream で clone し、device ごとに 1 回 event を待ってから公開する（`codec.wait_for_clones`）
- **持ち主の device に狭めた理由（2026-10-25）。** 持ち主以外の読み手（CPU のノード、別の device のノード）は、自分で設定していない stream（ワーカーに残る stream、何も無ければ legacy default stream）でコピーする。ホストが stream を設定せずに計算すると、その仕事も legacy default stream に載るので、読み手がホストの write の後の GPU の仕事を待った（約 0.5 s の仕事で recv が 0.47〜0.48 s。「読み手は書き手を待たない」に反する）。持ち主の device なら読み手は自分の stream でコピーする。失うのは持ち主と別の device の D2D だけ
- **current device が値の device と違うときは、clone を `codec._OnItsDevice` に包む。** torch の `__dlpack__` は current device を要求するため、export の間だけその device を current にする。ホストの write（ホストの current device は保証できない）と、hook の中で別の device を current にしたノードの send の両方（2026-10-25 から。それまでノードの send はホストへ回していた）。読み手の codec は包みを外す
- **コアは device・stream・event を知らない。** メッセージのフィールドは「プールのバッファ」か「不透明な値（`buffer::Held = Arc<dyn Any + Send + Sync>`）」。中身は bindings が入れる `Py<PyAny>`（codec が書き手の device の上で `clone(memory_format=contiguous_format)` した detach 済みのテンソル）。bindings は DLPack の device 種別が CUDA なら shape と dtype を `dlpack::import::check` で照合してオブジェクトごと持つ
- **stream の同期は S2（確認事項 1 の推奨案で確定）。** 書き手が send の中で clone を積んだ自分の stream に event を記録してホストで待ち、それから公開する。読み手は待たない。読み手はコピーを積んだあと元の値に `record_stream(コピーした stream)` を付け、キャッシュアロケータが書き手の次の clone に塊を再利用するのを、コピーが終わるまで止める
- **別の CUDA device の値を読むノード（CPU のノードを含む）は、Driver がその device ごとに副 stream を持ち、hook のたびに current にする。** ATen は device の間のコピーを「呼んだスレッドの、元の device の current stream」に積むため。自分の stream を最後に設定する（`set_stream` が current device を変える）
- **副 stream もノードの stream と同じ device ごと 32 本のプールから取る。** 超えて書き手の stream と重なると、読み手が書き手の公開後の仕事を待つ（正しさは保たれ性能だけ。2026-10-24 の監査で再現、`specs/nodes.md` の CUDA の節に制約として書いた）
- **`on_update` の後の stream の同期は残す。** メッセージの正しさには要らないが、Δt と実際に計算した時間を一致させ、CPU が GPU を追い越して仕事を積み続けないため
- **ノードのデバイスは CPU と CUDA だけ（2026-10-17、ユーザーの決定）。** それ以外の種類（`mps`、`xpu` など）は、torch が使えるかどうかに関わらず `Node(device=...)` が断る（K4）
- ノードの内側で PyTorch を使うときの設定（ノードごとの stream、同期）は Driver が行う（[[project-torch-module]]）

**Why:** 往復は送るたびに同期の D2H を、読むたびに H2D を払う（100 万要素で send が 0.5〜1.6 ms）。device に置いたままなら send は clone と自分の stream の待ちだけになる。S1（読み手の stream が書き手の event を待つ）は 1 send あたり約 12 µs 速いが、読み手の GPU の仕事が書き手の進行中の計算を待ち、「ノードは他のノードを待たない」が GPU の側で崩れ、スロットの時刻も値が出来上がる前になるので採らない。コアに CUDA の意味論を入れず不透明な値にしたのは、torch を import しない原則を保ち、版番号の一斉切り替え・`wait_newer`・寿命の管理を二重に持たないため。ホストからの書き込みを device に置かないのは、torch の `__dlpack__` が current device をテンソルの device に求め、ホストのスレッドの stream と device を保証できないため。MPS は 2 スレッドから同時に使うとプロセスが落ち（torch 2.14.1、GIL ありでも 10/10 で再現）、ノードを複数のスレッドで並行に計算する実行モデルと合わない。

**How to apply:**

- コア・スロット・バッファプールに device・stream・event の概念を入れない。device の判断は bindings（DLPack の device 種別）と codec（torch）に閉じる
- device に置く値を増やす（ホストの書き込みなど）ときは、`__dlpack__` の current device の制約と、読み手の副 stream の device が静的に決まらない問題を先に解く
- ノード・ランタイムに `torch.accelerator` の汎用経路や MPS 向けの分岐を入れない。サポートするデバイスを増やすときはユーザーに確認する
- 仕様は `specs/messages.md` の「コアの中の値」と「stream の同期」、`specs/nodes.md` の「CUDA」

関連: [[project-native-core-rust]], [[project-message-types]], [[project-compute-granularity]]
