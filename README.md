[日本語](#comfyui-h3-streaming-v2) | [English](#english-documentation)

---

# ComfyUI H3 Streaming v2

**MiniMax H3 Ref2VA（参照画像から動画を作るモード）を、RTX 4070 12GBで1024×1792・音声付き約3分17秒で生成するComfyUIノードです。同じモデルをスパースなしで動かす場合（約11分12秒）の約3.4倍速です。**

| Ref2VA・1024×1792・124フレーム・4step | 生成時間 |
|---|---:|
| スパースなし（通常のattention） | 671.8秒（11分11.8秒） |
| **本repo：W4A4 + Streaming VSA 5%** | **197.0秒（3分17.0秒）** |
| 参考：ComfyUIネイティブのスパースattention（SLA 5%） | 189.2秒（3分9.2秒） |

[さやかベンチマーク](https://sayakabenchmark.pages.dev/)のRef2VAタブ、case01の実測です（2026-09-29、同日・各1回）。上から順にRef2VA [013](https://sayakabenchmark.pages.dev/benchmarks/ref2va/013-matlow-fused4-dense/workflow.json)、[016](https://sayakabenchmark.pages.dev/benchmarks/ref2va/016-w4a4-streaming-vsa5/workflow.json)、[012](https://sayakabenchmark.pages.dev/benchmarks/ref2va/012-matlow-fused4-sla5/workflow.json)のworkflowです（番号はサイト上の登録番号）。速度だけで見ると、ネイティブのSLA 5%が本repoと同程度かわずかに速い結果でした。

> **より新しい構成：** 1088×1920出力でさらに速いRef2VAの構成として、[ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream)のノードを使うRef2VA [020](https://sayakabenchmark.pages.dev/benchmarks/ref2va/020-matlow-fused4-sla5-x2vae-int8-stream/workflow.json)があります（ネイティブSLA 5%、544×960で生成して1088×1920で保存、68.9秒）。

> 作者：かみもと（Kamimoto）｜検証結果の速報は **[X（@sep_is_heim）](https://x.com/sep_is_heim)**、解説記事は **[note](https://note.com/sepiablue)**（[記事INDEX](https://note.com/sepiablue/n/n8e612c619a3e)）で公開しています。

### 仕組み

- **FC1 W4A4**：各層のFFN前半（FC1）の重みと計算を4bitにします。変換は導入時に一度だけ行い、生成時はCPUから読み込んでComfyUIのDynamic VRAM / offloadを使います。
- **Streaming VSA**：学習済みのGateで重要なattentionだけを計算します（初期値は5%だけ保持）。GateはINT8でCPUに置き、計算中の層の分だけGPUへ送ります。
- 技術的には **FC1 Plain ConvRot W4A4、事前保存したINT8 Gate、固定padding判定キャッシュ** を組み合わせています。付属workflowは832×1408 / 124f / 4stepで実生成を確認しています。

**Windowsでは `setup.bat` で簡単に導入できます。ComfyUI v0.36.0で導入・実生成を確認済みです。** 対応ComfyUIと必要モデルを準備すれば、Python選択・互換性検査・ノード配置・事前変換をまとめて実行できます。モデルの自動ダウンロードは行いません。

### 導入前に知っておきたいこと

| 項目 | 要点 |
|---|---|
| SSD追加容量 | 変換キャッシュ **約5.79 GB（5.39 GiB）**。元モデル・Text Encoder・VAE・出力動画などは別途必要です。 |
| 既存キャッシュの再利用 | `-CacheSource` を指定。同一ボリュームのハードリンクならキャッシュ分の追加消費はほぼありません。別ボリュームではコピーします。 |
| 本リポジトリが追加するノード | **2個**：`H3V2PreconvertedLoader` と `H3V2StreamingVSAPatch`。 |
| 外部ノードの依存 | KJNodesとMotionCache-FastVAEの**2パッケージ**。未導入なら `-InstallDependencies` で不足分を追加できます。各パッケージには本ワークフロー以外のノードも含まれます。 |

### ComfyUI v0.36.0での実測結果

2026-09-20、Windows 11 / RTX 4070 12GBで1回実行。1024×1792、124フレーム、24fps、4 steps、seed 43、res_multistep/simple、Sigma Shift 12/3、ChunkFFN 4、VSA keep 5%、FastVAE batch 2です。

| 指標 | 実測結果 |
|---|---:|
| 生成時間（モデル読み込み込み） | **209.233秒（約3分29秒）** |
| GPU使用量の最大観測値 | **11,479 MiB** |
| プロセスRAMピーク（Windows Peak Working Set） | **11,610.7 MiB** |
| 出力検証 | 音声あり、124フレーム、24fps、FFmpeg全デコード成功 |
| 中断・再実行 | なし |

GPU使用量は60秒間隔と完了時の標本で、瞬間ピークではありません。単発の互換性確認であり、他方式との速度比較や画質評価ではありません。既存キャッシュを全SHA-256照合して再利用したため、この確認では新規50層変換・依存ノード新規インストールは実施していません。詳細は [VALIDATION.md](VALIDATION.md) を参照してください。

このディレクトリの内容を、そのままGitHubリポジトリのルートとして配布できます。
別の評価フォルダや旧リリースへの参照、実験・失敗案・profiling用コードは実行に必要ありません。
モデル、変換済み重み、参照画像、出力動画は同梱しません。

## 構成

- `H3V2PreconvertedLoader`: 元INT8モデルのCPU state_dictへ、50層のFC1 W4A4 shardを重ねて標準ModelPatcherを構築。MODELとキャッシュ接続を出力。
- `H3V2StreamingVSAPatch`: 事前保存済みINT8 GateをCPUで保持し、現在のblockのGateだけGPUへ転送。固定planのpadding判定を一度だけCPUで計算。
- `convert.py`: 既存INT8 ConvRot FC1 → ネイティブConvRot対応BF16復元 → Plain W4A4。Gateも同時に事前保存。
- GUI/API workflow: 同じノード・設定・参照順序。GUI版はComfyUIへドラッグ＆ドロップ可能。

FC2はINT8のままです。QKV、Attention kernel、GateのINT8計算式、ChunkFFN、FastVAE、sampler / schedulerを変更しません。SVDQuantは使用しません。
通常ComfyUIのクラスや既存custom nodeをグローバルに差し替えません。旧版と異なるノードIDなので同居可能です。ただし同じMODELへ旧VSAとv2 VSAを二重適用しないでください。

## 必要な環境

2026-09-20: **ComfyUI 0.36.0、Python 3.13.13、PyTorch 2.14.0+cu130、comfy-kitchen 0.2.34、comfy-aimdo 0.5.5** でセットアップと1024×1792の実生成を確認しました。RTX 4070 12GBでモデル読込を含め209.233秒（単発）。詳細は [VALIDATION.md](VALIDATION.md) を参照してください。

初回検証環境: Windows 11、RTX 4070 12GB、Python 3.13.14、PyTorch 2.13.0+cu130、comfy-kitchen 0.2.33、comfy-aimdo 0.5.2。
ComfyUI検証commitは `15eb748b3ec5f8a0a2d470b7fb280e2d7579f916`。
詳細なファイル指紋は [compatibility.json](compatibility.json) に記録します。
同じversion表示だけではGPU向けバイナリの互換性を保証できないため、後述の `--check` を実行してください。

必要API:

- ComfyUI: `comfy.quant_ops.QUANT_ALGOS['convrot_w4a4']`、native quantized state_dict loader、`comfy_extras.nodes_sparse_attention`。
- comfy-kitchen: `TensorCoreConvRotW4A4Layout`、CUDA `cutlass_int4_dequant`、`sol_attn_chunked`、`int8_linear`。
- この検証版kitchenではnative ConvRot INT4はSM8x（Ampere/Ada）経路です。Hopper/BlackwellやINT8 fallback強制設定は対応対象から除外して停止します。**実測対象はRTX 4070のみ**。他GPUは `--check` に加え実生成で確認してください。
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes): `MiniMaxChunkFeedForward`。
- [ComfyUI-MiniMax-H3-MotionCache-FastVAE](https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE): `MiniMaxH3FastVAEDecode`。
- `torch` / `safetensors` はComfyUIと同じPython環境のものを使用します。

Windows用の `setup.bat` を同梱しています。既存ComfyUIの専用Pythonで検査・導入・一度だけの変換を行います。ComfyUI本体・PyTorch・comfy-kitchenの自動更新やモデルダウンロードは行いません。不足APIがある場合は停止します。

### Windows簡単セットアップ

ComfyUIでの生成を終了してから、リポジトリを取得・展開し、`setup.bat` を実行してください。`custom_nodes` 内へ配置済みならComfyUIを自動検出し、それ以外ではComfyUIフォルダを入力します。コマンドラインからは次のように指定できます。

```bat
setup.bat -ComfyRoot "C:\ComfyUI_windows_portable\ComfyUI"
```

- portableの `python_embeded\python.exe` またはComfyUI内／親フォルダの `.venv\Scripts\python.exe`、ComfyUI内の `venv\Scripts\python.exe` を探します。複数候補がある場合は `-Python "...\python.exe"` で指定してください。グローバルPythonは拒否します。
- ComfyUIの `models` と `extra_model_paths.yaml` から既存モデルを探索します。元DiffusionモデルとGateは `-Model "...safetensors" -Gate "...safetensors"` でも指定できます。Text encoder・VAEは先にComfyUIで使える状態にしてください。
- 外部ノード不足時は、`-InstallDependencies` を付けるとKJNodes / MotionCache-FastVAEの不足分だけを取得し、そのrequirementsを専用Pythonへインストールします。既存ノードは更新しません。通常のセットアップはpipを実行しません。
- 対応API・ネイティブINT4・CPUオフロード／再ロードを確認してから、FC1とGateを `ComfyUI\models\h3_preconverted\fc1_gate` に事前変換します。元Diffusionモデルも生成時に必要です。
- 既存の本プロジェクト形式のキャッシュは `-CacheSource "...\cache"` で指定できます。元モデル・Gateと全shardのSHA-256を照合し、同一ボリュームではハードリンク、それ以外ではコピーします。ハードリンク元／先の重みを直接編集しないでください。
- `-CheckOnly` はGPU検査とモデル／既存キャッシュの検証だけを行い、ファイルを作成しません。キャッシュがない場合はその旨を表示します。不完全・不一致の既存キャッシュは上書きせず停止します。
- 別の場所から再導入するときにインストール済みコードが異なる場合は停止します。差分を確認後に `-Update` を指定すると、このリポジトリの配布ファイルだけを置き換えます。

完了後にComfyUIを再起動し、付属GUIワークフローの参照画像を自分の画像へ変更してください。モデルをサブフォルダに置いている場合は各loaderでも選択してください。

## モデル

付属workflowのファイル名です。Gateは通常のLoRAとして適用しません。

| 用途 | ファイル | ComfyUI内の配置先 / 配布元 |
|---|---|---|
| Diffusion | `minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors` | `models/diffusion_models/`・[MATLOWAI](https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot) |
| VSA Gate | `fasth3_vsa_gate.safetensors` | 変換時にパス指定・[barelymining](https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo) |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/`・[Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| Video VAE | `minimax_h3_video_vae_int8_convrot.safetensors` | `models/vae/`・[Kijai](https://huggingface.co/Kijai/MiniMax-H3-experimental) |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/`・[Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3) |

BF16/FP16元モデルは不要です。変換ツールは上記INT8 ConvRotの50層・形状・キーを前提にし、異なる構造は拒否します。
元Diffusionモデルは、FC1以外の重みを読み込むため生成時にも必要です。変換したキャッシュだけでは動きません。
追加のTurbo/FastH3 LoRAは付属workflowでは使用しません。

## 導入と一度だけの変換

通常は上記の **`setup.bat` を使うだけ**で検査と変換を行えます。以下は手動で実行したい場合の手順です。

1. このリポジトリ全体を `ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-W4A4-VSA/` に配置します。
2. 上記モデルと外部ノードを利用可能にします。起動中の生成と変換がGPUを競合しないよう、生成の終了後に変換してください。
3. **ComfyUI自身のPython**で確認・変換します。以下はWindows portableのルートから実行するPowerShell例です。

```powershell
$h3Python = '.\python_embeded\python.exe'
$h3Root = '.\ComfyUI'
$h3Convert = '.\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-W4A4-VSA\convert.py'

# 数秒のnative INT4 / CPU offload / 再GPUロード確認。ファイル生成なし。
& $h3Python -B -X utf8 $h3Convert --comfy-root $h3Root --check --gpu 0

# 50層を順番に事前変換。出力先は存在しない新しいディレクトリを指定。
& $h3Python -B -X utf8 $h3Convert `
  --comfy-root $h3Root `
  --model '.\ComfyUI\models\diffusion_models\minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors' `
  --gate '.\ComfyUI\models\loras\fasth3_vsa_gate.safetensors' `
  --output '.\ComfyUI\models\h3_preconverted\fc1_gate' --gpu 0

# コピー後・破損が疑われるときの全キャッシュSHA-256検証。
& $h3Python -B -X utf8 $h3Convert --comfy-root $h3Root `
  --verify '.\ComfyUI\models\h3_preconverted\fc1_gate'
```

`--gpu` は変換プロセスだけのCUDA選択です。通常ComfyUIのGPU設定は変えません。
venv環境では `$h3Python` をそのvenvのPythonへ置換します。Linuxは同じ引数を1行で実行できますが、このリリースでは未検証です。

必要な追加ディスク容量は約 **5.8 GB**（FC1約3.86 GB＋Gate約1.93 GB）。RAMはモデル全体・Text Encoder・VAEも使うため、12GB VRAMだけでは容量条件を満たしません。検証PCのRAMは約49GB、実行時モデルstagingは約16GiBでした。
変換時に全50層をGPU常駐させず、1層ずつ保存とGPU往復検証を行います。

完了時は `COMPLETE: FC1 50/50 + Gate 50/50` と表示され、最後に `manifest.json` を書きます。
既存出力先の上書き・削除はしません。途中で失敗した場合、未完了ディレクトリを残して停止します。原因を直し、別の新しい出力先で実行してください。

## Workflowを実行

### スパース率を変更するには

**「H3 v2 Streaming VSA (Preconverted Gate)」ノードの `keep_percent`** を変更します。指定値は削減率ではなく、疎なattentionで保持する割合です。

| `keep_percent` | 保持率 | 対象領域の概算削減率 |
|---:|---:|---:|
| 5（初期値） | 5% | 95% |
| 10 | 10% | 90% |
| 20 | 20% | 80% |

小さいほど疎になります。設定範囲は0.1〜100です。テキスト・参照などのprefix保護領域は別扱いなので、表の割合はモデル全体の計算量削減率ではありません。**変更時の重み再変換は不要**です。値を変更して再度生成してください。

- **GUI:** [workflows/H3_Streaming_v2.json](workflows/H3_Streaming_v2.json)
- **API:** [workflows/H3_Streaming_v2.api.json](workflows/H3_Streaming_v2.api.json)

ComfyUIを再起動してGUI版を読み込み、Reference 1/2/3を自分の画像へ差し替えます（全身・上半身・顔の順）。
`cache_directory=h3_preconverted/fc1_gate` は **ComfyUI/modelsからの相対パス**です。絶対パスも指定できます。
Loaderの `cache` 出力をVSAノードの `cache` 入力へ接続します。モデル名・キャッシュ・workflowをひと組で管理してください。

推奨起動オプションは `--disable-comfy-compiler`。計測環境では `--disable-pinned-memory` も使用しました。
このオプション下でも、本ノードのGateは明示的にpinを試みます。失敗時は通常のCPU tensorを保持します。

付属設定: **832×1408、124 frames、24 fps、seed 43、4 steps、res_multistep / simple、Sigma Shift 12/3、ChunkFFN 4、VSA keep 5%、FastVAE batch 2**。
出力先はComfyUI標準の `output/H3_Streaming_v2_*.mp4` です。

API版はGUI用JSONと異なり、`{"prompt": <API JSON>, "client_id": "..."}` として通常のComfyUI `/prompt` に渡します。
APIでも画像は事前にComfyUIのinputへ配置し、3つのLoadImageのファイル名を変更してください。
サンプルは汎用の参照説明を使っています。評価時の個人用参照画像は非同梱のため、第三者の入力で同一画素や同じ秒数になる保証はありません。

## 確認と制限

- consoleの `[H3 v2]` にFC1 50層、Gate 50層、runtime weight conversion 0sが表示されます。
- `verbose=True` で `H3 v2 sparse producer` を確認します。CUDA/BF16/head_dimやtoken数・sigma範囲が不適合なら、ComfyUIの既存dense経路になります。完走だけではVSA使用の証明にはなりません。
- prefixは常にexact KV / dense-query sinkです。効果のなかった `sink_conditioning` や不採用のpadding行省略の切替は公開していません。
- 元DiffusionモデルまたはGateを変更した場合はキャッシュを新規作成してください。追加LoRAやモデル編集との組合せは未検証です。
- runtimeは高速なheader/サイズ/shape/metadata検査を行い、全ファイルのSHA-256は毎回計算しません。`--verify` に `--model` / `--gate` を追加すると元weightの全内容も照合できます。mtime・ユーザー名・絶対パスに依存しません。
- `ModuleNotFoundError` / API不足は別の対応環境を用意してから再確認してください。この配布物を入れただけで古いComfyUIが対応するわけではありません。
- `ModelMMAP`のアクセスエラーは元モデルとキャッシュの読取権限・共有状態を確認してください。ツールは権限変更や他プロセスの停止を行いません。

性能・検証範囲は [VALIDATION.md](VALIDATION.md) を参照してください。動画の主観的な画質・キャラ再現・音声品質は評価していません。
ソースは [GPL-3.0](LICENSE)、由来と改変点は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。重みには各配布元のライセンスが適用されます。

## 作者・関連リポジトリ

かみもと（Kamimoto）です。RTX 4070 12GBで、MiniMax H3とComfyUIの高速化・省VRAM化を実測しながら公開しています。

- X：[@sep_is_heim](https://x.com/sep_is_heim) — step数・秒数・短縮率などの検証速報
- note：[かみもと｜ローカルAI生成](https://note.com/sepiablue) — 導入方法と実測の解説記事（[記事INDEX](https://note.com/sepiablue/n/n8e612c619a3e)）
- [さやかベンチマーク](https://sayakabenchmark.pages.dev/) — H3ワークフローの速度・画質比較

| リポジトリ | 内容 |
|---|---|
| [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream) | X2 VAEのINT8化と非同期NVENC保存。1088×1920の音声付き5秒動画を約60秒で生成 |
| [ComfyUI-H3-LBH-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-LBH-X2-Stream) | 低解像度3step＋高解像度1step。14秒動画で約25%短縮 |
| [ComfyUI-H3-AudioFirst-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-AudioFirst-X2-Stream) | 音声を先に作って固定し、金属的な響きを抑える2パス生成 |

---

# English Documentation

**ComfyUI nodes that generate MiniMax H3 Ref2VA (reference images → video) at 1024×1792 with audio in about 3 min 17 s on an RTX 4070 12GB, about 3.4× faster than the same model without sparse attention (about 11 min 12 s).**

| Ref2VA, 1024×1792, 124 frames, 4 steps | Generation time |
|---|---:|
| No sparse attention (dense) | 671.8 s (11 min 11.8 s) |
| **This repo: W4A4 + Streaming VSA 5%** | **197.0 s (3 min 17.0 s)** |
| Reference: ComfyUI native sparse attention (SLA 5%) | 189.2 s (3 min 9.2 s) |

Measured on [Sayaka Benchmark](https://sayakabenchmark.pages.dev/), Ref2VA tab, case01 (2026-09-29, same day, one run each). The rows are Ref2VA [013](https://sayakabenchmark.pages.dev/benchmarks/ref2va/013-matlow-fused4-dense/workflow.json), [016](https://sayakabenchmark.pages.dev/benchmarks/ref2va/016-w4a4-streaming-vsa5/workflow.json) and [012](https://sayakabenchmark.pages.dev/benchmarks/ref2va/012-matlow-fused4-sla5/workflow.json) (entry numbers on the site). On speed alone, native SLA 5% was on par with or slightly faster than this repo.

> **Newer configuration:** for faster Ref2VA with 1088×1920 output, see Ref2VA [020](https://sayakabenchmark.pages.dev/benchmarks/ref2va/020-matlow-fused4-sla5-x2vae-int8-stream/workflow.json), which uses the nodes from [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream) (native SLA 5%, 544×960 sampling saved at 1088×1920, 68.9 s).

> Author: Kamimoto (かみもと) | Quick results on **[X (@sep_is_heim)](https://x.com/sep_is_heim)**, write-ups (Japanese) on **[note](https://note.com/sepiablue)** ([article index](https://note.com/sepiablue/n/n8e612c619a3e)).

### How it works

- **FC1 W4A4**: the first FFN projection (FC1) in each block runs with 4-bit weights and activations. Conversion runs once at install time; during generation, weights are loaded from CPU through ComfyUI's Dynamic VRAM / offload path.
- **Streaming VSA**: a trained gate keeps only the important attention blocks (5% by default). The gate is held as INT8 on the CPU and only the current block's gate is sent to the GPU.
- Technically it combines **FC1 Plain ConvRot W4A4, a preconverted INT8 Gate, and a fixed padding decision cache**. The bundled workflow was validated through full generation at 832×1408 / 124 frames / 4 steps.

**Easy Windows installation with `setup.bat`; setup and full generation verified on ComfyUI v0.36.0.** With a compatible ComfyUI installation and the required models available, it handles Python selection, compatibility checks, node installation, and preconversion. Model weights are never downloaded automatically.

### Installation at a glance

| Item | Summary |
|---|---|
| Additional SSD space | **About 5.79 GB (5.39 GiB)** for the converted cache. Source models, text encoder, VAEs, and generated videos require separate space. |
| Reusing a cache | Pass `-CacheSource`. Same-volume hard links consume almost no additional space for the cache; another volume requires a copy. |
| Nodes added by this repository | **Two**: `H3V2PreconvertedLoader` and `H3V2StreamingVSAPatch`. |
| External node dependencies | **Two packages**, KJNodes and MotionCache-FastVAE. `-InstallDependencies` installs only missing packages. These packages also contain nodes unrelated to this workflow. |

### Measured results on ComfyUI v0.36.0

One run on 2026-09-20, Windows 11 / RTX 4070 12GB: 1024×1792, 124 frames, 24 fps, 4 steps, seed 43, res_multistep/simple, Sigma Shift 12/3, ChunkFFN 4, VSA keep 5%, and FastVAE batch 2.

| Metric | Measured result |
|---|---:|
| Generation time, including model loading | **209.233 seconds (about 3 min 29 sec)** |
| Maximum observed GPU memory usage | **11,479 MiB** |
| Process RAM peak, Windows Peak Working Set | **11,610.7 MiB** |
| Output validation | Audio present, 124 frames, 24 fps, full FFmpeg decode passed |
| Interruptions / retries | None |

GPU usage was sampled every 60 seconds and on completion; it is not an instantaneous peak. This is a single compatibility run, not a speed comparison or quality assessment. A fully SHA-256-verified existing cache was reused, so fresh 50-block conversion and fresh dependency installation were not exercised in this run. See [VALIDATION.md](VALIDATION.md) for details.

This directory can be distributed directly as the root of a GitHub repository.
No external benchmark folders, references to older releases, experimental/discarded designs, or profiling code are required for execution.
Model weights, converted caches, reference images, and generated output videos are not included.

## 1. Overview / What This Project Does

- `H3V2PreconvertedLoader`: Overlays 50 FC1 W4A4 shards onto the original INT8 diffusion model state_dict in CPU memory to build a standard ComfyUI ModelPatcher. Outputs `MODEL` and `cache` connection.
- `H3V2StreamingVSAPatch`: Retains the preconverted INT8 Gate in CPU memory and streams only the current block's Gate to GPU. Evaluates per-plan padding decisions once on CPU.
- `convert.py`: Converts existing INT8 ConvRot FC1 → native ConvRot BF16 recovery → Plain W4A4. Simultaneously preconverts and saves the INT8 Gate.
- GUI / API workflows: Identical node layout, parameter settings, and reference order. The GUI version can be dragged & dropped directly into ComfyUI.

FC2 remains INT8. QKV, Attention kernel, Gate INT8 arithmetic, ChunkFFN, FastVAE, and sampler / scheduler are unchanged. SVDQuant is not used.
Standard ComfyUI classes and existing custom nodes are not globally monkey-patched. Unique node IDs prevent collisions with previous versions. Do not apply both legacy VSA and v2 VSA to the same MODEL simultaneously.

## 2. Tested Environment

On 2026-09-20, setup and a complete 1024×1792 generation passed with **ComfyUI 0.36.0, Python 3.13.13, PyTorch 2.14.0+cu130, comfy-kitchen 0.2.34, and comfy-aimdo 0.5.5**. The single RTX 4070 12GB run took 209.233 seconds including model loading. See [VALIDATION.md](VALIDATION.md) for conditions and measurement limits. The original validation environment follows:

- **OS**: Windows 11
- **GPU**: NVIDIA GeForce RTX 4070 12GB
- **Python**: 3.13.14
- **PyTorch**: 2.13.0+cu130
- **comfy-kitchen**: 0.2.33
- **comfy-aimdo**: 0.5.2
- **ComfyUI commit**: `15eb748b3ec5f8a0a2d470b7fb280e2d7579f916`
- Exact file fingerprints are recorded in [compatibility.json](compatibility.json).
- A matching version string alone does not guarantee GPU binary compatibility; run `--check` before conversion.

## 3. Requirements & Dependencies

Required APIs:

- **ComfyUI**: `comfy.quant_ops.QUANT_ALGOS['convrot_w4a4']`, native quantized state_dict loader, `comfy_extras.nodes_sparse_attention`.
- **comfy-kitchen**: `TensorCoreConvRotW4A4Layout`, CUDA `cutlass_int4_dequant`, `sol_attn_chunked`, `int8_linear`.
  - In this tested kitchen build, native ConvRot INT4 follows the SM8x (Ampere/Ada) path. Hopper/Blackwell and forced INT8 fallback modes are unsupported and rejected. **Measured target is RTX 4070 only**. Other GPUs must verify via `--check` and actual generation.
- **External Custom Nodes** (required by workflows):
  - [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes): `MiniMaxChunkFeedForward`.
  - [ComfyUI-MiniMax-H3-MotionCache-FastVAE](https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE): `MiniMaxH3FastVAEDecode`.
- `torch` and `safetensors` must come from the same Python environment used by ComfyUI.

The Windows `setup.bat` installer uses an existing isolated ComfyUI Python. It does not update ComfyUI, PyTorch, or comfy-kitchen, and never downloads model weights. Missing runtime APIs cause a clear failure.

### Windows setup

Finish any active generation, then run `setup.bat` from the downloaded repository. Inside `custom_nodes`, it detects ComfyUI automatically; otherwise it asks for the ComfyUI directory. You can also specify it directly:

```bat
setup.bat -ComfyRoot "C:\ComfyUI_windows_portable\ComfyUI"
```

- Selects the portable `python_embeded` interpreter, a `.venv` in ComfyUI or its parent, or a `venv` inside ComfyUI. Use `-Python "...\python.exe"` when multiple candidates exist. Global Python is rejected.
- Searches existing `models` and `extra_model_paths.yaml`. Use `-Model "...safetensors" -Gate "...safetensors"` for explicit source paths. The text encoder and VAEs must already be available to ComfyUI.
- `-InstallDependencies` clones only missing KJNodes / MotionCache-FastVAE repositories and installs their requirements into the selected isolated Python. Existing nodes are not updated. Without this option, setup never runs pip.
- Checks required APIs, native INT4 execution, and CPU offload/reload, then converts FC1 and the gate once into `ComfyUI\models\h3_preconverted\fc1_gate`. The original diffusion model remains necessary for generation.
- Use `-CacheSource "...\cache"` to reuse an existing cache in this project's format. All shards, the source model, and gate are SHA-256 verified. Same-volume files are hard-linked; cross-volume files are copied. Do not edit hard-linked weights in place.
- `-CheckOnly` checks the GPU, models, and any existing cache without writing files. It reports when conversion is still needed. Incomplete or mismatched caches are rejected and never overwritten.
- When installing from another directory, differing installed code is preserved unless you pass `-Update` after reviewing the differences. Only this repository's distribution files are replaced.

Restart ComfyUI when setup finishes. Open the bundled GUI workflow, select your reference images, and adjust loader model names if your weights are in subdirectories.

## 4. Required Models and Download Sources

File names matching the bundled workflows. The Gate is not applied as a standard LoRA.

| Role | File | ComfyUI Directory / Source |
|---|---|---|
| Diffusion | `minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors` | `models/diffusion_models/` · [MATLOWAI](https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot) |
| VSA Gate | `fasth3_vsa_gate.safetensors` | Specified via path during conversion · [barelymining](https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo) |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` · [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| Video VAE | `minimax_h3_video_vae_int8_convrot.safetensors` | `models/vae/` · [Kijai](https://huggingface.co/Kijai/MiniMax-H3-experimental) |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` · [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3) |

No BF16/FP16 base model is required. The conversion script specifically targets the 50 layers, shapes, and keys of the INT8 ConvRot model above; differing architectures will be rejected.
The original diffusion model is still required during inference to read non-FC1 weights. The converted cache cannot run on its own.
No additional Turbo/FastH3 LoRAs are used in the bundled workflows.

## 5. Simple 4-Step Quick Start

Follow these 4 steps to get up and running:

1. **Place this repository in custom_nodes**
   - Place this repository under `ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-W4A4-VSA/`.
   - Ensure external nodes [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) and [ComfyUI-MiniMax-H3-MotionCache-FastVAE](https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE) are installed.
2. **Make required models available**
   - Place diffusion, text encoder, and VAE weights into their corresponding `ComfyUI/models/` subdirectories as listed in the table above.
   - Place `fasth3_vsa_gate.safetensors` in an accessible path (e.g., `ComfyUI/models/loras/fasth3_vsa_gate.safetensors`).
3. **Run setup.bat**
   - Run `setup.bat` to select ComfyUI's isolated Python, check native INT4 compatibility, and generate or verify `h3_preconverted/fc1_gate`. Add `-InstallDependencies` if the external node packages are missing. Existing models are reused.
4. **Open workflow and generate**
   - Start ComfyUI (recommended: `--disable-comfy-compiler`) and drag & drop [workflows/H3_Streaming_v2.json](workflows/H3_Streaming_v2.json).
   - In the 3 `LoadImage` nodes, select your reference images (Full-body, Upper-body, Face close-up).
   - Verify that the loader's `cache_directory` points to `h3_preconverted/fc1_gate` and click **Queue Prompt**.

## 6. One-Time Conversion Procedure

**`setup.bat` performs these checks and conversion automatically.** The commands below are an alternative for manual operation.

Run conversion using **ComfyUI's own Python environment**. The example below uses PowerShell from the root of a Windows portable ComfyUI setup:

```powershell
$h3Python = '.\python_embeded\python.exe'
$h3Root = '.\ComfyUI'
$h3Convert = '.\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-W4A4-VSA\convert.py'

# 1. Smoke check native INT4 / CPU offload / GPU reload (takes seconds, writes no files)
& $h3Python -B -X utf8 $h3Convert --comfy-root $h3Root --check --gpu 0

# 2. Convert all 50 layers. Output directory must be a new, non-existing path.
& $h3Python -B -X utf8 $h3Convert `
  --comfy-root $h3Root `
  --model '.\ComfyUI\models\diffusion_models\minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors' `
  --gate '.\ComfyUI\models\loras\fasth3_vsa_gate.safetensors' `
  --output '.\ComfyUI\models\h3_preconverted\fc1_gate' --gpu 0

# 3. Verify SHA-256 integrity of all cache shards (useful after copying or suspected corruption)
& $h3Python -B -X utf8 $h3Convert --comfy-root $h3Root `
  --verify '.\ComfyUI\models\h3_preconverted\fc1_gate'
```

- `--gpu` selects the CUDA device exclusively for the conversion process. It does not affect ComfyUI's general settings.
- In a venv environment, replace `$h3Python` with your venv's Python executable.
- Required additional disk space is approximately **5.8 GB** (FC1 ~3.86 GB + Gate ~1.93 GB). RAM requirement: because the full model, text encoder, and VAEs are utilized, 12GB VRAM alone is insufficient. The benchmark system had ~49GB RAM with ~16GiB staging during runtime.
- Layers are converted and roundtrip-verified one by one; all 50 layers are never resident in GPU memory simultaneously.
- Successful completion outputs `COMPLETE: FC1 50/50 + Gate 50/50` and writes `manifest.json`. Existing output directories are never overwritten or deleted. If conversion fails partway, the partial directory is left intact; resolve the issue and specify a new directory.

## 7. Workflow Usage

### Changing sparsity

Change **`keep_percent` in the “H3 v2 Streaming VSA (Preconverted Gate)” node**. This is the percentage retained by sparse attention, not the percentage removed.

| `keep_percent` | Retained | Approximate reduction in the sparse region |
|---:|---:|---:|
| 5 (default) | 5% | 95% |
| 10 | 10% | 90% |
| 20 | 20% | 80% |

Lower values are sparser. The accepted range is 0.1–100. Protected prefix regions, including text and references, are handled separately; these percentages do not describe the reduction in total model computation. **No weight reconversion is needed.** Change the value and generate again.

- **GUI Workflow**: [workflows/H3_Streaming_v2.json](workflows/H3_Streaming_v2.json)
- **API Workflow**: [workflows/H3_Streaming_v2.api.json](workflows/H3_Streaming_v2.api.json)

1. Restart ComfyUI, load the GUI workflow, and swap Reference 1 / 2 / 3 with your own images (full-body, upper-body, face, in that order).
2. `cache_directory=h3_preconverted/fc1_gate` is **relative to `ComfyUI/models`** (absolute paths are also supported).
3. Connect the Loader's `cache` output to the VSA node's `cache` input. Manage model names, caches, and workflows as a unified set.
4. Recommended launch flag is `--disable-comfy-compiler`. The benchmark environment also used `--disable-pinned-memory`. Even under this option, the node explicitly attempts to pin Gate memory (falling back to standard CPU tensors if pinning fails).
5. Default parameters: **832×1408, 124 frames, 24 fps, seed 43, 4 steps, res_multistep / simple, Sigma Shift 12/3, ChunkFFN 4, VSA keep 5%, FastVAE batch 2**.
6. Output videos are saved to ComfyUI's default output directory: `output/H3_Streaming_v2_*.mp4`.
7. For the API workflow, submit `{"prompt": <API JSON>, "client_id": "..."}` to ComfyUI's standard `/prompt` endpoint. Images must be placed in `ComfyUI/input` with matching filenames in LoadImage nodes. Sample workflows use generic prompts; user inputs will produce distinct pixels and timings.

## 8. Validation / Benchmark Results

From [VALIDATION.md](VALIDATION.md), measured on Windows 11, RTX 4070 12GB:

### Benchmark Metrics (832×1408 / 124 frames / 4 steps)

| Metric | Result |
|---|---:|
| Resolution / Frames / FPS | 832×1408 / 124 / 24 |
| Steps / Seed | 4 / 43 |
| FC1 Layers Applied | 50/50 layers |
| FC1 Native INT4 Calls | 800 (16 per layer, all succeeded) |
| FC1 Backing Data / Scale | CPU resident, Dynamic VRAM active |
| Runtime FC1 / Gate Weight Conversion | 0s |
| FC1 Shard Load | 0.0197s |
| Manifest Check + Model Build & Load | 0.1875s |
| Gate CPU Load / Pin | 1.221s |
| Sampler Wall-Clock Time | 99.858s |
| Step Times (1 / 2 / 3 / 4) | 32.067s / 22.496s / 22.601s / 22.673s |
| Steady Step Time (avg steps 2–4) | **22.590s/step** |
| Progress Bar Corrected Display | 90s |
| **Prompt Total (End-to-End)** | **149.704s** *(approx. 2m 30s)* |
| FastVAE Decode (batch 2) | 24.90s |
| Model Staged CPU | 16,320 MiB |
| Video / Audio Latents | Exact finite match with FC1 Preconverted reference |
| Output Media | 832×1408, 124 frames with audio, FFmpeg decoded fully |

### Comparison Across Approaches (Prompt Total)

Earlier development record at 832×1408; the runs were not measured in one session. For a same-day comparison, see the Sayaka Benchmark table at the top.

- **This Configuration (v2 Preconverted FC1 W4A4 + Streaming VSA)**: **149.70s** *(Candidate eval: 146.26s)*
- **Baseline INT8**: **168.44s** *(v2 is ~19s / 11% faster)*
- **Runtime-Convert FC1 W4A4**: **213.28s** *(v2 is ~64s / 30% faster due to elimination of runtime quantization overhead)*

*Note: Measurements are single-run values influenced by OS file cache, load order, and measurement overhead. Small variations should not be interpreted as guaranteed improvements or regressions. Do not compare the progress bar display time (90s) directly with wall-clock time (99.858s).*

## 9. Limitations / Compatibility Notes

- The console displays `[H3 v2]` confirming 50 FC1 layers, 50 Gate layers, and runtime weight conversion of 0s.
- `verbose=True` confirms `H3 v2 sparse producer`. If CUDA, BF16, head_dim, token count, or sigma ranges are incompatible, ComfyUI falls back to the existing dense path. A completed run alone does not prove VSA activation.
- Prefix tokens are permanently exact KV / dense-query sinks. Experimental switches such as alternate `sink_conditioning` or omitted padding rows are not included.
- If the base diffusion model or Gate is modified, create a new cache. Combinations with additional LoRAs or model edits are unverified.
- Runtime uses fast header/size/shape/metadata verification and does not compute SHA-256 for all files on every load. Passing `--model` / `--gate` to `--verify` allows full source hash verification. It does not depend on file modification times, usernames, or absolute paths.
- `ModuleNotFoundError` or missing APIs must be addressed by using a compatible environment; this custom node does not monkey-patch older ComfyUI versions.
- If `ModelMMAP` access errors occur, verify read permissions and sharing state of base models and caches. This tool does not alter permissions or terminate third-party processes.
- Subjective visual quality, character fidelity, and audio quality are not evaluated.

## 10. License / Third-Party Notices

- Distributed source code is licensed under [GPL-3.0](LICENSE).
- Upstream origins and modifications are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Model weights remain subject to their respective upstream licenses, including applicable MiniMax H3 terms. This repository does not distribute or grant redistribution rights for model weights.

## Author and Related Repositories

I'm Kamimoto (かみもと). I measure and publish MiniMax H3 and ComfyUI speed-ups and VRAM reductions on an RTX 4070 12 GB.

- X: [@sep_is_heim](https://x.com/sep_is_heim) — quick results with step counts, seconds and speed-ups
- note: [かみもと｜ローカルAI生成](https://note.com/sepiablue) — setup guides and measured write-ups in Japanese ([article index](https://note.com/sepiablue/n/n8e612c619a3e))
- [Sayaka Benchmark](https://sayakabenchmark.pages.dev/) — speed and quality comparison of H3 workflows

| Repository | What it does |
|---|---|
| [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream) | INT8 X2 VAE and asynchronous NVENC saving; 5 s 1088×1920 video with audio in about 60 s |
| [ComfyUI-H3-LBH-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-LBH-X2-Stream) | 3 low-resolution steps + 1 high-resolution step; about 25% faster for 14 s video |
| [ComfyUI-H3-AudioFirst-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-AudioFirst-X2-Stream) | Audio-first two-pass generation that reduces the metallic voice |
