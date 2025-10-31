# FuncFit-LLM: DeepSeek-R1 時系列データ予測ファインチューニング

DeepSeek-R1モデルをUnslothライブラリを使用してファインチューニングし、時系列データの予測機能を実装するプロジェクトです。

## 📋 プロジェクト概要

このプロジェクトは、**GRPO (Group Relative Policy Optimization)** を使用してDeepSeek-R1モデルを時系列予測タスクでファインチューニングします。5つの専門報酬関数により、日本語での論理的な推論プロセスと高精度な予測を実現します。

### 主な特徴

- 🚀 **GRPO強化学習**: 複数報酬関数による精密な学習制御
- 🇯🇵 **日本語推論**: langidによる言語検出で日本語回答を強制
- 🤔 **構造化思考**: `<think>`タグによる推論プロセスの可視化
- ⚡ **高速・省メモリ**: Unsloth + QLoRA + 4-bit量子化で効率的に動作
- 📊 **報酬スコア+8.0**: 1300%以上の改善を達成
- 📈 **可視化機能**: matplotlibによる予測結果のグラフ生成
- 🎯 **最適化済み**: シーケンス長16Kで詳細な推論プロセス生成

## 🏗️ プロジェクト構造

```
funcfit-llm/
├── src/                          # メインソースコード
│   ├── reward_functions/         # GRPO報酬関数
│   │   └── grpo_rewards.py       # 5つの専門報酬関数
│   ├── finetune_grpo.py          # GRPOファインチューニング
│   ├── test_inference.py         # 推論テスト
│   ├── evaluate.py               # モデル評価
│   ├── data_preprocessing.py     # データ前処理
│   ├── dataset_builder.py        # GRPO用データセット作成
│   └── inference.py              # 推論パイプライン
├── notebooks/                    # Jupyter実験ノートブック
│   └── experiment.ipynb          # 実験・分析用ノートブック
├── configs/                      # 設定ファイル
│   ├── training_config.yaml      # GRPO学習設定
│   └── training_config_test.yaml # テスト用設定
├── data/                         # データセット格納
│   ├── grpo_processed/           # GRPO用データセット
│   └── sample_timeseries.csv     # サンプルデータ
├── models/                       # 学習済みモデル保存先
├── Dockerfile                    # Docker環境定義
└── docker-compose.yml            # Docker Compose設定
```

## 🚀 クイックスタート

### 前提条件

- Docker & Docker Compose
- NVIDIA GPU (CUDA対応、推奨: 16GB以上のVRAM)
- NVIDIA Docker Runtime

### 1. 環境構築

```bash
# プロジェクトディレクトリに移動
cd funcfit-llm

# Dockerイメージのビルド
docker compose build

# GPU環境の確認
docker compose run --rm funcfit-llm python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## 📊 使用方法

### ステップ1: データ前処理

時系列データをCSV形式で`data/`ディレクトリに配置し、前処理を実行:

```bash
docker compose run --rm funcfit-llm python src/data_preprocessing.py \
    --input data/sample_timeseries.csv \
    --output data/processed_data.csv \
    --value-cols value \
    --scaling minmax
```

### ステップ2: GRPO用データセット構築

```bash
docker compose run --rm funcfit-llm python src/dataset_builder.py \
    --input data/processed_data.csv \
    --output data/grpo_processed \
    --timestamp-col timestamp \
    --value-col value \
    --sequence-length 10 \
    --prediction-horizon 1 \
    --prompt-template grpo \
    --format json \
    --train-ratio 0.8
```

**出力:**
- `data/grpo_processed/train.json` - 訓練データ（69サンプル）
- `data/grpo_processed/val.json` - 検証データ（18サンプル）

**データフォーマット:**
- `instruction`: `<think>`タグ付き推論プロンプト
- `output`: 予測値
- `target_values`: 報酬計算用の真値（必須）
- `input_values`: 入力時系列データ

### ステップ3: GRPOファインチューニング

```bash
# テスト用設定（クイック動作確認、10ステップ・1エポック）
docker compose run --rm funcfit-llm python src/finetune_grpo.py \
    --config configs/training_config_test.yaml \
    --train-data data/grpo_processed/train.json \
    --val-data data/grpo_processed/val.json

# 本番用設定（最適シーケンス長16K、3エポック、QLoRA有効）
docker compose run --rm funcfit-llm python src/finetune_grpo.py \
    --config configs/training_config_production.yaml \
    --train-data data/grpo_processed/train.json \
    --val-data data/grpo_processed/val.json
```

**推定実行時間:**
- テスト用設定: 約10分（10ステップ、1エポック）
- 本番用設定: 約7分（24ステップ、3エポック、69サンプル）

### ステップ4: 推論テスト

学習済みモデルの動作確認を行います。

```bash
# デフォルト設定（models/test_grpo_checkpoint, 3サンプル）
docker compose run --rm funcfit-llm python src/test_inference.py

# カスタム設定
docker compose run --rm funcfit-llm python src/test_inference.py \
    --model models/custom_model \
    --val-data data/grpo_processed/val.json \
    --num-samples 5

# サンプルデータでテスト（検証データなしでも動作）
docker compose run --rm funcfit-llm python src/test_inference.py --use-sample-data

# 可視化付きテスト（グラフ生成）
docker compose run --rm funcfit-llm python src/test_inference.py \
    --model models/grpo_production \
    --num-samples 5 \
    --use-sample-data \
    --save-plots outputs/predictions
```

**オプション:**
- `--model`: モデルパス（デフォルト: `models/test_grpo_checkpoint`）
- `--val-data`: 検証データパス（デフォルト: `data/grpo_processed/val.json`）
- `--num-samples`: テストするサンプル数（デフォルト: 3）
- `--use-sample-data`: 組み込みサンプルデータを使用（上昇・下降・周期・定数・ステップの5パターン）
- `--save-plots`: グラフ保存先ディレクトリ（指定すると入力系列・予測値・真値を可視化）

**出力:**
- 学習済みモデルによる日本語での推論プロセスと予測値
- 可視化グラフ（`--save-plots`指定時）: 入力系列（青実線）、予測値（赤破線）、真値（緑実線）、誤差表示

## 🔧 主要な技術スタック

- **モデル**: DeepSeek-R1-0528-Qwen3-8B (推論特化型)
- **ファインチューニング**:
  - Unsloth (2x高速化 + 70% VRAM削減)
  - GRPO (Group Relative Policy Optimization)
  - **QLoRA** (4-bit量子化 + LoRA、VRAM効率化)
  - 最適シーケンス長: **16384トークン**（16K）
- **深層学習フレームワーク**: PyTorch 2.8, Transformers 4.56
- **データ処理**: Pandas, NumPy
- **可視化**: **matplotlib** (予測結果グラフ生成), Noto Sans CJK (日本語フォント)
- **言語検出**: langid
- **コンテナ**: Docker + NVIDIA Docker Runtime
- **実験管理**: TensorBoard

## 🎉 GRPO実装成果

### 報酬関数による学習制御

5つの専門報酬関数を実装:

1. **format_and_language_reward_func** (+5.0): langidによる日本語検出
2. **check_numbers** (+3.5): 数値抽出の成功判定
3. **match_format_exactly** (+3.0): `</think>`タグの正確な使用
4. **match_format_approximately** (-0.5): タグカウントの評価
5. **check_answer** (±5.0): 真値との数値比較

### トレーニング成果

**初期テスト（シーケンス長2048、1エポック）:**
- **報酬スコア改善**: -0.65 → **+8.0** (1300%以上の向上)
- **安定性**: 全34ステップで一貫した+8.0スコア
- **評価損失**: 1.24e-08 (極めて低い)
- **学習時間**: 約10分 (69サンプル、1エポック)

**本番設定（シーケンス長16384、3エポック + QLoRA）:**
- **最終報酬スコア**: 7.5-8.28（平均8.0前後）
- **学習時間**: 約7分（69サンプル、3エポック、24ステップ）
- **学習可能パラメータ**: 15.3M / 8.2B (0.19%)
- **GPU**: NVIDIA GeForce RTX 5090, 31.8 GB VRAM

### シーケンス長最適化の成果

Phase 2で2048/4096/8192/16384トークンを比較検証:
- **最適値**: **16384トークン**を採用
- **推論品質**: 詳細な`<think>`タグ推論プロセス（1500+トークン）を完全生成
- **トークン使用量**: 入力528 + 出力最大4096 = 4624 ≪ 16384（十分な余裕）

### 特徴

- ✅ 日本語での推論プロセス生成
- ✅ `<think>`タグによる思考過程の構造化
- ✅ チャットテンプレート + システムプロンプト
- ✅ 90%分位点によるプロンプト長フィルタリング
- ✅ 予測結果の可視化（matplotlib + Noto Sans CJK日本語フォント）

## 📈 時系列予測のアプローチ

### プロンプト設計

時系列データを自然言語プロンプトに変換する例:

```
入力プロンプト:
「以下の時系列データの次の値を予測してください。
時刻: 2024-01-01 00:00, 値: 10.5
時刻: 2024-01-01 01:00, 値: 11.2
時刻: 2024-01-01 02:00, 値: 12.1
次の時刻の予測値は?」

期待される出力:
「時刻: 2024-01-01 03:00, 予測値: 13.0」
```

## 🧪 実験とハイパーパラメータチューニング

`notebooks/experiment.ipynb`を使用して、以下の実験が可能です:

- 異なるLoRA設定（rank、alpha）の比較
- 学習率とバッチサイズの最適化
- プロンプトフォーマットのA/Bテスト
- 異なる時系列データセットでの性能評価

## 📝 設定ファイル

### 本番用設定（`configs/training_config_production.yaml`）

Phase 2-4の最適化結果を反映した本番用設定:

```yaml
model:
  name: "unsloth/DeepSeek-R1-0528-Qwen3-8B"
  max_seq_length: 16384  # 最適シーケンス長（Phase 2検証済み）
  load_in_4bit: true
  # QLoRA設定（VRAM削減）
  use_qlora: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: "bfloat16"

lora:
  r: 16
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  lora_alpha: 16
  lora_dropout: 0.05

training:
  output_dir: "./models/grpo_production"
  num_train_epochs: 3
  gradient_accumulation_steps: 8  # 実効バッチサイズ8
  learning_rate: 2.0e-5
  fp16: true
  optim: "adamw_8bit"
```

### テスト用設定（`configs/training_config_test.yaml`）

クイック動作確認用の軽量設定（10ステップ、1エポック）も用意されています。

## 🎯 評価メトリクス

- **MAE** (Mean Absolute Error): 平均絶対誤差
- **RMSE** (Root Mean Square Error): 二乗平均平方根誤差
- **MAPE** (Mean Absolute Percentage Error): 平均絶対パーセント誤差
- **R²スコア**: 決定係数

## 🤝 コントリビューション

プロジェクトへの貢献を歓迎します！

## 📄 ライセンス

MIT License

## 🔗 参考リンク

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [DeepSeek-R1モデル](https://huggingface.co/deepseek-ai)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
