# トラブルシューティングガイド

このドキュメントでは、funcfit-llmプロジェクトで発生する可能性のある問題とその解決方法を説明します。

## 📋 目次

1. [インストール関連](#インストール関連)
2. [GPU/CUDA関連](#gpucuda関連)
3. [メモリ不足エラー](#メモリ不足エラー)
4. [Unsloth関連](#unsloth関連)
5. [データ処理関連](#データ処理関連)
6. [ファインチューニング関連](#ファインチューニング関連)
7. [推論関連](#推論関連)

---

## インストール関連

### 問題: Dockerイメージのビルドが失敗する

**エラーメッセージ:**
```
ERROR: failed to solve: process "/bin/sh -c ..." did not complete successfully
```

**解決方法:**

1. Dockerを最新版に更新
2. ビルドキャッシュをクリア:
```bash
docker compose build --no-cache
```

### 問題: NVIDIA Dockerランタイムが見つからない

**エラーメッセージ:**
```
could not select device driver "" with capabilities: [[gpu]]
```

**解決方法:**

1. NVIDIA Docker Runtimeをインストール:
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

2. docker-compose.ymlでruntimeが正しく設定されているか確認:
```yaml
services:
  funcfit-llm:
    runtime: nvidia
```

---

## GPU/CUDA関連

### 問題: CUDAが認識されない

**確認コマンド:**
```bash
docker compose run --rm funcfit-llm python -c "import torch; print(torch.cuda.is_available())"
```

**結果がFalseの場合:**

1. ホストマシンのNVIDIAドライバーのインストール確認:
```bash
nvidia-smi
```

2. NVIDIA Docker Runtimeが正しくインストールされているか確認:
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

3. docker-compose.ymlで環境変数が正しく設定されているか確認:
```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### 問題: `CUDA out of memory`以外のCUDAエラー

**エラーメッセージ:**
```
RuntimeError: CUDA error: ...
```

**解決方法:**

1. GPUの状態をリセット:
```bash
nvidia-smi --gpu-reset
```

2. Pythonプロセスを完全に終了してから再実行

3. システムを再起動

---

## メモリ不足エラー

### 問題: `CUDA out of memory`

これは最も一般的な問題です。

**解決方法1: バッチサイズの削減**

`configs/training_config.yaml`を編集:

```yaml
training:
  batch_size: 2  # 4から2に減少
  gradient_accumulation_steps: 8  # 4から8に増加
```

**解決方法2: シーケンス長の削減**

```yaml
model:
  max_seq_length: 1024  # 2048から1024に減少
```

**解決方法3: より小規模なモデルを使用**

```yaml
model:
  name: "unsloth/Llama-3.2-1B-Instruct"  # DeepSeek-R1の代わり
```

**解決方法4: 4bit量子化の確認**

```yaml
model:
  load_in_4bit: true  # 必ずtrueに設定
```

### 問題: システムRAM不足

**エラーメッセージ:**
```
MemoryError
```

**解決方法:**

1. 不要なプロセスを終了
2. スワップ領域を増やす（Linux）:
```bash
sudo swapon --show
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Unsloth関連

### 問題: `KeyError: 'align_logprobs_with_mask'`

**エラーメッセージ:**
```
KeyError: 'align_logprobs_with_mask'
```

これはUnslothとtrlのバージョン互換性の問題です。

**解決方法1: trlをダウングレード**

```bash
uv pip install trl==0.23.0
```

**解決方法2: Unslothを最新版に更新**

```bash
uv pip install --upgrade "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**解決方法3: 互換性のある組み合わせを使用**

以下の組み合わせが動作確認済み:
- Unsloth 2025.10.x + trl 0.23.0
- transformers 4.57.x

### 問題: Unslothのインポートエラー

**エラーメッセージ:**
```
ModuleNotFoundError: No module named 'unsloth'
```

**解決方法:**

```bash
# アンインストール
uv pip uninstall unsloth

# 再インストール
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 依存関係も再インストール
uv pip install torchvision xformers
```

---

## データ処理関連

### 問題: `FileNotFoundError` - ファイルが見つからない

**解決方法:**

絶対パスを使用するか、プロジェクトルートから実行:

```bash
cd /path/to/funcfit-llm
uv run python src/data_preprocessing.py --input data/sample_timeseries.csv ...
```

### 問題: データの欠損値エラー

**エラーメッセージ:**
```
ValueError: Input contains NaN, infinity or ...
```

**解決方法:**

データ前処理で自動的に処理されますが、問題が続く場合:

```python
# カスタム前処理
import pandas as pd
df = pd.read_csv('your_data.csv')
df = df.fillna(method='ffill').fillna(method='bfill')
df = df.replace([np.inf, -np.inf], np.nan).dropna()
```

### 問題: データセット構築時のエラー

**エラーメッセージ:**
```
IndexError: list index out of range
```

**原因:** データ行数がシーケンス長より少ない

**解決方法:**

1. より多くのデータを用意
2. シーケンス長を短くする:
```bash
python src/dataset_builder.py --sequence-length 5 ...
```

---

## ファインチューニング関連

### 問題: 学習が収束しない

**症状:**
- Lossが下がらない
- 検証精度が向上しない

**解決方法:**

1. 学習率の調整:
```yaml
training:
  learning_rate: 5.0e-4  # 2.0e-4から増加
```

2. ウォームアップステップを増やす:
```yaml
training:
  warmup_steps: 100  # 10から増加
```

3. エポック数を増やす:
```yaml
training:
  num_epochs: 5  # 3から増加
```

### 問題: 学習が不安定

**症状:**
- Lossが激しく変動
- NaNが発生

**解決方法:**

1. 学習率を下げる:
```yaml
training:
  learning_rate: 1.0e-4
```

2. 勾配クリッピングを有効化:

`src/finetune_grpo.py`のGRPOConfig設定に追加:
```python
grpo_config = GRPOConfig(
    ...
    max_grad_norm=1.0,  # 追加
)
```

### 問題: `RuntimeError: expected scalar type Float but found Half`

**解決方法:**

混合精度設定を確認:

```yaml
training:
  # fp16とbf16の両方をfalseに
  fp16: false
  bf16: false
```

または:

```python
# スクリプト内で明示的に設定
model = model.float()
```

---

## 推論関連

### 問題: 予測値が抽出できない

**症状:**
- `predicted`列がNaNまたはNone

**原因:** モデルの出力形式が期待と異なる

**解決方法:**

`src/inference.py`の`extract_prediction_value`関数を調整:

```python
def extract_prediction_value(self, response: str) -> float | None:
    # より多くのパターンを試す
    patterns = [
        r"予測値[:\s]*([-+]?\d*\.?\d+)",
        r"値[:\s]*([-+]?\d*\.?\d+)",
        r"次の値[:\s]*([-+]?\d*\.?\d+)",
        r"([-+]?\d+\.\d{2,})",  # 小数点以下2桁以上
    ]
    # ...
```

### 問題: 推論が遅い

**解決方法:**

1. バッチ推論の実装（カスタマイズが必要）
2. `max_new_tokens`を減らす:
```bash
python src/inference.py --max-new-tokens 64 ...
```

3. 量子化モデルを使用していることを確認

### 問題: メモリリーク

**症状:**
- 推論を繰り返すとメモリ使用量が増加

**解決方法:**

```python
# 推論ループ内で
torch.cuda.empty_cache()
```

---

## 一般的なデバッグ手順

### 1. 詳細なエラーログを取得

```bash
docker compose run --rm funcfit-llm bash -c "CUDA_LAUNCH_BLOCKING=1 python src/finetune_grpo.py ..."
```

### 2. Pythonデバッガーを使用

```bash
docker compose run --rm funcfit-llm python -m pdb src/finetune_grpo.py ...
```

### 3. 簡略版でテスト

小さなデータセットとモデルで動作確認:

```python
# テスト用の小規模データセット作成
import json
small_data = json.load(open('data/processed/train.json'))[:5]
json.dump(small_data, open('data/test_small.json', 'w'))
```

### 4. ログレベルを上げる

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## サポートとコミュニティ

問題が解決しない場合:

1. **GitHubイシュー**: プロジェクトのイシューを検索または作成
2. **Unsloth Discord**: https://discord.gg/unsloth
3. **Hugging Face Forums**: https://discuss.huggingface.co/
4. **Stack Overflow**: タグ: `transformers`, `pytorch`, `llm`

## 既知の問題

### DeepSeek-R1の制限事項

- 非常に大規模（70B+パラメータ）
- 推奨VRAM: 48GB以上（4bit量子化でも）
- 初回ダウンロード: 100GB以上のストレージ必要

### 代替モデルの推奨

テストや学習には以下の小規模モデルを推奨:

```yaml
model:
  # 選択肢
  name: "unsloth/Llama-3.2-1B-Instruct"       # 1B, 軽量
  name: "unsloth/Llama-3.2-3B-Instruct"       # 3B, バランス型
  name: "unsloth/mistral-7b-v0.3"             # 7B, 高性能
```

---

## チェックリスト

問題が発生したら、以下を順番に確認:

- [ ] GPU/CUDAが正常に動作している
- [ ] 必要な依存関係が全てインストールされている
- [ ] データファイルのパスが正しい
- [ ] メモリ（RAM/VRAM）が十分にある
- [ ] 設定ファイルの値が適切
- [ ] Pythonのバージョンが3.10以上
- [ ] 最新のエラーログを確認

この順番で確認すれば、ほとんどの問題は解決できます!
