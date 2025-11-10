"""
シンプルなフォーマットに変換（<think>タグを削除）

ベースモデルが自動的に<think>タグを生成することが確認できたため、
プロンプトと訓練データから<think>タグを削除し、予測結果のみを学習させる。
"""
import json
from pathlib import Path

def convert_sample(sample, prediction_horizon=96):
    """サンプルをシンプルフォーマットに変換"""
    input_values = sample["input_values"]
    target_values = sample["target_values"]

    input_str = ", ".join([f"{v:.4f}" for v in input_values])

    # シンプルなプロンプト（<think>タグの指示を削除）
    instruction = f"""あなたは時系列予測の専門家です。以下の形式に厳密に従って予測を行ってください。

## 出力形式の例

入力: 0.3188, 0.3076, 0.3090, 0.3118
出力:
予測結果（4点）:
0.3100, 0.3085, 0.3095, 0.3110

---

## あなたのタスク

入力系列({len(input_values)}点): {input_str}

「予測結果（{prediction_horizon}点）:」に続けてカンマ区切りの数値列を{prediction_horizon}個出力してください。"""

    # 期待される出力（<think>タグを削除し、予測結果のみ）
    output = f"""予測結果（{len(target_values)}点）:
{", ".join([f"{v:.4f}" for v in target_values])}"""

    return {
        "instruction": instruction,
        "output": output,
        "input_values": input_values,
        "target_values": target_values
    }

# 変換実行
for split in ["train", "val"]:
    input_path = Path(f"data/ett_256_processed_v2/{split}.json")
    output_path = Path(f"data/ett_256_simple/{split}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        data = json.load(f)

    converted = [convert_sample(sample) for sample in data]

    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"✓ {split}: {len(converted)} サンプル変換完了 -> {output_path}")

print("\n変換完了:")
print("  - <think>タグをプロンプトと訓練データから削除")
print("  - 予測結果のみを学習対象とする")
print("  - ベースモデルが自動的に推論を生成することを期待")
