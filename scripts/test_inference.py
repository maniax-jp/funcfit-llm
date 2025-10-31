#!/usr/bin/env python3
"""
GRPO学習済みモデルの推論テストスクリプト

時系列データから未来値を予測し、推論プロセスを表示します。
"""

import sys
from pathlib import Path
import json
import torch
from unsloth import FastLanguageModel


def load_model_and_tokenizer(model_path: str):
    """学習済みモデルとトークナイザーをロード"""
    print(f"モデルをロード中: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        dtype=None,  # 自動検出
        load_in_4bit=True,
    )

    # 推論モードに設定
    FastLanguageModel.for_inference(model)

    print("✓ モデルロード完了")
    return model, tokenizer


def generate_prediction(model, tokenizer, time_series: list, system_prompt: str = None):
    """時系列データから予測を生成"""

    # デフォルトのシステムプロンプト
    if system_prompt is None:
        system_prompt = "あなたは時系列データを分析し、未来の値を予測する専門家です。与えられたデータから傾向やパターンを見つけ出し、日本語で推論過程を説明してください。"

    # ユーザープロンプト
    user_prompt = f"""以下の時系列データの次の値を予測してください:

データ: {time_series}

ステップ:
1. データの傾向を分析
2. パターンを特定
3. 次の値を予測
4. 予測根拠を説明"""

    # チャットテンプレートを適用
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # トークン化
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # デコード
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # レスポンス部分のみ抽出
    if "<|im_start|>assistant" in generated_text:
        response = generated_text.split("<|im_start|>assistant")[-1].strip()
        response = response.replace("<|im_end|>", "").strip()
    else:
        response = generated_text

    return response, prompt


def main():
    """メイン実行関数"""

    # モデルパス
    model_path = "models/test_grpo_checkpoint"

    # モデルとトークナイザーをロード
    model, tokenizer = load_model_and_tokenizer(model_path)

    # テストデータを読み込み
    test_data_path = Path("data/grpo_processed/val.json")

    if test_data_path.exists():
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        print(f"\n検証データから {min(3, len(test_data))} サンプルをテスト\n")
        print("=" * 80)

        for i, sample in enumerate(test_data[:3], 1):
            print(f"\n【テストケース {i}】")
            print(f"真値: {sample['target_values'][0]}")
            print(f"入力: {sample['instruction'][:100]}...")
            print("-" * 80)

            # 時系列データを抽出（簡易版）
            # 実際のデータ形式に応じて調整が必要
            import re
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", sample['instruction'])
            time_series = [float(n) for n in numbers[:10]]  # 最初の10個

            # 予測生成
            response, prompt = generate_prediction(model, tokenizer, time_series)

            print("【モデルの出力】")
            print(response)
            print("=" * 80)

    else:
        # サンプルデータでテスト
        print("\n検証データが見つからないため、サンプルデータでテスト\n")
        print("=" * 80)

        test_cases = [
            {
                "name": "上昇トレンド",
                "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "expected": 11.0
            },
            {
                "name": "下降トレンド",
                "data": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                "expected": 0.0
            },
            {
                "name": "周期パターン",
                "data": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                "expected": 1.0
            },
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n【テストケース {i}: {test_case['name']}】")
            print(f"データ: {test_case['data']}")
            print(f"期待値: {test_case['expected']}")
            print("-" * 80)

            response, prompt = generate_prediction(model, tokenizer, test_case['data'])

            print("【モデルの出力】")
            print(response)
            print("=" * 80)

    print("\n✓ 推論テスト完了")


if __name__ == "__main__":
    main()
