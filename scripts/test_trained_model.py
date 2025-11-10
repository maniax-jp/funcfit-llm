#!/usr/bin/env python3
"""学習済みモデルの推論テストスクリプト"""

import json
import torch
from pathlib import Path
from unsloth import FastLanguageModel

def load_trained_model(model_dir: str):
    """学習済みモデルをロード"""
    print(f"モデルをロード中: {model_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=32768,
        dtype=None,
        load_in_4bit=True,
    )

    # 推論モード
    FastLanguageModel.for_inference(model)

    return model, tokenizer

def test_inference(model, tokenizer, test_samples, num_samples=3):
    """推論テストを実行"""
    print(f"\n=== 推論テスト ({num_samples}サンプル) ===\n")

    for i, sample in enumerate(test_samples[:num_samples]):
        print(f"--- サンプル {i+1} ---")

        instruction = sample["instruction"]
        expected_output = sample["output"]

        # プロンプト作成
        prompt = f"{instruction}"

        # トークナイズ
        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16384
        ).to(model.device)

        print(f"入力トークン数: {inputs['input_ids'].shape[1]}")

        # 推論実行
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16384,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # デコード
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"生成されたトークン数: {outputs[0].shape[0]}")
        print(f"生成テキスト全体の長さ: {len(generated_text)} 文字")
        print(f"\n【生成テキスト全体(最初の500文字)】\n{generated_text[:500]}")

        # 入力部分をスキップして出力のみを取得
        input_length = inputs['input_ids'].shape[1]
        output_tokens = outputs[0][input_length:]
        print(f"出力トークン数: {len(output_tokens)}")
        print(f"出力トークンID: {output_tokens[:20].tolist()}")  # 最初の20トークンを表示

        generated_output = tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f"出力のみをデコードした結果: 「{generated_output[:200]}」")

        print(f"\n【期待される出力】\n{expected_output[:200]}...")
        print(f"\n【モデルの出力】")
        print(f"出力長: {len(generated_output)} 文字")
        print(generated_output if len(generated_output) < 2000 else generated_output[:2000] + "\n... (truncated)")

        # フォーマットチェック
        has_think_tag = "<think>" in generated_output and "</think>" in generated_output
        has_prediction_header = "予測結果" in generated_output

        print(f"\n【フォーマットチェック】")
        print(f"  - <think>タグ: {'✓' if has_think_tag else '✗'}")
        print(f"  - 予測結果ヘッダー: {'✓' if has_prediction_header else '✗'}")
        print()

def main():
    # 学習済みモデルのパス
    model_dir = "models/ett_grpo_256"

    # 検証データをロード
    val_data_path = "data/ett_256_format_focused/val.json"
    print(f"検証データをロード: {val_data_path}")

    with open(val_data_path) as f:
        val_data = json.load(f)

    print(f"検証サンプル数: {len(val_data)}")

    # モデルをロード
    model, tokenizer = load_trained_model(model_dir)

    # 推論テスト
    test_inference(model, tokenizer, val_data, num_samples=3)

    print("\n推論テスト完了")

if __name__ == "__main__":
    main()
