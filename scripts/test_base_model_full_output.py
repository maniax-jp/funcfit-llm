#!/usr/bin/env python3
"""ベースモデル(学習前)の推論テスト - 完全な出力を保存"""

import json
import torch
from pathlib import Path
from unsloth import FastLanguageModel

def load_base_model():
    """学習前のベースモデルをロード"""
    print("ベースモデルをロード中: unsloth/DeepSeek-R1-0528-Qwen3-8B")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/DeepSeek-R1-0528-Qwen3-8B",
        max_seq_length=32768,
        dtype=None,
        load_in_4bit=True,
    )

    # 推論モード
    FastLanguageModel.for_inference(model)

    return model, tokenizer

def test_inference(model, tokenizer, test_samples, output_dir, num_samples=3):
    """推論テストを実行し、完全な出力を保存"""
    print(f"\n=== ベースモデル推論テスト ({num_samples}サンプル) ===\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(test_samples[:num_samples]):
        print(f"--- サンプル {i+1} ---")

        instruction = sample["instruction"]
        expected_output = sample["output"]

        # チャットテンプレートを適用
        messages = [
            {"role": "user", "content": instruction}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # トークナイズ
        inputs = tokenizer(
            prompt,
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

        # 入力部分をスキップして出力のみを取得
        input_length = inputs['input_ids'].shape[1]
        output_tokens = outputs[0][input_length:]
        print(f"出力トークン数: {len(output_tokens)}")

        generated_output = tokenizer.decode(output_tokens, skip_special_tokens=True)

        # 完全な出力を保存
        output_path = output_dir / f"base_model_sample_{i}_full_output.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(generated_output)
        print(f"✓ 完全な出力を保存: {output_path}")

        # プロンプトも保存
        prompt_path = output_dir / f"base_model_sample_{i}_prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"✓ プロンプトを保存: {prompt_path}")

        # 期待される出力も保存
        expected_path = output_dir / f"base_model_sample_{i}_expected.txt"
        with open(expected_path, "w", encoding="utf-8") as f:
            f.write(expected_output)
        print(f"✓ 期待される出力を保存: {expected_path}")

        # フォーマットチェック
        has_think_tag = "<think>" in generated_output and "</think>" in generated_output
        has_prediction_header = "予測結果" in generated_output

        print(f"\n【フォーマットチェック】")
        print(f"  - <think>タグ: {'✓' if has_think_tag else '✗'}")
        print(f"  - 予測結果ヘッダー: {'✓' if has_prediction_header else '✗'}")
        print(f"  - 出力長: {len(generated_output)} 文字")
        print()

def main():
    # 検証データをロード
    val_data_path = "data/ett_256_format_focused/val.json"
    print(f"検証データをロード: {val_data_path}")

    with open(val_data_path) as f:
        val_data = json.load(f)

    print(f"検証サンプル数: {len(val_data)}")

    # ベースモデルをロード
    model, tokenizer = load_base_model()

    # 出力ディレクトリ
    output_dir = Path("outputs/base_model_full_output")

    # 推論テスト
    test_inference(model, tokenizer, val_data, output_dir, num_samples=3)

    print("\n✓ ベースモデル推論テスト完了")
    print(f"✓ 完全な出力は {output_dir} に保存されました")

if __name__ == "__main__":
    main()
