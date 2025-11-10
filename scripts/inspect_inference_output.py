"""
推論出力の詳細確認スクリプト

サンプル0の完全な生成テキストを確認する
"""

import json
import sys
from pathlib import Path

import torch
from unsloth import FastLanguageModel

# ルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_model(model_path: str):
    """モデルとトークナイザーをロード"""
    print(f"モデルをロード中: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=16384,
        dtype=None,
        load_in_4bit=True,
    )

    # 推論モードに設定
    FastLanguageModel.for_inference(model)

    print("✓ モデルロード完了")
    return model, tokenizer


def predict_sequence(model, tokenizer, instruction: str, max_new_tokens: int = 16384):
    """instructionから予測を生成"""
    # チャットテンプレートを適用
    messages = [
        {"role": "user", "content": instruction}
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
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # デコード（特殊トークンを保持）
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # レスポンス部分のみ抽出
    if "<|im_start|>assistant" in generated_text:
        response = generated_text.split("<|im_start|>assistant")[-1].strip()
        response = response.replace("<|im_end|>", "").strip()
    else:
        response = generated_text

    return response, prompt


def main():
    """メイン処理"""
    print("=" * 70)
    print("推論出力の詳細確認")
    print("=" * 70)

    # パス設定
    model_path = "models/ett_grpo_256"
    val_data_path = "data/ett_256_format_focused/val.json"
    output_dir = Path("outputs/inference_inspection_format_focused")
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデルロード
    model, tokenizer = load_model(model_path)

    # 検証データロード
    print(f"\n検証データをロード中: {val_data_path}")
    with open(val_data_path) as f:
        val_data = json.load(f)
    print(f"✓ 検証データロード完了: {len(val_data)} サンプル")

    # 3つのサンプルを詳細検査（先頭、中間、末尾）
    test_indices = [0, 26, 51]

    for idx in test_indices:
        sample = val_data[idx]
        print(f"\n{'=' * 70}")
        print(f"サンプル {idx} を処理中...")
        print(f"{'=' * 70}")

        # 予測実行
        instruction = sample["instruction"]
        response, prompt = predict_sequence(model, tokenizer, instruction)

        # 真値系列を取得
        true_output = sample["output"]

        # プロンプトを保存
        prompt_path = output_dir / f"sample_{idx}_prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"✓ プロンプト保存: {prompt_path}")

        # 完全なレスポンスを保存
        response_path = output_dir / f"sample_{idx}_response.txt"
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"✓ レスポンス保存: {response_path}")

        # 真値を保存
        true_path = output_dir / f"sample_{idx}_true.txt"
        with open(true_path, "w", encoding="utf-8") as f:
            f.write(true_output)
        print(f"✓ 真値保存: {true_path}")

        # レスポンスの先頭500文字を表示
        print(f"\nレスポンス（先頭500文字）:")
        print("-" * 70)
        print(response[:500])
        print("-" * 70)

        # 統計情報
        print(f"\nレスポンス長: {len(response)} 文字")
        print(f"真値長: {len(true_output)} 文字")
        print(f"カンマの数（レスポンス）: {response.count(',')}")
        print(f"カンマの数（真値）: {true_output.count(',')}")

    print("\n" + "=" * 70)
    print("✓ 検査完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
