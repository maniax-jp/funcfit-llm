"""
256×96モデルの推論テストと可視化

検証データから数サンプルを選んで推論を実行し、結果を可視化します。
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from unsloth import FastLanguageModel

# ヘッドレス環境用にバックエンド設定
matplotlib.use("Agg")

# ルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import extract_number_from_response

# 日本語フォント設定
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"


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
    """
    instructionから96点の予測系列を生成

    Args:
        model: 学習済みモデル
        tokenizer: トークナイザー
        instruction: 入力プロンプト
        max_new_tokens: 最大生成トークン数

    Returns:
        生成されたレスポンステキスト
    """
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

    # デコード
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # レスポンス部分のみ抽出
    if "<|im_start|>assistant" in generated_text:
        response = generated_text.split("<|im_start|>assistant")[-1].strip()
        response = response.replace("<|im_end|>", "").strip()
    else:
        response = generated_text

    return response


def extract_predicted_sequence(response: str) -> list[float]:
    """
    レスポンスから予測系列を抽出

    「予測結果（N点）:」パターンに続く数値列を優先的に抽出
    """
    import re

    # <think>タグの外側のテキストのみを対象とする
    text = response
    if "<think>" in response and "</think>" in response:
        parts = re.split(r"<think>.*?</think>", response, flags=re.DOTALL)
        text = " ".join(parts)

    # パターン1: 「予測結果（N点）」の後の数値列（コロンあり・なし、コードブロック内も対応）
    pattern1 = r"予測結果[（\(](\d+)点[）\)][:：]?\s*(?:```\s*)?([\d.,\s-]+)"
    match1 = re.search(pattern1, text)
    if match1:
        try:
            # 数値列を抽出（コードブロック内も考慮）
            numbers_text = match1.group(2)
            numbers = [float(x.strip()) for x in numbers_text.replace(" ", "").split(",") if x.strip()]
            if len(numbers) >= 90:
                return numbers[:96]
        except ValueError:
            pass

    # パターン2: カンマ区切りの数値パターンを検索（従来の方法）
    pattern2 = r"([\d.]+(?:\s*,\s*[\d.]+)+)"
    matches = re.findall(pattern2, text)

    for match in matches:
        try:
            numbers = [float(x.strip()) for x in match.split(",")]
            # 96点前後の系列を探す（90-100点の範囲を許容）
            if 90 <= len(numbers) <= 100:
                return numbers[:96]
        except ValueError:
            continue

    return []


def visualize_prediction(
    input_seq: list[float],
    true_seq: list[float],
    pred_seq: list[float],
    output_path: Path,
    title: str = "時系列予測結果 (256→96)"
):
    """
    入力系列、真値系列、予測系列を可視化

    Args:
        input_seq: 入力系列（256点）
        true_seq: 真値系列（96点）
        pred_seq: 予測系列（96点）
        output_path: 保存先パス
        title: グラフタイトル
    """
    plt.figure(figsize=(16, 6))

    # X軸の位置を設定
    x_input = np.arange(len(input_seq))
    x_pred = np.arange(len(input_seq), len(input_seq) + len(true_seq))

    # 入力系列をプロット
    plt.plot(x_input, input_seq, "b-", linewidth=2, label="入力系列 (256点)", alpha=0.8)

    # 真値系列をプロット
    plt.plot(x_pred, true_seq, "g-", linewidth=2, label="真値 (96点)", alpha=0.8)

    # 予測系列をプロット（取得できた場合のみ）
    if pred_seq and len(pred_seq) > 0:
        x_pred_actual = np.arange(len(input_seq), len(input_seq) + len(pred_seq))
        plt.plot(x_pred_actual, pred_seq, "r--", linewidth=2, label=f"予測 ({len(pred_seq)}点)", alpha=0.8)

        # MAE、RMSE、R²を計算（長さが一致する範囲で）
        min_len = min(len(true_seq), len(pred_seq))
        if min_len > 0:
            true_array = np.array(true_seq[:min_len])
            pred_array = np.array(pred_seq[:min_len])

            mae = np.mean(np.abs(true_array - pred_array))
            rmse = np.sqrt(np.mean((true_array - pred_array) ** 2))

            # R²計算
            ss_res = np.sum((true_array - pred_array) ** 2)
            ss_tot = np.sum((true_array - np.mean(true_array)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # メトリクスを表示
            metrics_text = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\n予測点数: {len(pred_seq)}"
            plt.text(
                0.02,
                0.98,
                metrics_text,
                transform=plt.gca().transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.7},
            )

    # 境界線を引く
    plt.axvline(x=len(input_seq), color="gray", linestyle=":", linewidth=1.5, alpha=0.5)

    # グラフ装飾
    plt.xlabel("時刻ステップ", fontsize=12)
    plt.ylabel("値", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ グラフを保存: {output_path}")

    plt.close()


def main():
    """メイン処理"""
    print("=" * 70)
    print("256×96モデルの推論テスト")
    print("=" * 70)

    # パス設定
    model_path = "models/ett_grpo_256"
    val_data_path = "data/ett_256_format_focused/val.json"
    output_dir = Path("outputs/inference_test_256_format_focused")
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデルロード
    model, tokenizer = load_model(model_path)

    # 検証データロード
    print(f"\n検証データをロード中: {val_data_path}")
    with open(val_data_path) as f:
        val_data = json.load(f)
    print(f"✓ 検証データロード完了: {len(val_data)} サンプル")

    # テスト対象サンプルを選択（先頭、中間、末尾の3サンプル）
    test_indices = [0, len(val_data) // 2, len(val_data) - 1]

    print(f"\n推論テストを実行: {len(test_indices)} サンプル")
    print("-" * 70)

    results = []

    for i, idx in enumerate(test_indices, 1):
        sample = val_data[idx]
        print(f"\n[{i}/{len(test_indices)}] サンプル {idx} を処理中...")

        # 予測実行
        instruction = sample["instruction"]
        response = predict_sequence(model, tokenizer, instruction)

        # 予測系列を抽出
        pred_seq = extract_predicted_sequence(response)

        # 真値系列を取得（output文字列から、<think>タグを除去）
        true_output = sample["output"]
        # extract_predicted_sequence関数を再利用して真値も抽出
        true_seq = extract_predicted_sequence(true_output)
        # 抽出できない場合はtarget_valuesから取得
        if not true_seq:
            true_seq = sample.get("target_values", [])

        # 入力系列を取得
        input_seq = sample["input_values"]

        # 結果を可視化
        output_path = output_dir / f"prediction_sample_{idx}.png"
        visualize_prediction(
            input_seq,
            true_seq,
            pred_seq,
            output_path,
            title=f"予測結果 - サンプル {idx}"
        )

        # 結果を保存
        result = {
            "sample_idx": idx,
            "input_length": len(input_seq),
            "true_length": len(true_seq),
            "pred_length": len(pred_seq),
            "response_preview": response[:500] if len(response) > 500 else response,
        }

        # メトリクスを計算（予測系列が取得できた場合）
        if pred_seq and len(pred_seq) > 0:
            min_len = min(len(true_seq), len(pred_seq))
            true_array = np.array(true_seq[:min_len])
            pred_array = np.array(pred_seq[:min_len])

            result["mae"] = float(np.mean(np.abs(true_array - pred_array)))
            result["rmse"] = float(np.sqrt(np.mean((true_array - pred_array) ** 2)))

            ss_res = np.sum((true_array - pred_array) ** 2)
            ss_tot = np.sum((true_array - np.mean(true_array)) ** 2)
            result["r2"] = float(1 - (ss_res / ss_tot) if ss_tot != 0 else 0)

            print(f"  MAE: {result['mae']:.4f}")
            print(f"  RMSE: {result['rmse']:.4f}")
            print(f"  R²: {result['r2']:.4f}")
            print(f"  予測点数: {len(pred_seq)}/96")
        else:
            print("  ⚠ 予測系列を抽出できませんでした")

        results.append(result)

    # 結果サマリーを保存
    summary_path = output_dir / "test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ テスト結果を保存: {summary_path}")

    # 平均メトリクスを計算
    valid_results = [r for r in results if "mae" in r]
    if valid_results:
        avg_mae = np.mean([r["mae"] for r in valid_results])
        avg_rmse = np.mean([r["rmse"] for r in valid_results])
        avg_r2 = np.mean([r["r2"] for r in valid_results])

        print("\n" + "=" * 70)
        print("平均メトリクス")
        print("=" * 70)
        print(f"平均 MAE:  {avg_mae:.4f}")
        print(f"平均 RMSE: {avg_rmse:.4f}")
        print(f"平均 R²:   {avg_r2:.4f}")
        print(f"成功率:    {len(valid_results)}/{len(results)} ({len(valid_results)/len(results)*100:.1f}%)")

    print("\n✓ 推論テスト完了")


if __name__ == "__main__":
    main()
