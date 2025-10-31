#!/usr/bin/env python3
"""
GRPO学習済みモデルの推論テストスクリプト

学習完了後の動作確認用スクリプト。
src/inference.pyのTimeSeriesPredictorクラスを使用して推論を実行します。
"""

import argparse
import json
import re
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import TimeSeriesPredictor, extract_number_from_response


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description="GRPO学習済みモデルの推論テスト"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/test_grpo_checkpoint",
        help="学習済みモデルのパス（デフォルト: models/test_grpo_checkpoint）",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="data/grpo_processed/val.json",
        help="検証データのパス（デフォルト: data/grpo_processed/val.json）",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="テストするサンプル数（デフォルト: 3）",
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="検証データの代わりにサンプルデータを使用",
    )
    parser.add_argument(
        "--save-plots",
        type=str,
        default=None,
        help="グラフを保存するディレクトリ（指定した場合、各テストケースのグラフを保存）",
    )

    args = parser.parse_args()

    # TimeSeriesPredictorを初期化
    predictor = TimeSeriesPredictor(model_path=Path(args.model), max_seq_length=1024)
    predictor.load_model()

    # プロット保存ディレクトリの準備
    plots_dir = None
    if args.save_plots:
        plots_dir = Path(args.save_plots)
        plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"グラフ保存先: {plots_dir}\n")

    # テストデータを読み込み
    test_data_path = Path(args.val_data)

    if test_data_path.exists() and not args.use_sample_data:
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        num_samples = min(args.num_samples, len(test_data))
        print(f"\n検証データから {num_samples} サンプルをテスト\n")
        print("=" * 80)

        for i, sample in enumerate(test_data[:num_samples], 1):
            print(f"\n【テストケース {i}】")
            print(f"真値: {sample['target_values'][0]}")
            print(f"入力: {sample['instruction'][:100]}...")
            print("-" * 80)

            # 時系列データを抽出
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", sample['instruction'])
            time_series = [float(n) for n in numbers[:10]]  # 最初の10個

            # 予測生成（src/inference.pyのメソッドを使用）
            response, _ = predictor.predict_with_chat_template(time_series)

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

            # 予測生成（src/inference.pyのメソッドを使用）
            response, _ = predictor.predict_with_chat_template(test_case['data'])

            print("【モデルの出力】")
            print(response)
            print("=" * 80)

            # グラフ保存（オプション）
            if plots_dir:
                predicted_value = extract_number_from_response(response)
                if predicted_value is not None:
                    output_path = plots_dir / f"test_case_{i}_{test_case['name']}.png"
                    predictor.visualize_prediction(
                        input_series=test_case['data'],
                        predicted_value=predicted_value,
                        true_value=test_case['expected'],
                        output_path=output_path,
                        title=f"テストケース {i}: {test_case['name']}",
                    )
                else:
                    print(f"⚠️ 予測値を抽出できませんでした（テストケース {i}）")

    print("\n✓ 推論テスト完了")


if __name__ == "__main__":
    main()
