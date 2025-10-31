"""
LLM用時系列データセットビルダー

時系列データをLLMのファインチューニングに適したプロンプト形式に変換します。
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset


class TimeSeriesDatasetBuilder:
    """時系列データをLLM用データセットに変換するクラス"""

    def __init__(
        self,
        sequence_length: int = 10,
        prediction_horizon: int = 1,
        prompt_template: str = "default",
    ) -> None:
        """
        Args:
            sequence_length: 入力シーケンスの長さ
            prediction_horizon: 予測する未来のステップ数
            prompt_template: 使用するプロンプトテンプレート
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.prompt_template = prompt_template

    def create_prompt(
        self, timestamps: list[str], values: list[float], context: str = ""
    ) -> str:
        """
        時系列データから入力プロンプトを生成

        Args:
            timestamps: タイムスタンプのリスト
            values: 値のリスト
            context: 追加のコンテキスト情報

        Returns:
            生成されたプロンプト
        """
        if self.prompt_template == "default":
            prompt = "以下の時系列データの次の値を予測してください。\n\n"

            if context:
                prompt += f"コンテキスト: {context}\n\n"

            prompt += "過去のデータ:\n"
            for ts, val in zip(timestamps, values):
                prompt += f"時刻: {ts}, 値: {val:.4f}\n"

            prompt += f"\n次の{self.prediction_horizon}ステップの予測値を教えてください。"

        elif self.prompt_template == "conversational":
            prompt = f"{context}\n" if context else ""
            prompt += "以下は時系列データです:\n"

            for i, (ts, val) in enumerate(zip(timestamps, values), 1):
                prompt += f"{i}. {ts}: {val:.4f}\n"

            prompt += f"\nこのトレンドから、次の値を予測してください。"

        elif self.prompt_template == "numeric_only":
            prompt = "数値シーケンス: " + ", ".join([f"{v:.4f}" for v in values])
            prompt += f"\n次の{self.prediction_horizon}個の値を予測してください。"

        elif self.prompt_template == "reasoning":
            # GRPO用の推論促進テンプレート
            prompt = "あなたは時系列データの予測を専門とするAIアシスタントです。以下のデータを分析し、論理的な推論プロセスを示して次の値を予測してください。\n\n"

            if context:
                prompt += f"コンテキスト: {context}\n\n"

            prompt += "時系列データ:\n"
            for i, (ts, val) in enumerate(zip(timestamps, values), 1):
                prompt += f"{i}. 時刻 {ts}: {val:.4f}\n"

            prompt += "\n<think>\n"
            prompt += "ステップ1: データのトレンドやパターンを分析してください。\n"
            prompt += "ステップ2: 増加傾向か減少傾向かを判断してください。\n"
            prompt += "ステップ3: 過去の変化率から次の値を推論してください。\n"
            prompt += "</think>\n\n"
            prompt += f"上記の分析を基に、次の{self.prediction_horizon}ステップの予測値を論理的根拠とともに教えてください。"

        else:
            raise ValueError(f"未対応のテンプレート: {self.prompt_template}")

        return prompt

    def create_response(self, timestamps: list[str], values: list[float]) -> str:
        """
        予測値から応答テキストを生成

        Args:
            timestamps: 予測時刻のリスト
            values: 予測値のリスト

        Returns:
            生成された応答
        """
        if self.prompt_template == "numeric_only":
            response = ", ".join([f"{v:.4f}" for v in values])
        else:
            response = ""
            for ts, val in zip(timestamps, values):
                response += f"時刻: {ts}, 予測値: {val:.4f}\n"
            response = response.strip()

        return response

    def build_dataset_from_dataframe(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        value_col: str = "value",
        context_col: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        DataFrameからLLM用データセットを構築

        Args:
            df: 入力DataFrame
            timestamp_col: タイムスタンプ列名
            value_col: 値列名
            context_col: コンテキスト列名（オプション）

        Returns:
            プロンプトと応答のペアのリスト
        """
        dataset = []

        # シーケンス長 + 予測ホライズンのウィンドウでスライド
        window_size = self.sequence_length + self.prediction_horizon

        for i in range(len(df) - window_size + 1):
            # 入力部分
            input_window = df.iloc[i : i + self.sequence_length]
            input_timestamps = input_window[timestamp_col].astype(str).tolist()
            input_values = input_window[value_col].tolist()

            # 出力部分（予測対象）
            target_window = df.iloc[
                i + self.sequence_length : i + self.sequence_length + self.prediction_horizon
            ]
            target_timestamps = target_window[timestamp_col].astype(str).tolist()
            target_values = target_window[value_col].tolist()

            # コンテキスト情報
            context = ""
            if context_col and context_col in df.columns:
                context = str(input_window[context_col].iloc[0])

            # プロンプトと応答を生成
            prompt = self.create_prompt(input_timestamps, input_values, context)
            response = self.create_response(target_timestamps, target_values)

            dataset.append(
                {
                    "instruction": prompt,
                    "output": response,
                    "input_values": input_values,
                    "target_values": target_values,
                }
            )

        print(f"データセット構築完了: {len(dataset)} サンプル")
        return dataset

    def save_dataset(
        self, dataset: list[dict[str, Any]], output_path: Path, format: str = "json"
    ) -> None:
        """
        データセットを保存

        Args:
            dataset: データセットのリスト
            output_path: 出力ファイルパス
            format: 保存形式 ('json' または 'huggingface')
        """
        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"データセットをJSON形式で保存: {output_path}")

        elif format == "huggingface":
            # Hugging Face Datasets形式で保存
            hf_dataset = Dataset.from_list(dataset)
            hf_dataset.save_to_disk(str(output_path))
            print(f"データセットをHugging Face形式で保存: {output_path}")

        else:
            raise ValueError(f"未対応の保存形式: {format}")

    def split_dataset(
        self, dataset: list[dict[str, Any]], train_ratio: float = 0.8
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        データセットを訓練用と検証用に分割

        Args:
            dataset: データセット
            train_ratio: 訓練データの割合

        Returns:
            (訓練データ, 検証データ) のタプル
        """
        split_idx = int(len(dataset) * train_ratio)
        train_data = dataset[:split_idx]
        val_data = dataset[split_idx:]

        print(f"データセット分割: 訓練={len(train_data)}, 検証={len(val_data)}")
        return train_data, val_data


def main() -> None:
    """コマンドライン実行用のメイン関数"""
    parser = argparse.ArgumentParser(description="LLM用時系列データセット構築")
    parser.add_argument("--input", type=str, required=True, help="入力CSVファイルのパス")
    parser.add_argument("--output", type=str, required=True, help="出力ディレクトリまたはファイルのパス")
    parser.add_argument(
        "--timestamp-col", type=str, default="timestamp", help="タイムスタンプ列名"
    )
    parser.add_argument("--value-col", type=str, default="value", help="値列名")
    parser.add_argument("--context-col", type=str, default=None, help="コンテキスト列名")
    parser.add_argument(
        "--sequence-length", type=int, default=10, help="入力シーケンスの長さ"
    )
    parser.add_argument(
        "--prediction-horizon", type=int, default=1, help="予測ホライズン"
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        choices=["default", "conversational", "numeric_only", "reasoning"],
        default="default",
        help="プロンプトテンプレート",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "huggingface"],
        default="json",
        help="保存形式",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="訓練データの割合"
    )

    args = parser.parse_args()

    # データ読み込み
    df = pd.read_csv(args.input)
    print(f"データ読み込み: {len(df)} 行")

    # データセットビルダー作成
    builder = TimeSeriesDatasetBuilder(
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        prompt_template=args.prompt_template,
    )

    # データセット構築
    dataset = builder.build_dataset_from_dataframe(
        df,
        timestamp_col=args.timestamp_col,
        value_col=args.value_col,
        context_col=args.context_col,
    )

    # 訓練/検証分割
    train_data, val_data = builder.split_dataset(dataset, train_ratio=args.train_ratio)

    # 保存
    output_path = Path(args.output)
    if args.format == "json":
        output_path.mkdir(parents=True, exist_ok=True)
        builder.save_dataset(train_data, output_path / "train.json", format="json")
        builder.save_dataset(val_data, output_path / "val.json", format="json")
    else:
        builder.save_dataset(train_data, output_path / "train", format="huggingface")
        builder.save_dataset(val_data, output_path / "val", format="huggingface")


if __name__ == "__main__":
    main()
