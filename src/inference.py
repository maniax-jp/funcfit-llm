"""
時系列予測推論パイプライン

ファインチューニング済みDeepSeek-R1モデルを使用して時系列予測を実行します。
"""

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from unsloth import FastLanguageModel


class TimeSeriesPredictor:
    """時系列予測クラス"""

    def __init__(self, model_path: Path, max_seq_length: int = 2048) -> None:
        """
        Args:
            model_path: ファインチューニング済みモデルのパス
            max_seq_length: 最大シーケンス長
        """
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """モデルとトークナイザーをロード"""
        print(f"モデルをロード中: {self.model_path}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.model_path),
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        # 推論モードに設定
        FastLanguageModel.for_inference(self.model)

        print("✓ モデルロード完了")

    def create_prompt(self, timestamps: list[str], values: list[float]) -> str:
        """
        時系列データから予測用プロンプトを生成

        Args:
            timestamps: タイムスタンプのリスト
            values: 値のリスト

        Returns:
            生成されたプロンプト
        """
        prompt = "以下の時系列データの次の値を予測してください。\n\n"
        prompt += "過去のデータ:\n"

        for ts, val in zip(timestamps, values):
            prompt += f"時刻: {ts}, 値: {val:.4f}\n"

        prompt += "\n次の1ステップの予測値を教えてください。"

        # DeepSeek-R1のチャット形式
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        return formatted_prompt

    def predict(self, prompt: str, max_new_tokens: int = 128) -> str:
        """
        プロンプトから予測を生成

        Args:
            prompt: 入力プロンプト
            max_new_tokens: 最大生成トークン数

        Returns:
            生成された予測テキスト
        """
        # トークナイズ
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # デコード
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # アシスタントの応答部分を抽出
        assistant_response = generated_text.split("assistant")[-1].strip()

        return assistant_response

    def extract_prediction_value(self, response: str) -> float | None:
        """
        応答テキストから予測値を抽出

        Args:
            response: モデルの応答テキスト

        Returns:
            抽出された予測値、または None
        """
        # 数値パターンを探索
        patterns = [
            r"予測値:\s*([-+]?\d*\.?\d+)",
            r"値:\s*([-+]?\d*\.?\d+)",
            r"([-+]?\d+\.\d+)",
            r"([-+]?\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        timestamp_col: str = "timestamp",
        value_col: str = "value",
    ) -> list[dict[str, Any]]:
        """
        DataFrameから時系列予測を実行

        Args:
            df: 入力DataFrame
            sequence_length: 入力シーケンス長
            timestamp_col: タイムスタンプ列名
            value_col: 値列名

        Returns:
            予測結果のリスト
        """
        predictions = []

        # スライディングウィンドウで予測
        for i in range(len(df) - sequence_length):
            window = df.iloc[i : i + sequence_length]
            timestamps = window[timestamp_col].astype(str).tolist()
            values = window[value_col].tolist()

            # プロンプト生成
            prompt = self.create_prompt(timestamps, values)

            # 予測実行
            response = self.predict(prompt)

            # 予測値抽出
            predicted_value = self.extract_prediction_value(response)

            # 実際の次の値（検証用）
            actual_value = df.iloc[i + sequence_length][value_col]

            predictions.append(
                {
                    "index": i + sequence_length,
                    "timestamp": df.iloc[i + sequence_length][timestamp_col],
                    "actual": actual_value,
                    "predicted": predicted_value,
                    "response": response,
                }
            )

            print(
                f"予測 {i+1}/{len(df)-sequence_length}: "
                f"実測={actual_value:.4f}, 予測={predicted_value}"
            )

        return predictions

    def predict_with_chat_template(
        self,
        time_series: list[float],
        system_prompt: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> tuple[str, str]:
        """
        チャットテンプレートを使用して時系列データから予測を生成

        Args:
            time_series: 時系列データのリスト
            system_prompt: システムプロンプト（Noneの場合はデフォルト）
            max_new_tokens: 最大生成トークン数
            temperature: 生成時の温度パラメータ
            top_p: Top-pサンプリングパラメータ

        Returns:
            (生成されたレスポンス, 使用したプロンプト)のタプル
        """
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

        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # トークン化
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # デコード
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # レスポンス部分のみ抽出
        if "<|im_start|>assistant" in generated_text:
            response = generated_text.split("<|im_start|>assistant")[-1].strip()
            response = response.replace("<|im_end|>", "").strip()
        else:
            response = generated_text

        return response, prompt

    def save_predictions(self, predictions: list[dict[str, Any]], output_path: Path) -> None:
        """
        予測結果を保存

        Args:
            predictions: 予測結果のリスト
            output_path: 出力ファイルパス
        """
        df = pd.DataFrame(predictions)
        df.to_csv(output_path, index=False)
        print(f"予測結果を保存: {output_path}")


def main() -> None:
    """コマンドライン実行用のメイン関数"""
    parser = argparse.ArgumentParser(description="時系列予測推論")
    parser.add_argument("--model", type=str, required=True, help="モデルパス")
    parser.add_argument("--input", type=str, required=True, help="入力CSVファイルのパス")
    parser.add_argument(
        "--output", type=str, default="predictions.csv", help="出力CSVファイルのパス"
    )
    parser.add_argument(
        "--sequence-length", type=int, default=10, help="入力シーケンス長"
    )
    parser.add_argument(
        "--timestamp-col", type=str, default="timestamp", help="タイムスタンプ列名"
    )
    parser.add_argument("--value-col", type=str, default="value", help="値列名")

    args = parser.parse_args()

    # 予測器初期化
    predictor = TimeSeriesPredictor(model_path=Path(args.model))

    # モデルロード
    predictor.load_model()

    # データ読み込み
    df = pd.read_csv(args.input)
    print(f"データ読み込み: {len(df)} 行")

    # 予測実行
    predictions = predictor.predict_from_dataframe(
        df,
        sequence_length=args.sequence_length,
        timestamp_col=args.timestamp_col,
        value_col=args.value_col,
    )

    # 結果保存
    predictor.save_predictions(predictions, Path(args.output))


if __name__ == "__main__":
    main()
