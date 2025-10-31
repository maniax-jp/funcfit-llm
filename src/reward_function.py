"""
GRPO用報酬関数

時系列予測タスクのための報酬計算モジュール。
予測精度、出力形式、推論の質を総合的に評価します。
"""

import re
from typing import Any

import numpy as np


class TimeSeriesRewardFunction:
    """時系列予測タスク用の報酬関数クラス"""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Args:
            config: 報酬関数の設定辞書
        """
        self.config = config
        self.weights = config.get("weights", {})
        self.accuracy_metrics = config.get("accuracy_metrics", {})
        self.bonuses = config.get("bonuses", {})
        self.penalties = config.get("penalties", {})

    def calculate_reward(
        self,
        generated_text: str,
        true_value: float,
        prompt: str = "",
    ) -> float:
        """
        生成されたテキストから報酬を計算

        Args:
            generated_text: モデルが生成したテキスト
            true_value: 正解の予測値
            prompt: 入力プロンプト（オプション）

        Returns:
            報酬値（スカラー）
        """
        # 各コンポーネントの計算
        accuracy_reward = self._calculate_accuracy_reward(generated_text, true_value)
        format_reward = self._calculate_format_reward(generated_text)
        reasoning_reward = self._calculate_reasoning_reward(generated_text)

        # 重み付け合計
        total_reward = (
            self.weights.get("accuracy", 0.6) * accuracy_reward
            + self.weights.get("format_correctness", 0.2) * format_reward
            + self.weights.get("reasoning_quality", 0.2) * reasoning_reward
        )

        return float(total_reward)

    def _calculate_accuracy_reward(self, generated_text: str, true_value: float) -> float:
        """
        予測精度に基づく報酬を計算

        Args:
            generated_text: 生成テキスト
            true_value: 正解値

        Returns:
            精度報酬（0.0 ~ 1.0 + ボーナス）
        """
        # 生成テキストから予測値を抽出
        predicted_value = self._extract_prediction_value(generated_text)

        if predicted_value is None:
            # 予測値が抽出できない場合は大きなペナルティ
            return self.penalties.get("invalid_output", -1.0)

        # 誤差を計算（正規化スケールを想定）
        error = abs(predicted_value - true_value)

        # 主要メトリクスに基づく報酬計算
        primary_metric = self.accuracy_metrics.get("primary", "mae")
        threshold = self.accuracy_metrics.get("threshold", 0.1)

        if primary_metric == "mae":
            # MAE（Mean Absolute Error）ベース
            # 誤差が小さいほど高い報酬
            accuracy_score = max(0.0, 1.0 - (error / threshold))

        elif primary_metric == "mse":
            # MSE（Mean Squared Error）ベース
            mse = error ** 2
            accuracy_score = max(0.0, 1.0 - (mse / (threshold ** 2)))

        elif primary_metric == "rmse":
            # RMSE（Root Mean Squared Error）ベース
            rmse = np.sqrt(error ** 2)
            accuracy_score = max(0.0, 1.0 - (rmse / threshold))

        else:
            # デフォルトはMAE
            accuracy_score = max(0.0, 1.0 - (error / threshold))

        # 完璧な予測へのボーナス
        if error < 0.001:  # ほぼ完璧
            accuracy_score += self.bonuses.get("perfect_prediction", 1.0)

        return float(accuracy_score)

    def _calculate_format_reward(self, generated_text: str) -> float:
        """
        出力形式の正しさに基づく報酬

        Args:
            generated_text: 生成テキスト

        Returns:
            形式報酬（0.0 ~ 1.0）
        """
        score = 0.0

        # 期待されるキーワードの存在確認
        expected_keywords = ["予測値", "値", "時刻"]

        keyword_count = sum(1 for keyword in expected_keywords if keyword in generated_text)
        score += keyword_count / len(expected_keywords) * 0.5

        # 数値が正しく抽出できるか
        if self._extract_prediction_value(generated_text) is not None:
            score += 0.3

        # 適切な長さ（短すぎず長すぎず）
        text_length = len(generated_text)
        if 20 <= text_length <= 300:
            score += 0.2
        elif text_length < 20:
            score += self.penalties.get("format_error", -0.5) * 0.5

        return float(min(1.0, score))

    def _calculate_reasoning_reward(self, generated_text: str) -> float:
        """
        推論の質に基づく報酬

        Args:
            generated_text: 生成テキスト

        Returns:
            推論報酬（0.0 ~ 1.0 + ボーナス）
        """
        score = 0.0

        # 思考プロセスの存在
        reasoning_indicators = [
            "トレンド",
            "傾向",
            "パターン",
            "増加",
            "減少",
            "周期",
            "変動",
            "推測",
            "考えると",
            "から",
            "ため",
            "理由",
        ]

        reasoning_count = sum(
            1 for indicator in reasoning_indicators if indicator in generated_text
        )

        if reasoning_count > 0:
            score += min(0.6, reasoning_count * 0.15)

        # 論理的な接続詞の使用
        logical_connectors = ["そのため", "したがって", "従って", "よって", "ゆえに"]
        if any(connector in generated_text for connector in logical_connectors):
            score += 0.2

        # 数値的な根拠の提示
        if re.search(r"\d+\.\d+", generated_text):
            score += 0.2

        # 優れた推論へのボーナス
        if score > 0.8:
            score += self.bonuses.get("good_reasoning", 0.5)

        return float(min(1.0, score))

    def _extract_prediction_value(self, text: str) -> float | None:
        """
        テキストから予測値を抽出

        Args:
            text: 生成テキスト

        Returns:
            抽出された予測値、または None
        """
        # 複数のパターンで数値を探索
        patterns = [
            r"予測値[:\s]*(?:は)?[:\s]*([-+]?\d*\.?\d+)",
            r"値[:\s]*(?:は)?[:\s]*([-+]?\d*\.?\d+)",
            r"次の値[:\s]*(?:は)?[:\s]*([-+]?\d*\.?\d+)",
            r"([-+]?\d+\.\d{2,})",  # 小数点以下2桁以上の数値
            r"([-+]?\d+\.\d+)",  # 任意の小数
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    value = float(match.group(1))
                    # 正規化スケール（0-1）の範囲チェック
                    if -0.5 <= value <= 1.5:  # 若干の余裕を持たせる
                        return value
                except (ValueError, IndexError):
                    continue

        return None

    def batch_calculate_rewards(
        self,
        generated_texts: list[str],
        true_values: list[float],
        prompts: list[str] | None = None,
    ) -> list[float]:
        """
        複数の生成テキストに対してバッチで報酬を計算

        Args:
            generated_texts: 生成テキストのリスト
            true_values: 正解値のリスト
            prompts: プロンプトのリスト（オプション）

        Returns:
            報酬のリスト
        """
        if prompts is None:
            prompts = [""] * len(generated_texts)

        rewards = []
        for text, true_val, prompt in zip(generated_texts, true_values, prompts):
            reward = self.calculate_reward(text, true_val, prompt)
            rewards.append(reward)

        return rewards

    def get_reward_statistics(self, rewards: list[float]) -> dict[str, float]:
        """
        報酬の統計情報を取得

        Args:
            rewards: 報酬のリスト

        Returns:
            統計情報の辞書
        """
        rewards_array = np.array(rewards)

        return {
            "mean": float(np.mean(rewards_array)),
            "std": float(np.std(rewards_array)),
            "min": float(np.min(rewards_array)),
            "max": float(np.max(rewards_array)),
            "median": float(np.median(rewards_array)),
        }


def create_reward_function(config: dict[str, Any]) -> TimeSeriesRewardFunction:
    """
    設定から報酬関数インスタンスを作成

    Args:
        config: YAML設定から読み込んだ辞書

    Returns:
        TimeSeriesRewardFunction インスタンス
    """
    reward_config = config.get("reward_function", {})
    return TimeSeriesRewardFunction(reward_config)
