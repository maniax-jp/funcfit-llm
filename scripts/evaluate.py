"""
モデル評価スクリプト

ファインチューニング済みモデルの予測性能を評価します。
"""

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    """モデル評価クラス"""

    def __init__(self) -> None:
        """評価クラスの初期化"""
        self.metrics: dict[str, float] = {}

    def load_predictions(self, predictions_path: Path) -> pd.DataFrame:
        """
        予測結果を読み込む

        Args:
            predictions_path: 予測結果CSVファイルのパス

        Returns:
            予測結果のDataFrame
        """
        df = pd.read_csv(predictions_path)
        print(f"予測データ読み込み: {len(df)} 行")
        return df

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        評価メトリクスを計算

        Args:
            y_true: 実測値
            y_pred: 予測値

        Returns:
            メトリクスの辞書
        """
        # 欠損値を除去
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            print("警告: 有効な予測値がありません")
            return {}

        # メトリクス計算
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_clean, y_pred_clean)

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100

        self.metrics = {
            "MAE (Mean Absolute Error)": mae,
            "MSE (Mean Squared Error)": mse,
            "RMSE (Root Mean Squared Error)": rmse,
            "R² Score": r2,
            "MAPE (%)": mape,
        }

        return self.metrics

    def print_metrics(self) -> None:
        """メトリクスを表示"""
        print("\n" + "=" * 50)
        print("評価メトリクス")
        print("=" * 50)

        for name, value in self.metrics.items():
            print(f"{name:35s}: {value:.6f}")

        print("=" * 50 + "\n")

    def plot_predictions(
        self,
        df: pd.DataFrame,
        output_path: Path | None = None,
        max_points: int = 100,
    ) -> None:
        """
        予測値と実測値をプロット

        Args:
            df: 予測結果DataFrame
            output_path: 保存先パス（Noneの場合は表示のみ）
            max_points: プロットする最大点数
        """
        # データが多い場合はサンプリング
        if len(df) > max_points:
            df_plot = df.sample(n=max_points, random_state=42).sort_index()
        else:
            df_plot = df

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # 時系列プロット
        axes[0].plot(df_plot.index, df_plot["actual"], label="実測値", marker="o", alpha=0.7)
        axes[0].plot(
            df_plot.index, df_plot["predicted"], label="予測値", marker="x", alpha=0.7
        )
        axes[0].set_xlabel("インデックス")
        axes[0].set_ylabel("値")
        axes[0].set_title("時系列予測: 実測値 vs 予測値")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 散布図
        axes[1].scatter(df_plot["actual"], df_plot["predicted"], alpha=0.6)
        axes[1].plot(
            [df_plot["actual"].min(), df_plot["actual"].max()],
            [df_plot["actual"].min(), df_plot["actual"].max()],
            "r--",
            label="理想線",
        )
        axes[1].set_xlabel("実測値")
        axes[1].set_ylabel("予測値")
        axes[1].set_title("予測精度: 実測値 vs 予測値")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"プロットを保存: {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_error_distribution(
        self, df: pd.DataFrame, output_path: Path | None = None
    ) -> None:
        """
        誤差の分布をプロット

        Args:
            df: 予測結果DataFrame
            output_path: 保存先パス
        """
        # 誤差計算
        errors = df["predicted"] - df["actual"]
        errors = errors.dropna()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # ヒストグラム
        axes[0].hist(errors, bins=30, edgecolor="black", alpha=0.7)
        axes[0].axvline(0, color="red", linestyle="--", label="誤差=0")
        axes[0].set_xlabel("予測誤差")
        axes[0].set_ylabel("頻度")
        axes[0].set_title("誤差分布")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot(errors, vert=True)
        axes[1].set_ylabel("予測誤差")
        axes[1].set_title("誤差のボックスプロット")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"誤差分布プロットを保存: {output_path}")
        else:
            plt.show()

        plt.close()

    def evaluate(
        self,
        predictions_path: Path,
        output_dir: Path | None = None,
    ) -> dict[str, float]:
        """
        評価パイプライン全体を実行

        Args:
            predictions_path: 予測結果CSVのパス
            output_dir: 出力ディレクトリ

        Returns:
            評価メトリクスの辞書
        """
        # 予測データ読み込み
        df = self.load_predictions(predictions_path)

        # メトリクス計算
        y_true = df["actual"].values
        y_pred = df["predicted"].values
        metrics = self.calculate_metrics(y_true, y_pred)

        # メトリクス表示
        self.print_metrics()

        # プロット作成
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

            self.plot_predictions(
                df, output_path=output_dir / "predictions_plot.png"
            )
            self.plot_error_distribution(
                df, output_path=output_dir / "error_distribution.png"
            )

            # メトリクスをCSVとして保存
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(output_dir / "metrics.csv", index=False)
            print(f"メトリクスを保存: {output_dir / 'metrics.csv'}")

        return metrics


def main() -> None:
    """コマンドライン実行用のメイン関数"""
    parser = argparse.ArgumentParser(description="モデル評価")
    parser.add_argument(
        "--predictions", type=str, required=True, help="予測結果CSVファイルのパス"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="出力ディレクトリ",
    )

    args = parser.parse_args()

    # 評価実行
    evaluator = ModelEvaluator()
    evaluator.evaluate(
        predictions_path=Path(args.predictions),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
