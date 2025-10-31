"""
時系列データの前処理モジュール

このモジュールは時系列データのクリーニング、正規化、特徴量エンジニアリングを行います。
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class TimeSeriesPreprocessor:
    """時系列データの前処理クラス"""

    def __init__(self, scaling_method: str = "minmax") -> None:
        """
        Args:
            scaling_method: スケーリング手法 ('minmax' または 'standard')
        """
        self.scaling_method = scaling_method
        self.scaler: Optional[MinMaxScaler | StandardScaler] = None

        if scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling_method == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"未対応のスケーリング手法: {scaling_method}")

    def load_data(self, filepath: Path) -> pd.DataFrame:
        """
        CSVファイルから時系列データを読み込む

        Args:
            filepath: データファイルのパス

        Returns:
            読み込んだDataFrame
        """
        df = pd.read_csv(filepath)
        print(f"データ読み込み完了: {len(df)} 行, {len(df.columns)} 列")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データのクリーニング（欠損値処理、外れ値除去など）

        Args:
            df: 入力DataFrame

        Returns:
            クリーニング後のDataFrame
        """
        # 欠損値の確認
        missing_count = df.isnull().sum()
        if missing_count.sum() > 0:
            print(f"欠損値が検出されました:\n{missing_count[missing_count > 0]}")

            # 数値列の欠損値を前方補完
            df = df.fillna(method="ffill")
            # 残った欠損値（最初の行など）を後方補完
            df = df.fillna(method="bfill")
            print("欠損値を補完しました")

        # 重複行の削除
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            print(f"重複行を削除しました: {duplicates} 行")

        return df

    def create_features(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        時系列特徴量の作成

        Args:
            df: 入力DataFrame
            timestamp_col: タイムスタンプ列名

        Returns:
            特徴量追加後のDataFrame
        """
        if timestamp_col in df.columns:
            # タイムスタンプをdatetime型に変換
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

            # 時間関連の特徴量を抽出
            df["hour"] = df[timestamp_col].dt.hour
            df["day_of_week"] = df[timestamp_col].dt.dayofweek
            df["day_of_month"] = df[timestamp_col].dt.day
            df["month"] = df[timestamp_col].dt.month
            df["year"] = df[timestamp_col].dt.year

            print("時間特徴量を追加しました")

        return df

    def scale_data(
        self, df: pd.DataFrame, value_columns: list[str]
    ) -> Tuple[pd.DataFrame, dict]:
        """
        数値データのスケーリング

        Args:
            df: 入力DataFrame
            value_columns: スケーリング対象の列名リスト

        Returns:
            スケーリング後のDataFrameとスケーラー情報の辞書
        """
        df_scaled = df.copy()
        scaler_info = {}

        for col in value_columns:
            if col in df.columns:
                if self.scaler is not None:
                    # 列ごとにスケーラーを作成
                    scaler = (
                        MinMaxScaler()
                        if self.scaling_method == "minmax"
                        else StandardScaler()
                    )
                    df_scaled[col] = scaler.fit_transform(df[[col]])
                    scaler_info[col] = scaler
                    print(f"列 '{col}' をスケーリングしました ({self.scaling_method})")

        return df_scaled, scaler_info

    def create_sequences(
        self, df: pd.DataFrame, sequence_length: int, value_col: str = "value"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        時系列データをシーケンスに変換（教師あり学習用）

        Args:
            df: 入力DataFrame
            sequence_length: シーケンスの長さ
            value_col: 対象の値列名

        Returns:
            入力シーケンス (X) と目標値 (y) のタプル
        """
        values = df[value_col].values
        X, y = [], []

        for i in range(len(values) - sequence_length):
            X.append(values[i : i + sequence_length])
            y.append(values[i + sequence_length])

        X = np.array(X)
        y = np.array(y)

        print(f"シーケンス作成完了: X shape={X.shape}, y shape={y.shape}")
        return X, y

    def preprocess(
        self,
        filepath: Path,
        output_path: Path,
        timestamp_col: str = "timestamp",
        value_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        前処理パイプライン全体の実行

        Args:
            filepath: 入力データのパス
            output_path: 出力データのパス
            timestamp_col: タイムスタンプ列名
            value_columns: スケーリング対象の列名リスト

        Returns:
            前処理後のDataFrame
        """
        # データ読み込み
        df = self.load_data(filepath)

        # クリーニング
        df = self.clean_data(df)

        # 特徴量作成
        df = self.create_features(df, timestamp_col)

        # スケーリング
        if value_columns:
            df, _ = self.scale_data(df, value_columns)

        # 結果を保存
        df.to_csv(output_path, index=False)
        print(f"前処理完了。結果を {output_path} に保存しました")

        return df


def main() -> None:
    """コマンドライン実行用のメイン関数"""
    parser = argparse.ArgumentParser(description="時系列データの前処理")
    parser.add_argument("--input", type=str, required=True, help="入力CSVファイルのパス")
    parser.add_argument("--output", type=str, required=True, help="出力CSVファイルのパス")
    parser.add_argument(
        "--timestamp-col", type=str, default="timestamp", help="タイムスタンプ列名"
    )
    parser.add_argument(
        "--value-cols",
        type=str,
        nargs="+",
        default=["value"],
        help="スケーリング対象の列名（複数指定可）",
    )
    parser.add_argument(
        "--scaling",
        type=str,
        choices=["minmax", "standard"],
        default="minmax",
        help="スケーリング手法",
    )

    args = parser.parse_args()

    # 前処理実行
    preprocessor = TimeSeriesPreprocessor(scaling_method=args.scaling)
    preprocessor.preprocess(
        filepath=Path(args.input),
        output_path=Path(args.output),
        timestamp_col=args.timestamp_col,
        value_columns=args.value_cols,
    )


if __name__ == "__main__":
    main()
