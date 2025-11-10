"""
時系列データの前処理モジュール

このモジュールは時系列データのクリーニング、正規化、特徴量エンジニアリングを行います。
"""

import argparse
import json
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

    def load_ett_data(
        self,
        csv_path: Path,
        timestamp_col: str = "date",
        target_col: str = "OT"
    ) -> pd.DataFrame:
        """
        ETTデータセットをロード

        Args:
            csv_path: ETT CSVファイルパス
            timestamp_col: タイムスタンプ列名
            target_col: 予測ターゲット列名

        Returns:
            ロード済みDataFrame

        処理内容:
        - date列をdatetime型に変換
        - date列を時系列インデックスに設定
        - OT列を浮動小数点数に変換
        - 欠損値チェック（あればエラー）
        """
        print(f"ETTデータをロード中: {csv_path}")
        df = pd.read_csv(csv_path)

        # date列をdatetime型に変換
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # date列を時系列インデックスに設定
        df = df.set_index(timestamp_col)

        # ターゲット列の型変換
        df[target_col] = df[target_col].astype(float)

        # 欠損値チェック
        if df[target_col].isna().sum() > 0:
            raise ValueError(f"{target_col}列に欠損値があります")

        print(f"✓ ETTデータロード完了: {len(df)}サンプル, {len(df.columns)}列")
        print(f"  時間範囲: {df.index[0]} ～ {df.index[-1]}")

        return df

    def normalize_data(
        self,
        df: pd.DataFrame,
        method: str = "minmax",
        feature_cols: list[str] | None = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        データの正規化

        Args:
            df: 入力DataFrame
            method: 正規化方法（"minmax", "standard", "robust"）
            feature_cols: 正規化する列名リスト（None=OT列のみ）

        Returns:
            (正規化済みDataFrame, スケーラー情報dict)

        スケーラー情報の保存:
        - method, min, max (minmaxの場合)
        - method, mean, std (standardの場合)
        - 推論時の逆変換に使用
        """
        if feature_cols is None:
            feature_cols = ["OT"]

        df_normalized = df.copy()
        scaler_info = {"method": method, "features": {}}

        print(f"データ正規化中: {method}方式")

        for col in feature_cols:
            if col not in df.columns:
                print(f"⚠️ 列が見つかりません: {col}")
                continue

            if method == "minmax":
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
                scaler_info["features"][col] = {
                    "min": min_val,
                    "max": max_val
                }
                print(f"  ✓ {col}: [{min_val:.4f}, {max_val:.4f}] → [0.0, 1.0]")

            elif method == "standard":
                mean_val = float(df[col].mean())
                std_val = float(df[col].std())
                df_normalized[col] = (df[col] - mean_val) / std_val
                scaler_info["features"][col] = {
                    "mean": mean_val,
                    "std": std_val
                }
                print(f"  ✓ {col}: 平均={mean_val:.4f}, 標準偏差={std_val:.4f}")

            elif method == "robust":
                median_val = float(df[col].median())
                q1 = float(df[col].quantile(0.25))
                q3 = float(df[col].quantile(0.75))
                iqr = q3 - q1
                df_normalized[col] = (df[col] - median_val) / iqr
                scaler_info["features"][col] = {
                    "median": median_val,
                    "iqr": iqr
                }
                print(f"  ✓ {col}: 中央値={median_val:.4f}, IQR={iqr:.4f}")

            else:
                raise ValueError(f"未対応の正規化方法: {method}")

        return df_normalized, scaler_info

    def split_train_val_test(
        self,
        df: pd.DataFrame,
        train_months: int = 12,
        val_months: int = 4,
        test_months: int = 4
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        時系列データを訓練/検証/テストに分割

        Args:
            df: 入力DataFrame
            train_months: 訓練データの月数
            val_months: 検証データの月数
            test_months: テストデータの月数

        Returns:
            (train_df, val_df, test_df)

        分割方法:
        - 時系列順に分割（リークなし）
        - train: 最初の12ヶ月（約8,760サンプル）
        - val: 次の4ヶ月（約2,920サンプル）
        - test: 最後の4ヶ月（約2,920サンプル）
        """
        print("データを訓練/検証/テストに分割中...")

        # 1ヶ月あたりのサンプル数を推定（時間粒度: 30日 × 24時間 = 720）
        samples_per_month = 730  # 概算（平均30.4日/月 × 24時間）

        train_size = train_months * samples_per_month
        val_size = val_months * samples_per_month

        # 時系列順に分割
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]

        print(f"  ✓ 訓練データ: {len(train_df)}サンプル ({train_df.index[0]} ～ {train_df.index[-1]})")
        print(f"  ✓ 検証データ: {len(val_df)}サンプル ({val_df.index[0]} ～ {val_df.index[-1]})")
        print(f"  ✓ テストデータ: {len(test_df)}サンプル ({test_df.index[0]} ～ {test_df.index[-1]})")

        return train_df, val_df, test_df

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
    parser.add_argument("--output", type=str, required=True, help="出力ディレクトリのパス")
    parser.add_argument(
        "--timestamp-col", type=str, default="date", help="タイムスタンプ列名（デフォルト: date）"
    )
    parser.add_argument(
        "--target-col", type=str, default="OT", help="予測ターゲット列名（デフォルト: OT）"
    )
    parser.add_argument(
        "--value-cols",
        type=str,
        nargs="+",
        default=None,
        help="正規化対象の列名（複数指定可、デフォルト: target-colのみ）",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["minmax", "standard", "robust"],
        default="minmax",
        help="正規化手法（デフォルト: minmax）",
    )
    parser.add_argument(
        "--ett-mode",
        action="store_true",
        help="ETTデータセット用のモード（訓練/検証/テスト分割を実行）"
    )

    args = parser.parse_args()

    # 出力ディレクトリの作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 前処理実行
    preprocessor = TimeSeriesPreprocessor(scaling_method=args.normalize)

    if args.ett_mode:
        # ETTモード: ロード → 正規化 → 分割
        print("=== ETTデータセット前処理モード ===")

        # 1. データロード
        df = preprocessor.load_ett_data(
            csv_path=Path(args.input),
            timestamp_col=args.timestamp_col,
            target_col=args.target_col
        )

        # 2. 正規化
        value_cols = args.value_cols if args.value_cols else [args.target_col]
        df_normalized, scaler_info = preprocessor.normalize_data(
            df, method=args.normalize, feature_cols=value_cols
        )

        # 3. 訓練/検証/テスト分割
        train_df, val_df, test_df = preprocessor.split_train_val_test(df_normalized)

        # 4. 保存
        # インデックス（date）をリセットして列として保存
        train_df_reset = train_df.reset_index()
        val_df_reset = val_df.reset_index()
        test_df_reset = test_df.reset_index()

        # ファイル名を入力から推測
        input_name = Path(args.input).stem  # "ETTH1" など
        train_path = output_dir / f"{input_name}_train.csv"
        val_path = output_dir / f"{input_name}_val.csv"
        test_path = output_dir / f"{input_name}_test.csv"
        scaler_path = output_dir / f"{input_name}_scaler.json"

        train_df_reset.to_csv(train_path, index=False)
        val_df_reset.to_csv(val_path, index=False)
        test_df_reset.to_csv(test_path, index=False)

        # スケーラー情報をJSON保存
        with open(scaler_path, "w") as f:
            json.dump(scaler_info, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 前処理完了:")
        print(f"  訓練データ: {train_path}")
        print(f"  検証データ: {val_path}")
        print(f"  テストデータ: {test_path}")
        print(f"  スケーラー情報: {scaler_path}")

    else:
        # 通常モード: 既存の処理
        value_cols = args.value_cols if args.value_cols else ["value"]
        preprocessor.preprocess(
            filepath=Path(args.input),
            output_path=output_dir / Path(args.input).name,
            timestamp_col=args.timestamp_col,
            value_columns=value_cols,
        )


if __name__ == "__main__":
    main()
