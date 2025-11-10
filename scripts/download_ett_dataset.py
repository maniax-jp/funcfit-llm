#!/usr/bin/env python3
"""
ETTデータセットのダウンロードスクリプト

Electricity Transformer Temperature (ETT) データセットをHugging Faceから
ダウンロードし、CSV形式で保存します。
"""

import argparse
from pathlib import Path
import pandas as pd
import urllib.request
import ssl


class ETTDatasetDownloader:
    """ETTデータセットのダウンロードと保存を管理"""

    def __init__(self, output_dir: Path = Path("data/raw")):
        """
        Args:
            output_dir: 保存先ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        dataset_name: str = "ett",
        variant: str = "h1"
    ) -> Path:
        """
        ETTデータセットをダウンロード

        Args:
            dataset_name: データセット名（"ett"固定）
            variant: 変種（"h1", "h2", "m1", "m2"）
                - h1, h2: 時間粒度（1時間間隔）
                - m1, m2: 分粒度（15分間隔）

        Returns:
            保存されたCSVファイルのパス

        Raises:
            ValueError: データセットのロードに失敗した場合
        """
        print(f"ETTデータセットをダウンロード中: {variant}")
        print(f"データソース: GitHub (zhouhaoyi/ETDataset)")

        # GitHubの生データURL
        variant_upper = variant.upper()
        base_url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"

        # 正しいファイル名にマッピング
        filename_map = {
            "h1": "ETTh1.csv",
            "h2": "ETTh2.csv",
            "m1": "ETTm1.csv",
            "m2": "ETTm2.csv",
        }
        csv_filename = filename_map.get(variant, f"ETT{variant}.csv")
        url = f"{base_url}/{csv_filename}"

        print(f"URL: {url}")

        # CSVファイルのパス
        csv_path = self.output_dir / f"ETT{variant_upper}.csv"

        try:
            # SSL証明書の検証を無効化（GitHub接続用）
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # ダウンロード
            print("ダウンロード中...")
            urllib.request.urlretrieve(url, csv_path)

            # ファイルサイズを表示
            file_size_mb = csv_path.stat().st_size / (1024 * 1024)
            print(f"✓ 保存完了: {csv_path} ({file_size_mb:.2f} MB)")

        except Exception as e:
            raise ValueError(f"データセットのダウンロードに失敗: {e}")

        # 検証を実行
        if self.validate(csv_path):
            print("✓ データ検証成功")
        else:
            print("⚠️ データ検証で警告が発生しました")

        return csv_path

    def validate(self, csv_path: Path) -> bool:
        """
        ダウンロードしたデータの検証

        Args:
            csv_path: CSVファイルパス

        Returns:
            検証結果（True=正常）

        検証項目:
        - ファイルが存在すること
        - 8カラム存在すること (date + 7特徴量)
        - date列がタイムスタンプ形式であること
        - OT列（予測ターゲット）に欠損値がないこと
        - サンプル数が妥当であること（時間粒度: 17,520、分粒度: 70,080）
        """
        if not csv_path.exists():
            print(f"❌ ファイルが存在しません: {csv_path}")
            return False

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ CSVの読み込みに失敗: {e}")
            return False

        # 検証結果
        is_valid = True

        # 1. 列数チェック
        expected_columns = 8
        if len(df.columns) != expected_columns:
            print(f"⚠️ 列数が不正: {len(df.columns)} (期待値: {expected_columns})")
            is_valid = False
        else:
            print(f"✓ 列数: {len(df.columns)}")

        # 2. 列名チェック
        expected_col_names = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        if list(df.columns) != expected_col_names:
            print(f"⚠️ 列名が期待と異なります")
            print(f"  期待: {expected_col_names}")
            print(f"  実際: {list(df.columns)}")
            is_valid = False
        else:
            print(f"✓ 列名: {', '.join(df.columns)}")

        # 3. サンプル数チェック
        num_samples = len(df)
        print(f"✓ サンプル数: {num_samples:,}")

        # 時間粒度（h1, h2）: 2年 × 365日 × 24時間 = 17,520
        # 分粒度（m1, m2）: 2年 × 365日 × 24時間 × 4 = 70,080
        if num_samples not in [17520, 69680, 70080]:
            print(f"⚠️ サンプル数が想定外: {num_samples}")
            print(f"  期待: 17,520 (時間粒度) または 69,680-70,080 (分粒度)")

        # 4. date列の型チェック
        try:
            pd.to_datetime(df["date"])
            print("✓ date列: datetime形式に変換可能")
        except Exception as e:
            print(f"⚠️ date列がdatetime形式に変換できません: {e}")
            is_valid = False

        # 5. OT列の欠損値チェック
        if "OT" in df.columns:
            num_missing = df["OT"].isna().sum()
            if num_missing > 0:
                print(f"⚠️ OT列に欠損値があります: {num_missing}個")
                is_valid = False
            else:
                print("✓ OT列: 欠損値なし")
        else:
            print("⚠️ OT列が見つかりません")
            is_valid = False

        # 6. 数値列の範囲チェック
        numeric_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        for col in numeric_cols:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                print(f"✓ {col}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}")

        return is_valid


def main():
    """コマンドライン実行のエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="ETTデータセットをHugging Faceからダウンロード"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="h1",
        choices=["h1", "h2", "m1", "m2"],
        help="データセットの変種（デフォルト: h1）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="保存先ディレクトリ（デフォルト: data/raw）"
    )

    args = parser.parse_args()

    # ダウンロード実行
    downloader = ETTDatasetDownloader(output_dir=Path(args.output))

    try:
        csv_path = downloader.download(dataset_name="ett", variant=args.variant)
        print(f"\n✅ ダウンロード完了: {csv_path}")
        print(f"\n次のステップ:")
        print(f"  python src/data_preprocessing.py --input {csv_path} --output data/processed")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
