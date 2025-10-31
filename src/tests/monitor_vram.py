"""
VRAM使用量モニタリングスクリプト

学習中のVRAM使用量をリアルタイムで記録し、ピーク値を保存します。
"""

import argparse
import subprocess
import time
from pathlib import Path


def get_vram_usage() -> tuple[float, float]:
    """
    現在のVRAM使用量を取得

    Returns:
        (使用量MB, 総量MB)のタプル
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    used_mb, total_mb = map(float, result.stdout.strip().split(","))
    return used_mb, total_mb


def monitor_vram(output_file: Path, interval: float = 1.0, duration: int | None = None) -> None:
    """
    VRAM使用量を定期的に記録

    Args:
        output_file: 出力ファイルパス
        interval: 測定間隔（秒）
        duration: 測定時間（秒、Noneの場合は無制限）
    """
    print(f"VRAM使用量のモニタリングを開始します（間隔: {interval}秒）")
    if duration:
        print(f"測定時間: {duration}秒")

    measurements = []
    start_time = time.time()

    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time

            if duration and elapsed >= duration:
                break

            used_mb, total_mb = get_vram_usage()
            usage_percent = (used_mb / total_mb) * 100

            measurements.append({
                "timestamp": elapsed,
                "used_mb": used_mb,
                "total_mb": total_mb,
                "usage_percent": usage_percent,
            })

            print(
                f"[{elapsed:7.2f}秒] VRAM: {used_mb:8.2f} MB / {total_mb:8.2f} MB ({usage_percent:5.2f}%)"
            )

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nモニタリングを停止します")

    # 結果を保存
    if measurements:
        peak = max(measurements, key=lambda x: x["used_mb"])
        average_used = sum(m["used_mb"] for m in measurements) / len(measurements)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("VRAM使用量レポート\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"測定回数: {len(measurements)}\n")
            f.write(f"測定時間: {elapsed:.2f}秒\n")
            f.write(f"測定間隔: {interval}秒\n\n")
            f.write("ピーク使用量:\n")
            f.write(f"  時刻: {peak['timestamp']:.2f}秒\n")
            f.write(f"  使用量: {peak['used_mb']:.2f} MB ({peak['usage_percent']:.2f}%)\n\n")
            f.write(f"平均使用量: {average_used:.2f} MB\n\n")
            f.write("詳細データ:\n")
            f.write("-" * 80 + "\n")
            f.write("時刻(秒)\t使用量(MB)\t使用率(%)\n")
            for m in measurements:
                f.write(f"{m['timestamp']:.2f}\t{m['used_mb']:.2f}\t{m['usage_percent']:.2f}\n")

        print(f"\n結果を保存しました: {output_file}")
        print(f"ピークVRAM使用量: {peak['used_mb']:.2f} MB ({peak['usage_percent']:.2f}%)")
        print(f"平均VRAM使用量: {average_used:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="VRAM使用量モニタリング")
    parser.add_argument(
        "--output", type=str, default="vram_usage.txt", help="出力ファイルパス"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0, help="測定間隔（秒）"
    )
    parser.add_argument(
        "--duration", type=int, default=None, help="測定時間（秒、省略時は無制限）"
    )

    args = parser.parse_args()

    monitor_vram(Path(args.output), args.interval, args.duration)


if __name__ == "__main__":
    main()
