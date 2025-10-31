# Unsloth公式Dockerイメージを使用
FROM unsloth/unsloth:latest

# 環境変数設定
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

# 作業ディレクトリ設定
WORKDIR /workspace

# rootユーザーに切り替えてNoto Sansフォントをインストール
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-cjk \
    fontconfig \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

# 追加で必要なパッケージをインストール
RUN pip install --no-cache-dir \
    pyyaml \
    python-dotenv \
    langid \
    matplotlib>=3.7.0 \
    numpy>=1.24.0

# 元のユーザーに戻す（unslothイメージのデフォルト）
USER user

# エントリーポイントを上書き（supervisordをスキップ）
ENTRYPOINT []

# デフォルトコマンド
CMD ["/bin/bash"]
