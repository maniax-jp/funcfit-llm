# Unsloth公式Dockerイメージを使用
FROM unsloth/unsloth:latest

# 環境変数設定
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

# 作業ディレクトリ設定
WORKDIR /workspace

# 追加で必要なパッケージをインストール
RUN pip install --no-cache-dir \
    pyyaml \
    python-dotenv \
    langid

# エントリーポイントを上書き（supervisordをスキップ）
ENTRYPOINT []

# デフォルトコマンド
CMD ["/bin/bash"]
