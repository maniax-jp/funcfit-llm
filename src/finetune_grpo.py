"""
DeepSeek-R1-Qwen3-8B GRPO強化学習スクリプト

Group Relative Policy Optimization (GRPO) を使用して
時系列予測タスクでモデルを強化学習します。
"""

# Unslothを最初にインポート（最適化のため必須）
from unsloth import FastLanguageModel, is_bfloat16_supported

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from transformers import BitsAndBytesConfig, TrainingArguments
from trl import GRPOConfig, GRPOTrainer

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.reward_functions import (
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers,
    format_and_language_reward_func,
    setup_special_tokens,
)


class DeepSeekGRPOTrainer:
    """DeepSeek-R1-Qwen3-8BのGRPO学習クラス"""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Args:
            config: 学習設定の辞書
        """
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Unslothを使用してモデルとトークナイザーをロード"""
        model_config = self.config["model"]

        print(f"モデルをロード中: {model_config['name']}")

        # QLoRA設定の準備
        quantization_config = None
        if model_config.get("use_qlora", False):
            print("QLoRA設定を有効化中...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=model_config.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_compute_dtype=getattr(torch, model_config.get("bnb_4bit_compute_dtype", "bfloat16")),
            )
            print(f"✓ QLoRA設定: {quantization_config.bnb_4bit_quant_type}, double_quant={quantization_config.bnb_4bit_use_double_quant}")

        # Unslothの高速モデルロード
        if quantization_config:
            # QLoRA有効時はquantization_configを使用
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_config["name"],
                max_seq_length=model_config.get("max_seq_length", 2048),
                dtype=None,  # quantization_configが優先
                load_in_4bit=True,
            )
        else:
            # QLoRA無効時は従来の設定
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_config["name"],
                max_seq_length=model_config.get("max_seq_length", 2048),
                dtype=model_config.get("dtype", None),
                load_in_4bit=model_config.get("load_in_4bit", True),
            )

        print("✓ モデルロード完了")

    def setup_lora(self) -> None:
        """LoRA設定を適用"""
        lora_config = self.config["lora"]

        print("LoRA設定を適用中...")

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_config.get("r", 16),
            target_modules=lora_config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
            ),
            lora_alpha=lora_config.get("alpha", 16),
            lora_dropout=lora_config.get("dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            use_gradient_checkpointing=lora_config.get("gradient_checkpointing", "unsloth"),
            random_state=3407,
        )

        print("✓ LoRA設定完了")

        # 特殊トークンのセットアップ
        special_tokens = setup_special_tokens(self.tokenizer)
        print(f"特殊トークン検出: {special_tokens}")

    def load_dataset(self, data_path: Path, apply_chat_template: bool = True, filter_long_prompts: bool = True) -> Dataset:
        """
        学習用データセットをロード

        Args:
            data_path: データセットのパス
            apply_chat_template: チャットテンプレートを適用するか
            filter_long_prompts: 長いプロンプトをフィルタリングするか

        Returns:
            Hugging Face Dataset
        """
        print(f"データセットをロード中: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # システムプロンプト
        system_prompt = "あなたは時系列データを分析し、未来の値を予測する専門家です。与えられたデータから傾向やパターンを見つけ出し、日本語で推論過程を説明してください。"

        # GRPO用のフォーマットに変換
        formatted_data = []
        for item in data:
            if apply_chat_template:
                # チャットテンプレート形式に変換
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item["instruction"]},
                ]
                # テンプレート適用（生成プロンプトとして）
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                prompt_text = item["instruction"]

            formatted_data.append({
                "prompt": prompt_text,
                "response": item["output"],
                "true_value": item["target_values"][0],  # 正解値を保持
            })

        dataset = Dataset.from_list(formatted_data)
        print(f"✓ データセットロード完了（変換前）: {len(dataset)} サンプル")

        # プロンプト長のフィルタリング
        if filter_long_prompts:
            # トークン長を計算
            def tokenize_prompt(example):
                tokens = self.tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
                return {"prompt_length": len(tokens)}

            dataset = dataset.map(tokenize_prompt)

            # 90%分位点を計算
            import numpy as np
            prompt_lengths = dataset["prompt_length"]
            max_length = int(np.quantile(prompt_lengths, 0.9))
            print(f"プロンプト長の90%分位点: {max_length} トークン")

            # フィルタリング
            original_size = len(dataset)
            dataset = dataset.filter(lambda x: x["prompt_length"] <= max_length)
            filtered_size = len(dataset)
            print(f"✓ フィルタリング完了: {original_size} → {filtered_size} サンプル（{original_size - filtered_size} サンプル除外）")

            # prompt_lengthカラムを削除
            dataset = dataset.remove_columns(["prompt_length"])

        return dataset

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        output_dir: Path = Path("models/deepseek-r1-qwen3-8b-grpo-timeseries"),
    ) -> None:
        """
        GRPOでモデルを学習

        Args:
            train_dataset: 訓練データセット
            val_dataset: 検証データセット
            output_dir: 出力ディレクトリ
        """
        training_config = self.config["training"]
        grpo_config_dict = self.config["grpo"]

        # GRPOConfig作成
        grpo_config = GRPOConfig(
            # 基本設定
            output_dir=str(output_dir),
            num_train_epochs=training_config.get("num_train_epochs", 3),
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
            max_steps=training_config.get("max_steps", None),

            # GRPO固有設定
            num_generations=grpo_config_dict.get("num_generations_per_prompt", 4),

            # 最適化設定
            learning_rate=training_config.get("learning_rate", 5.0e-6),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
            warmup_ratio=training_config.get("warmup_ratio", 0.1),
            weight_decay=training_config.get("weight_decay", 0.01),
            max_grad_norm=training_config.get("max_grad_norm", 1.0),

            # オプティマイザ
            optim=training_config.get("optim", "paged_adamw_8bit"),

            # 精度設定
            bf16=is_bfloat16_supported() and training_config.get("bf16", False),
            fp16=not is_bfloat16_supported() and training_config.get("fp16", True),

            # ロギング
            logging_steps=training_config.get("logging_steps", 10),
            save_steps=training_config.get("save_steps", 100),
            save_total_limit=training_config.get("save_total_limit", 3),

            # 評価
            eval_strategy=training_config.get("eval_strategy", "steps"),
            eval_steps=training_config.get("eval_steps", 50) if val_dataset else None,

            # その他
            seed=training_config.get("seed", 3407),
            dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
            report_to=training_config.get("report_to", ["tensorboard"]),
            logging_dir=training_config.get("logging_dir", "logs"),
        )

        # 生成設定
        generation_config = grpo_config_dict.get("generation", {})

        # GRPOTrainer作成
        print("GRPOTrainerを初期化中...")

        # 5つの報酬関数を使用（ノートブックサンプルに基づく）
        reward_functions = [
            match_format_exactly,           # </think>タグの正確性（+3.0）
            match_format_approximately,     # タグカウント評価（+0.5～-1.0）
            check_answer,                   # 答えの正確性チェック（+5.0～-4.5）
            check_numbers,                  # 数値抽出検証（+3.5または-1.5）
            format_and_language_reward_func,  # 言語検出（日本語+5.0、英語-3.0）
        ]

        trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=reward_functions,
            args=grpo_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
        )

        print("=" * 70)
        print("GRPO学習を開始します...")
        print("=" * 70)
        print(f"訓練サンプル数: {len(train_dataset)}")
        if val_dataset:
            print(f"検証サンプル数: {len(val_dataset)}")
        print(f"エポック数: {training_config.get('num_train_epochs', 3)}")
        print(f"バッチサイズ: {training_config.get('per_device_train_batch_size', 1)}")
        print(f"勾配累積: {training_config.get('gradient_accumulation_steps', 8)}")
        print(f"グループサイズ: {grpo_config_dict.get('num_generations_per_prompt', 4)}")
        print("=" * 70)

        # 学習開始
        trainer.train()

        # モデル保存
        print(f"\nモデルを保存中: {output_dir}")
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

        print("\n" + "=" * 70)
        print("✅ GRPO学習完了!")
        print("=" * 70)


def load_config(config_path: Path) -> dict[str, Any]:
    """
    YAML設定ファイルをロード

    Args:
        config_path: 設定ファイルのパス

    Returns:
        設定の辞書
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main() -> None:
    """コマンドライン実行用のメイン関数"""
    parser = argparse.ArgumentParser(
        description="DeepSeek-R1-Qwen3-8B GRPO強化学習"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="設定ファイル（YAML）のパス"
    )
    parser.add_argument(
        "--train-data", type=str, required=True, help="訓練データのパス"
    )
    parser.add_argument(
        "--val-data", type=str, default=None, help="検証データのパス"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力ディレクトリ（設定ファイルで指定されていない場合）",
    )

    args = parser.parse_args()

    # 設定ロード
    config = load_config(Path(args.config))

    # 出力ディレクトリの設定
    if args.output:
        config["training"]["output_dir"] = args.output

    output_dir = Path(config["training"]["output_dir"])

    # GPUチェック
    if not torch.cuda.is_available():
        print("⚠️  警告: CUDAが利用できません。CPU実行は非常に遅くなります。")
    else:
        print(f"✓ GPU検出: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # トレーナー初期化
    trainer = DeepSeekGRPOTrainer(config)

    # モデルロード
    trainer.load_model()

    # LoRA設定
    trainer.setup_lora()

    # データセットロード
    train_dataset = trainer.load_dataset(Path(args.train_data))
    val_dataset = trainer.load_dataset(Path(args.val_data)) if args.val_data else None

    # GRPO学習実行
    trainer.train(train_dataset, val_dataset, output_dir)


if __name__ == "__main__":
    main()
