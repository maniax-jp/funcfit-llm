"""
DeepSeek-R1 Unslothファインチューニングスクリプト

Unslothを使用してDeepSeek-R1モデルを時系列予測タスクでファインチューニングします。
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer


class DeepSeekFineTuner:
    """DeepSeek-R1モデルのファインチューニングクラス"""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Args:
            config: ファインチューニング設定の辞書
        """
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Unslothを使用してモデルとトークナイザーをロード"""
        model_config = self.config["model"]

        print(f"モデルをロード中: {model_config['name']}")

        # Unslothの高速モデルロード
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config["name"],
            max_seq_length=model_config.get("max_seq_length", 2048),
            dtype=None,  # 自動選択
            load_in_4bit=model_config.get("load_in_4bit", True),
            device_map="auto",
        )

        print("モデルロード完了")

    def setup_lora(self) -> None:
        """LoRA設定を適用"""
        lora_config = self.config["lora"]

        print("LoRA設定を適用中...")

        # Unslothのget_peft_modelを使用
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_config.get("r", 16),
            target_modules=lora_config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            lora_alpha=lora_config.get("alpha", 16),
            lora_dropout=lora_config.get("dropout", 0.0),
            bias=lora_config.get("bias", "none"),
            use_gradient_checkpointing=lora_config.get("gradient_checkpointing", True),
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        print("LoRA設定完了")

    def load_dataset(self, data_path: Path) -> Dataset:
        """
        学習用データセットをロード

        Args:
            data_path: データセットのパス

        Returns:
            Hugging Face Dataset
        """
        print(f"データセットをロード中: {data_path}")

        if data_path.suffix == ".json":
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)
        else:
            # Hugging Face形式
            dataset = load_dataset(str(data_path), split="train")

        print(f"データセットロード完了: {len(dataset)} サンプル")
        return dataset

    def format_prompt(self, sample: dict[str, Any]) -> str:
        """
        DeepSeek-R1用のプロンプトフォーマット

        Args:
            sample: データサンプル

        Returns:
            フォーマットされたプロンプト
        """
        # DeepSeek-R1のチャット形式
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{sample['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{sample['output']}<|eot_id|>"""

        return prompt

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        output_dir: Path = Path("models/deepseek-r1-timeseries"),
    ) -> None:
        """
        モデルをファインチューニング

        Args:
            train_dataset: 訓練データセット
            val_dataset: 検証データセット
            output_dir: 出力ディレクトリ
        """
        training_config = self.config["training"]

        # TrainingArguments設定
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=training_config.get("num_epochs", 3),
            per_device_train_batch_size=training_config.get("batch_size", 4),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            learning_rate=training_config.get("learning_rate", 2e-4),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=training_config.get("logging_steps", 10),
            save_steps=training_config.get("save_steps", 100),
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=training_config.get("eval_steps", 100) if val_dataset else None,
            save_total_limit=training_config.get("save_total_limit", 3),
            warmup_steps=training_config.get("warmup_steps", 10),
            weight_decay=training_config.get("weight_decay", 0.01),
            optim=training_config.get("optimizer", "adamw_8bit"),
            seed=3407,
            report_to=training_config.get("report_to", ["tensorboard"]),
        )

        # SFTTrainerでファインチューニング
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=self.config["model"].get("max_seq_length", 2048),
            args=training_args,
            packing=False,
            formatting_func=lambda x: self.format_prompt(x),
        )

        # 学習開始
        print("ファインチューニング開始...")
        trainer.train()

        # モデル保存
        print(f"モデルを保存中: {output_dir}")
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

        print("ファインチューニング完了!")


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
    parser = argparse.ArgumentParser(description="DeepSeek-R1 ファインチューニング")
    parser.add_argument(
        "--config", type=str, required=True, help="設定ファイル（YAML）のパス"
    )
    parser.add_argument("--train-data", type=str, required=True, help="訓練データのパス")
    parser.add_argument("--val-data", type=str, default=None, help="検証データのパス")
    parser.add_argument(
        "--output",
        type=str,
        default="models/deepseek-r1-timeseries",
        help="出力ディレクトリ",
    )

    args = parser.parse_args()

    # 設定ロード
    config = load_config(Path(args.config))

    # ファインチューナー初期化
    finetuner = DeepSeekFineTuner(config)

    # モデルロード
    finetuner.load_model()

    # LoRA設定
    finetuner.setup_lora()

    # データセットロード
    train_dataset = finetuner.load_dataset(Path(args.train_data))
    val_dataset = finetuner.load_dataset(Path(args.val_data)) if args.val_data else None

    # ファインチューニング実行
    finetuner.train(train_dataset, val_dataset, Path(args.output))


if __name__ == "__main__":
    main()
