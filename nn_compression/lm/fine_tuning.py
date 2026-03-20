# Copyright 2024 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0

import torch
from datasets import load_dataset
from pathlib import Path
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers.data.data_collator import _torch_collate_batch, pad_without_fast_tokenizer_warning
from trl import SFTConfig, SFTTrainer
from typing import Any, Dict, List, Mapping
import os
import argparse

try:
    from .compressed_data import default_quantized_model_dir, model_basename
except ImportError:
    from compressed_data import default_quantized_model_dir, model_basename




##########################################
# 1. DataCollator（安全版）
##########################################
class SafeDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[List[int] | Any | Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        for key in ['input_ids', 'attention_mask', 'labels']:
            if key in batch and hasattr(batch[key], 'dtype'):
                batch[key] = batch[key].to(torch.long)

        return batch


def train(model_path, model_name, train_dataset, test_dataset, output_dir="./output_qwen3_sft", num_train_epochs=1, learning_rate=2e-5, gradient_accumulation_steps=4, max_seq_length=512, ):

    model = torch.load(model_path, weights_only=False, map_location='cpu')

    ##########################################
    # 2. SFTConfigをPython内で直接定義
    ##########################################
    training_args = SFTConfig(
        output_dir=output_dir,  # モデルの保存先
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        report_to="none",   # WandBなど使わない場合
        bf16=False,         # bf16環境であればTrueでも可
        fp16=False,         # fp16環境であればTrueでも可
        max_seq_length=max_seq_length, # sequence長
    )

    ##########################################
    # 3. モデルとトークナイザの読み込み
    ##########################################
    # ★パスを自分の環境に合わせて変更してください

    print(f"Loading quantized model from: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token





    ##########################################
    # 4. トレーナーの設定
    ##########################################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=SafeDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    ##########################################
    # 5. 学習の実行
    ##########################################
    print("Starting training...")
    trainer.train()

    ##########################################
    # 7. 保存処理（量子化済み構造に戻して保存）
    ##########################################
    trainer.save_model(output_dir)
    state_dict = trainer.model.state_dict()

    # 元の量子化済み構造に適用して再保存
    _model = torch.load(model_path, weights_only=False, map_location='cpu')
    _model.load_state_dict(state_dict)
    torch.save(_model, f"{output_dir}/trained_model.pth")

    print(f"Training complete! Saved model to {output_dir}/trained_model.pth")




##########################################
# 1. データセットの準備
##########################################


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a quantized language model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--bit_width", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
        model_id = model_basename(args.model_name)
        model_path = default_quantized_model_dir(args.model_name) / f"{model_id}-{args.bit_width}bit-{args.group_size}gs-{args.num_steps}step.pth"

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "fine_tuned_models" / model_basename(args.model_name)


    def is_not_empty(example):
        text = example['text'].strip()
        return len(text) > 10

    train_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train').filter(is_not_empty)
    test_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test').select(range(256))

    train(
        model_path=str(model_path),
        model_name=args.model_name,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
