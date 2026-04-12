"""
Fine-tune a quantized BQQ language model.

Modes:
  - SFT only (default): standard next-token prediction loss
  - SFT + KL distillation (--teacher_model_name): adds KL divergence loss
    against a teacher model's logits

Usage:
  # SFT only
  python fine_tuning.py --model_name Qwen/Qwen2.5-1.5B --model_path model.pth

  # SFT + KL distillation
  python fine_tuning.py --model_name Qwen/Qwen2.5-1.5B --model_path model.pth \
    --teacher_model_name Qwen/Qwen2.5-1.5B --kl_alpha 1.0 --kl_temperature 2.0
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers.data.data_collator import _torch_collate_batch, pad_without_fast_tokenizer_warning
from trl import SFTConfig, SFTTrainer

try:
    from .compressed_data import default_quantized_model_dir, model_basename
except ImportError:
    from compressed_data import default_quantized_model_dir, model_basename


# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Distillation Trainer (SFT + KL)
# ---------------------------------------------------------------------------

class DistillationTrainer(SFTTrainer):
    """SFTTrainer with optional KL distillation loss against a teacher model."""

    def __init__(self, teacher_model: Optional[nn.Module] = None,
                 ce_alpha: float = 1.0, kl_alpha: float = 1.0,
                 kl_temperature: float = 2.0, **kwargs):
        """
        Args:
            ce_alpha: Weight for CE loss. Set to 0 for KL-only distillation.
            kl_alpha: Weight for KL distillation loss.
            kl_temperature: Temperature for softmax in KL divergence.
        """
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.ce_alpha = ce_alpha
        self.kl_alpha = kl_alpha
        self.kl_temperature = kl_temperature

        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)

        if self.teacher_model is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # KL distillation loss
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )

        T = self.kl_temperature
        student_logp = F.log_softmax(outputs.logits / T, dim=-1)
        teacher_p = F.softmax(teacher_outputs.logits / T, dim=-1)
        kl_loss = F.kl_div(student_logp, teacher_p, reduction="batchmean") * (T * T)

        loss = self.ce_alpha * outputs.loss + self.kl_alpha * kl_loss
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model_path: str,
    model_name: str,
    train_dataset,
    test_dataset,
    output_dir: str = "./output_sft",
    num_train_epochs: int = 1,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 512,
    teacher_model_name: Optional[str] = None,
    ce_alpha: float = 1.0,
    kl_alpha: float = 1.0,
    kl_temperature: float = 2.0,
):
    model = torch.load(model_path, weights_only=False, map_location="cpu")

    training_args = SFTConfig(
        output_dir=output_dir,
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
        report_to="none",
        bf16=False,
        fp16=False,
        max_seq_length=max_seq_length,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Teacher model (optional)
    teacher_model = None
    if teacher_model_name is not None:
        print(f"Loading teacher model: {teacher_model_name}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name, dtype="auto"
        )

    TrainerClass = DistillationTrainer if teacher_model is not None else SFTTrainer

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=SafeDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    if teacher_model is not None:
        trainer_kwargs.update(
            teacher_model=teacher_model,
            ce_alpha=ce_alpha,
            kl_alpha=kl_alpha,
            kl_temperature=kl_temperature,
        )

    trainer = TrainerClass(**trainer_kwargs)

    print("Starting training...")
    trainer.train()

    # Save — reload original structure and apply trained state_dict
    trainer.save_model(output_dir)
    state_dict = trainer.model.state_dict()

    _model = torch.load(model_path, weights_only=False, map_location="cpu")
    _model.load_state_dict(state_dict)

    # Derive output name from input: {stem}-finetuned.pth or {stem}-distilled.pth
    input_stem = Path(model_path).stem
    if teacher_model_name is not None and ce_alpha == 0:
        suffix = "distilled"
    elif teacher_model_name is not None:
        suffix = "finetuned"  # CE+KL, still call it finetuned
    else:
        suffix = "finetuned"
    output_path = Path(output_dir) / f"{input_stem}-{suffix}.pth"
    torch.save(_model, output_path)

    print(f"Training complete! Saved model to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a quantized BQQ language model")
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
    # Distillation
    parser.add_argument("--teacher_model_name", type=str, default=None,
                        help="HuggingFace model name for KL distillation teacher. "
                             "If omitted, only CE loss is used (standard SFT).")
    parser.add_argument("--ce_alpha", type=float, default=1.0,
                        help="Weight for CE loss. Set to 0 for KL-only distillation.")
    parser.add_argument("--kl_alpha", type=float, default=1.0,
                        help="Weight for KL distillation loss")
    parser.add_argument("--kl_temperature", type=float, default=2.0,
                        help="Temperature for softmax in KL distillation")

    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
        model_id = model_basename(args.model_name)
        model_path = default_quantized_model_dir(args.model_name) / \
            f"{model_id}-{args.bit_width}bit-{args.group_size}gs.pth"

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
        teacher_model_name=args.teacher_model_name,
        ce_alpha=args.ce_alpha,
        kl_alpha=args.kl_alpha,
        kl_temperature=args.kl_temperature,
    )


if __name__ == "__main__":
    main()
