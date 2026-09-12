"""Custom vLLM logits processors used by the generation demos."""

from __future__ import annotations

import os

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """Prevent an output n-gram from being generated more than once."""

    def __init__(self, _vllm_config, device: torch.device, _is_pin_memory: bool):
        self.ngram_size = int(os.environ["BQQ_NO_REPEAT_NGRAM_SIZE"])
        self.device = device
        self.output_token_ids: dict[int, list[int]] = {}

    @classmethod
    def validate_params(cls, _sampling_params: SamplingParams) -> None:
        size = int(os.environ.get("BQQ_NO_REPEAT_NGRAM_SIZE", "0"))
        if size < 2:
            raise ValueError("BQQ_NO_REPEAT_NGRAM_SIZE must be at least 2")

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return

        for index in batch_update.removed:
            self.output_token_ids.pop(index, None)
        for index, _params, _prompt_ids, output_ids in batch_update.added:
            self.output_token_ids[index] = output_ids
        for source, destination, directionality in batch_update.moved:
            source_ids = self.output_token_ids.get(source)
            destination_ids = self.output_token_ids.get(destination)
            if source_ids is not None:
                self.output_token_ids[destination] = source_ids
            else:
                self.output_token_ids.pop(destination, None)
            if directionality == MoveDirectionality.SWAP:
                if destination_ids is not None:
                    self.output_token_ids[source] = destination_ids
                else:
                    self.output_token_ids.pop(source, None)
            else:
                self.output_token_ids.pop(source, None)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        n = self.ngram_size
        for row, token_ids in self.output_token_ids.items():
            if len(token_ids) < n - 1:
                continue
            prefix = token_ids[-(n - 1) :]
            banned = {
                token_ids[start + n - 1]
                for start in range(len(token_ids) - n + 1)
                if token_ids[start : start + n - 1] == prefix
            }
            if banned:
                banned_ids = torch.tensor(tuple(banned), device=self.device)
                logits[row, banned_ids] = -torch.inf
        return logits
