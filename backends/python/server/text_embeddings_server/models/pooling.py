from abc import ABC, abstractmethod

import torch
from loguru import logger
from opentelemetry import trace
from sentence_transformers.models import Pooling
from torch import Tensor

tracer = trace.get_tracer(__name__)


class _Pooling(ABC):
    @abstractmethod
    def forward(self, model_output, attention_mask) -> Tensor:
        pass


class DefaultPooling(_Pooling):
    def __init__(self, hidden_size, pooling_mode) -> None:
        assert (
            pooling_mode != "splade"
        ), "Splade pooling is not supported for DefaultPooling"
        self.pooling = Pooling(hidden_size, pooling_mode=pooling_mode)

    @tracer.start_as_current_span("pooling")
    def forward(self, model_output, attention_mask) -> Tensor:
        pooling_features = {
            "token_embeddings": model_output[0],
            "attention_mask": attention_mask,
        }
        return self.pooling.forward(pooling_features)["sentence_embedding"]


class SpladePooling(_Pooling):
    @tracer.start_as_current_span("pooling")
    def forward(self, model_output, attention_mask) -> Tensor:
        # Implement Splade pooling
        logger.info(f"Attention Mask ({attention_mask.shape}): {attention_mask}")
        logger.info(f"Model output: {model_output}")
        hidden_states = model_output[0].contiguous()
        hidden_states = (1 + hidden_states).log()
        logger.info(f"Hidden state shape: {hidden_states.shape}")
        attention_mask = attention_mask[:, :, None]
        logger.info(f"Attention Mask new shape: {attention_mask.shape}")
        hidden_states = torch.mul(hidden_states, attention_mask)
        logger.info(f"Hidden state after attention_mask: {hidden_states}")
        hidden_states = hidden_states.max(dim=1).values
        logger.info(f"Hidden states after log max: {hidden_states}")
        return hidden_states
