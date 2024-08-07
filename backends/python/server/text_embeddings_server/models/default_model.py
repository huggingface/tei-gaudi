import inspect
import torch

from loguru import logger
from pathlib import Path
from typing import Type, List
from transformers import AutoModel
from opentelemetry import trace

from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding

tracer = trace.get_tracer(__name__)


class DefaultModel(Model):
    def __init__(self,
                model_path: Path,
                device: torch.device,
                dtype: torch.dtype,
                trust_remote: bool=False):
        if device == torch.device("hpu"):
            adapt_transformers_to_gaudi()
        model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote).to(dtype).to(device)
        if device == torch.device("hpu"):
            logger.info("Use graph mode for HPU")
            model = wrap_in_hpu_graph(model, disable_tensor_cache=True)
        self.hidden_size = model.config.hidden_size
        position_offset = 0
        model_type = model.config.model_type
        if model_type in ["xlm-roberta", "camembert", "roberta"]:
            position_offset = model.config.pad_token_id + 1
        max_input_length = 0
        if hasattr(model.config, "max_seq_length"):
            max_input_length = model.config.max_seq_length
        else:
            max_input_length = model.config.max_position_embeddings - position_offset
        self.max_input_length = max_input_length
        self.has_position_ids = (
            inspect.signature(model.forward).parameters.get("position_ids", None)
            is not None
        )
        self.has_token_type_ids = (
            inspect.signature(model.forward).parameters.get("token_type_ids", None)
            is not None
        )

        super(DefaultModel, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        kwargs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        if self.has_token_type_ids:
            kwargs["token_type_ids"] = batch.token_type_ids
        if self.has_position_ids:
            kwargs["position_ids"] = batch.position_ids

        output = self.model(**kwargs)
        embedding = output[0][:, 0]
        cpu_results = embedding.reshape(-1).tolist()

        return [
            Embedding(
                values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]

    @tracer.start_as_current_span("predict")
    def predict(self, batch):
        raise NotImplementedError(f"Predict is not a valid operation for model type {self.model.config.model_type}")
