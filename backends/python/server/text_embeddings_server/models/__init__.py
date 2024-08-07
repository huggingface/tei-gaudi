import os
import torch

from loguru import logger
from pathlib import Path
from typing import Optional
from transformers import AutoConfig
from transformers.models.bert import BertConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES

from text_embeddings_server.models.model import Model, B
from text_embeddings_server.models.default_model import DefaultModel
from text_embeddings_server.models.classification_model import ClassificationModel

__all__ = ["Model"]

HTCORE_AVAILABLE = True
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() in ["true", "1"]

try:
    import habana_frameworks.torch.core as htcore
except ImportError as e:
    logger.warning(f"Could not import htcore: {e}")
    HTCORE_AVAILABLE = False

# Disable gradients
torch.set_grad_enabled(False)

FLASH_ATTENTION = True
try:
    from text_embeddings_server.models.flash_bert import FlashBert
except ImportError as e:
    logger.warning(f"Could not import Flash Attention enabled models: {e}")
    FLASH_ATTENTION = False

if FLASH_ATTENTION:
    __all__.append(FlashBert)


def get_model(model_path: Path, dtype: Optional[str]):
    if dtype == "float32":
        dtype = torch.float32
    elif dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise RuntimeError(f"Unknown dtype {dtype}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif HTCORE_AVAILABLE and torch.hpu.is_available():
        device = torch.device("hpu")
    else:
        if dtype != torch.float32:
            raise ValueError("CPU device only supports float32 dtype")
        device = torch.device("cpu")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=TRUST_REMOTE_CODE)

    if config.model_type == "bert":
        config: BertConfig
        if (
            device.type == "cuda"
            and config.position_embedding_type == "absolute"
            and dtype in [torch.float16, torch.bfloat16]
            and FLASH_ATTENTION
        ):
            return FlashBert(model_path, device, dtype)
        else:
            if config.architectures[0] in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values():
                return ClassificationModel(model_path, device, dtype)
            else:
                return DefaultModel(model_path, device, dtype, trust_remote=TRUST_REMOTE_CODE)
    else:
        try:
            if config.architectures[0] in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values():
                return ClassificationModel(model_path, device, dtype)
            else:
                return DefaultModel(model_path, device, dtype, trust_remote=TRUST_REMOTE_CODE)
        except:
            raise RuntimeError(f"Unsupported model_type {config.model_type}")
