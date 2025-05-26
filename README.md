# Text Embeddings Inference on Habana Gaudi

> [!WARNING]
> This repository is deprecated. Please use the [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) repository instead for the latest version of TEI on Intel Gaudi. You should now use the latest image hosted on the TEI repo `ghcr.io/huggingface/text-embeddings-inference:hpu-latest` instead of `ghcr.io/huggingface/tei-gaudi:latest`. You can find the new Gaudi images in the [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference/pkgs/container/text-embeddings-inference) registry. You can also check the [Gaudi Backend documentation](https://huggingface.co/docs/text-embeddings-inference/intel_container#hpu) for more information.


## Table of contents
- [Get started](#get-started)
- [Supported Models](#supported-models)
  - [Text Embeddings](#text-embeddings)
  - [Sequence Classification and Re-Ranking](#sequence-classification-and-re-ranking)
- [How to Use](#how-to-use)
  - [Using Re-rankers models](#using-re-rankers-models)
  - [Using Sequence Classification models](#using-sequence-classification-models)
  - [Using SPLADE pooling](#using-splade-pooling)

## Get started
To use [🤗 text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) on Habana Gaudi/Gaudi2, follow these steps:

1. Pull the official Docker image with:
   ```bash
   docker pull ghcr.io/huggingface/tei-gaudi:latest
   ```
> [!NOTE]
> Alternatively, you can build the Docker image using `Dockerfile-hpu` located in this folder with:
> ```bash
> docker build -f Dockerfile-hpu -t tei_gaudi .
> ```
2. Launch a local server instance on 1 Gaudi card:
   ```bash
   model=BAAI/bge-large-en-v1.5
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e MAX_WARMUP_SEQUENCE_LENGTH=512 --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tei-gaudi:latest --model-id $model --pooling cls
   ```
   For models within the Transformers library that need remote code to run customized implementations, please set the environment variable `-e TRUST_REMOTE_CODE=TRUE` within `docker run` command line. Here is an example:
   ```
   model="Alibaba-NLP/gte-large-en-v1.5"
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e MAX_WARMUP_SEQUENCE_LENGTH=512 -e TRUST_REMOTE_CODE=TRUE --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tei-gaudi:latest --model-id $model --pooling cls
3. You can then send a request:
   ```bash
    curl 127.0.0.1:8080/embed \
        -X POST \
        -d '{"inputs":"What is Deep Learning?"}' \
        -H 'Content-Type: application/json'
   ```

For more information and documentation about Text Embeddings Inference, checkout [README](https://github.com/huggingface/text-embeddings-inference#text-embeddings-inference) of the original repo.


## Supported Models
### Text Embeddings
`tei-gaudi` currently supports Nomic, BERT, CamemBERT, XLM-RoBERTa models with absolute positions, JinaBERT model with Alibi positions and Mistral, Alibaba GTE and Qwen2 models with Rope positions.

Below are some examples of our validated models:

| Architecture | Pooling | Models |
|--------------|---------|--------|
| BERT | Cls/Mean/Last token | <li>[BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)</li><li>[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)</li><li>[sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)</li><li>[sentence-transformers/multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)</li><li>[sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)</li><li>[sentence-transformers/paraphrase-MiniLM-L3-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2)</li> |
| BERT | Splade | <li>[naver/efficient-splade-VI-BT-large-query](https://huggingface.co/naver/efficient-splade-VI-BT-large-query)</li> |
| MPNet | Cls/Mean/Last token | <li>[sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)</li><li>[sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)</li><li>[sentence-transformers/multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1)</li> |
| ALBERT | Cls/Mean/Last token | <li>[sentence-transformers/paraphrase-albert-small-v2](https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2)</li> |
| Mistral | Cls/Mean/Last token | <li>[intfloat/e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct)</li><li>[Salesforce/SFR-Embedding-2_R](https://huggingface.co/Salesforce/SFR-Embedding-2_R)</li> |
| GTE | Cls/Mean/Last token | <li>[Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)</li> |
| JinaBERT | Cls/Mean/Last token | <li>[jinaai/jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)</li> |

### Sequence Classification and Re-Ranking
`tei-gaudi` currently supports CamemBERT, and XLM-RoBERTa Sequence Classification models with absolute positions.

Below are some examples of the currently supported models:
| Task               | Model Type  | Model ID                                                                                    |
|--------------------|-------------|---------------------------------------------------------------------------------------------|
| Re-Ranking         | XLM-RoBERTa | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)                   |
| Re-Ranking         | XLM-RoBERTa | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)                     |
| Sentiment Analysis | RoBERTa     | [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions) |

## How to Use
### Using Re-rankers models

```shell
model=BAAI/bge-reranker-large
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e MAX_WARMUP_SEQUENCE_LENGTH=512 --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tei-gaudi:latest --model-id $model
```

And then you can rank the similarity between a query and a list of texts with:

```bash
curl 127.0.0.1:8080/rerank \
    -X POST \
    -d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}' \
    -H 'Content-Type: application/json'
```

### Using Sequence Classification models

You can also use classic Sequence Classification models like `SamLowe/roberta-base-go_emotions`:

```shell
model=SamLowe/roberta-base-go_emotions
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e MAX_WARMUP_SEQUENCE_LENGTH=512 --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tei-gaudi:latest --model-id $model
```

Once you have deployed the model you can use the `predict` endpoint to get the emotions most associated with an input:

```bash
curl 127.0.0.1:8080/predict \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```

### Using SPLADE pooling

You can choose to activate SPLADE pooling for Bert and Distilbert MaskedLM architectures:

```shell
docker build -f Dockerfile-hpu -t tei_gaudi .
model=naver/efficient-splade-VI-BT-large-query
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e MAX_WARMUP_SEQUENCE_LENGTH=512 --cap-add=sys_nice --ipc=host tei_gaudi --model-id $model --pooling splade
```

Once you have deployed the model you can use the `/embed_sparse` endpoint to get the sparse embedding:

```bash
curl 127.0.0.1:8080/embed_sparse \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```
> The license to use TEI on Habana Gaudi is the one of TEI: https://github.com/huggingface/text-embeddings-inference/blob/main/LICENSE
>
> Please reach out to api-enterprise@huggingface.co if you have any question.
