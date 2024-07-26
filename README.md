# Text Embeddings Inference on Habana Gaudi

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
3. You can then send a request:
   ```bash
    curl 127.0.0.1:8080/embed \
        -X POST \
        -d '{"inputs":"What is Deep Learning?"}' \
        -H 'Content-Type: application/json'
   ```

For more information and documentation about Text Embeddings Inference, checkout [the README](https://github.com/huggingface/text-embeddings-inference#text-embeddings-inference) of the original repo.

Not all features of TEI are currently supported as this is still a work in progress.

## Validated Models

| Architecture | Model Type | Models |
|--------------|------------|--------|
| BERT | Embedding | <li>[BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)</li><li>[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)</li><li>[sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)</li><li>[sentence-transformers/multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)</li><li>[sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)</li><li>[sentence-transformers/paraphrase-MiniLM-L3-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2)</li> |
| MPNet | Embedding | <li>[sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)</li><li>[sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)</li><li>[sentence-transformers/multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1)</li> |
| ALBERT | Embedding | <li>[sentence-transformers/paraphrase-albert-small-v2](https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2)</li> |
| Mistral | Embedding | <li>[intfloat/e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct)</li><li>[Salesforce/SFR-Embedding-2_R](https://huggingface.co/Salesforce/SFR-Embedding-2_R)</li> |
| GTE | Embedding | <li>[Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)</li> |
| JinaBERT | Embedding | <li>[jinaai/jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)</li> |

> The license to use TEI on Habana Gaudi is the one of TEI: https://github.com/huggingface/text-embeddings-inference/blob/main/LICENSE
>
> Please reach out to api-enterprise@huggingface.co if you have any question.
