# Text Embeddings Inference on Habana Gaudi

To use [🤗 text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) on Habana Gaudi/Gaudi2, follow these steps:

1. Build the Docker image located in this folder with:
   ```bash
   docker build -f Dockerfile-hpu -t tei_gaudi .
   ```
2. Launch a local server instance on 1 Gaudi card:
   ```bash
   model=BAAI/bge-large-en-v1.5
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host tei_gaudi --model-id $model --pooling cls
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

> The license to use TEI on Habana Gaudi is the one of TEI: https://github.com/huggingface/text-embeddings-inference/blob/main/LICENSE
>
> Please reach out to api-enterprise@huggingface.co if you have any question.
