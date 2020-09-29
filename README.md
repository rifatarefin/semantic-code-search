# semantic-code-search

Forked from the official [CodeSearchNet](https://github.com/github/CodeSearchNet) challenge repository

## Setup

1. Install docker from [official docs](https://docs.docker.com/get-started/)

2. Install [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker)

3. To download dataset, run from root directory
      ```
      script/setup
      ```
4. Start docker container by running
    ```
    script/console
    ```
5. Train a 1D-CNN model on Python data only, it will save the model in resources/saved_models directory
    ```
    python train.py --model 1dcnn /trained_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test
    ```

## Prediction

Check the [notebook](src/code_search.ipynb) to predict from example queries
