# semantic-code-search

Forked from the official [CodeSearchNet](https://github.com/github/CodeSearchNet) challenge repository

## Steps to run with Singularity

1. If running on TACC clusters, load the Singularity module
      ```
      module load tacc-singularity
      ``` 
2. To download dataset, run from root directory
      ```
      script/setup
      ```
3. Start the container by running
    ```
    script/console
    ```
4. Train a 1D-CNN model on Python data only, it will save the model in resources/saved_models directory
    ```
    python train.py --model 1dcnn /trained_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test
    ```

## Prediction

Check the [notebook](src/code_search.ipynb) to predict from example queries
