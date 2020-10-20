# semantic-code-search

Forked from the official [CodeSearchNet](https://github.com/github/CodeSearchNet) challenge repository

## Search for code (command line tool on TACC)

1. Steps 1-4 from the "Train a Model" section bellow.

2. Run `search.py`

        python src/search.py

## Train a Model
### Steps to run with Singularity on TACC

1. If running on TACC clusters, load the Singularity module
      
      module load tacc-singularity
      
2. Clone dir in the work directory of TACC
        
        cd $WORK
        cd semantic-code-search/
        
3. To download dataset (only the first time), run from root directory
      
        script/setup
      
4. Start the container by running
    
        script/console
    
5. Train

        cd src/
        
    * an 1D-CNN model on Python data only, it will save the model in resources/saved_models directory
    
            python train.py --model 1dcnn /trained_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test
    
    * a NBOW model on all programming languages
    
            python train.py --model neuralbow  ../resources/saved_models     

### Prediction

Check the [notebook](src/code_search.ipynb) to predict from example queries


