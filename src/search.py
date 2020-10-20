import pickle
import re
import shutil
import sys

from annoy import AnnoyIndex
from docopt import docopt
from dpu_utils.utils import RichPath
import pandas as pd
from tqdm import tqdm
import wandb
from wandb.apis import InternalApi

from dataextraction.python.parse_python_data import tokenize_docstring_from_string
import model_restore_helper

"""
Creates a the query embedding and initates a nearest neighbor clsuter on the indexing database
"""
def query_model(query, model, indices, language, topk=100):
    query_embedding = model.get_query_representations([{'docstring_tokens': tokenize_docstring_from_string(query),
                                                        'language': language}])[0]
    idxs, distances = indices.get_nns_by_vector(query_embedding, topk, include_distances=True)
    return idxs, distances

"""
Wandb
"""
#args_wandb_run_id = 'ligerfotis/semantic-code-search-src/2o0bi8kg'
args_wandb_run_id = 'rifatarefin/CodeSearchNet/6nvzrp2p'

# validate format of runid:
if len(args_wandb_run_id.split('/')) != 3:
    print("ERROR: Invalid wandb_run_id format: %s (Expecting: user/project/hash)" % args_wandb_run_id, file=sys.stderr)
    sys.exit(1)
wandb_api = wandb.Api()

# retrieve saved model from W&B for this run
print("Fetching run from W&B...")
try:
    run = wandb_api.run(args_wandb_run_id)
except wandb.CommError as e:
    print("ERROR: Problem querying W&B for wandb_run_id: %s" % args_wandb_run_id, file=sys.stderr)
    sys.exit(1)

print("Fetching run files from W&B...")
gz_run_files = [f for f in run.files() if f.name.endswith('gz')]
if not gz_run_files:
    print("ERROR: Run contains no model-like files")
    sys.exit(1)
model_file = gz_run_files[0].download(replace=True)
local_model_path = model_file.name
run_id = args_wandb_run_id.split('/')[-1]

model_path = RichPath.create(local_model_path, None)
print("Restoring model from %s" % model_path)
model = model_restore_helper.restore(
    path=model_path,
    is_train=False,
    hyper_overrides={})

"""
Loads the code embeddings in the Annoy database
"""
predictions = []
for language in ('python', 'go', 'javascript', 'java', 'php', 'ruby'):
    definitions = pickle.load(open('../resources/data/{}_dedupe_definitions_v2.pkl'.format(language), 'rb'))
    indexes = [{'code_tokens': d['function_tokens'], 'language': d['language']} for d in tqdm(definitions)]
    print(len(indexes))
    code_representations = model.get_code_representations(indexes[:int(len(indexes)/20)])
    # print(code_representations)
    print(len(indexes))

    indices = AnnoyIndex(code_representations[0].shape[0], 'angular')
    for index, vector in tqdm(enumerate(code_representations)):
        if vector is not None:
            indices.add_item(index, vector)
    indices.build(200)


supported_pl = ['python', 'go', 'javascript', 'java', 'php', 'ruby']

"""
Performs the code search for the given language and the query
"""
def search(language, query):
    print("Language detected: " + language)
    print("Query received: " + query)
    for idx, _ in zip(*query_model(query, model, indices, language)):
        predictions.append((query, language, definitions[idx]['identifier'], definitions[idx]['url']))

    df = pd.DataFrame(predictions, columns=['query', 'language', 'identifier', 'url'])
    print(df.loc[df['language'] == language])
    is_language_of_interest =  df['language']==language
    # urls = list(df.loc[df['language'] == language]['url'])
    urls = df[is_language_of_interest]['url'][:10]
    # print(df.head(10))
    print("Top ten results")
    for link in urls:
        print(link)



if __name__ == '__main__':
    print("Welcome to the Semantics Code Search App!")
    # Search menu
    while True:
        pl = input("Type your language:")
        while pl not in supported_pl:
            print(
                "Select one of the programming languages listed: ['python', 'go', 'javascript', 'java', 'php', 'ruby']")
            pl = input("Type your language:")

        query = input("Query:")

        search(pl, query)


