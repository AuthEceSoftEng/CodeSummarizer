# CodeSummarizer
Summarizing Software Functionality from Source Code

## Setup Environment:

1. Run

   ```sh
   git clone git@github.com:AuthEceSoftEng/CodeSummarizer.git ${HOME}/code_summarizer && cd ${HOME}/code_summarizer/src && make setup
   ```

   to setup the python virtual environment and install all dependencies

2. Run

   ```bash
   source ${HOME}/.code_summarizer_venv/bin/activate
   ```

   to get into the python virtual environment

## Usage:
```
main.py [-h] [--reload_extraction] [--reload_preprocessing]
             [--algorithm ALGORITHM] [--pkg_start PKG_START]
             [--vectorizer VECTORIZER] [--n_clusters N_CLUSTERS]
             [--ldepth LDEPTH] [--search] [--optimize] [--verbose]
             DATASET

positional arguments:
  DATASET               The folder contraining the dataset

optional arguments:
  -h, --help            show this help message and exit
  --reload_extraction   force to redo the extraction of the dataset (do not use previous run)
  --reload_preprocessing
                        force to redo the preprocessing on the extracted (do not use previous run)
  --algorithm ALGORITHM, -a ALGORITHM
                        The algorithm to be used for clustering. Available options are 'km' and 'lda'. Default 'lda'.
  --pkg_start PKG_START
                        Package start to keep (useful for excluding certain subpackages of a project)
  --vectorizer VECTORIZER, -v VECTORIZER
                        The vectorizer to be used. Available options are 'tfidf' and 'count'. Default 'count'.
  --n_clusters N_CLUSTERS, -n N_CLUSTERS
                        The number of clusters. Default 200.
  --ldepth LDEPTH, -l LDEPTH
                        The depth of the package structure to be used as ground truth labels. Default 1.
  --search, -s          Enable search of tags for 'good' clusters
  --optimize, -o        Run optimizer module from 10 to N_CLUSTERS
  --verbose, -V         Show output on screen (alt only saved in log file)

```

##  Example

```shell
python main.py /path/to/dataset -a lda -v tfidf -l 2 -n 100 -V -o
```

## Custom search API key

This software uses the Google JSON Custom Search API. To enable the search feature using the `-s` flag, an API key has to be obtained and placed in the `config/key.txt` file. 	More details on how to obtain the API key, can be found [here](https://developers.google.com/custom-search/json-api/v1/overview). 

