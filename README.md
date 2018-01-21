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
```shell
main.py [-h] [--reload_extraction] [--reload_preprocessing] --dataset
               DATASET --algorithm ALGORITHM [--pkg_start PKG_START]
               --vectorizer VECTORIZER --n_clusters N_CLUSTERS --ldepth LDEPTH
               [--search] [--optimize] [--verbose]

Required arguments:
  --dataset DATASET, -d DATASET
                        The folder contraining the dataset
  --algorithm ALGORITHM, -a ALGORITHM
                        The algorithm to be used for clustering. Available options are 'km' and 'lda'
  --vectorizer VECTORIZER, -v VECTORIZER
                        The vectorizer to be used. Available options are 'tfidf' and 'count'
  --n_clusters N_CLUSTERS, -n N_CLUSTERS
                        The number of clusters
  --ldepth LDEPTH, -l LDEPTH
                        The depth of the package name hierarchy. (considered ground truth labels)

Optional arguments:
  -h, --help            show this help message and exit
  --reload_extraction   force redo the extraction of the dataset
  --reload_preprocessing
                        force redo the preprocessing on the extracted
  --pkg_start PKG_START
                        Package start to keep (useful for rejecting subpackages in a project)
  --search, -s          Enable search of tags for 'good' clusters
  --optimize, -o        Run optimizer module from 10 to N\_CLUSTERS
  --verbose, -V         Show output on screen (alternatively only saved in log file)
```
