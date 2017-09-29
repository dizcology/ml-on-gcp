# Parallelized Hyperparameter Search with Google Container Engine

## Introduction

This sample package helps you run `scikit-learn`'s `GridSearchCV` and `RandomSearchCV` on [Google Container Engine](https://cloud.google.com/container-engine/).


## Requirements

You will need a Google Cloud Platform project which has the following products enabled:

- [Google Container Registry](https://cloud.google.com/container-registry/)

- [Google Container Engine](https://cloud.google.com/container-engine/)

- [Google Cloud Storage](https://cloud.google.com/storage/)


In addition, to follow the steps of the sample we recommend you work in a [Jupyter notebook](https://jupyter.org/) running [Python](https://www.python.org/) v2.7.10 or newer.


## Steps

1. Install [Google Cloud Platform SDK](https://cloud.google.com/sdk/downloads).

1. Install [kubectl](https://cloud.google.com/container-engine/docs/quickstart).

1. `git clone https://github.com/GoogleCloudPlatform/ml-on-gcp.git`

1. `cd ml-on-gcp`

1. `pip install -r requirements.txt`

1. Follow the steps in `gke_search.ipynb`.