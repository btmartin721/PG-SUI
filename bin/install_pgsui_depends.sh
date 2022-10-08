#!/bin/bash

if [ -z "$1" ]; then
	echo "
	No conda environment supplied!
	Usage: <script.sh> <conda_env_name_to_create>
	WARNING: This will overwrite an existing conda environment!!!
	";
	exit 1;
fi

current_env=$1

conda create -n $current_env python=3.8
source activate $current_env

conda install -y matplotlib seaborn jupyterlab scikit-learn=1.0 tqdm pandas=1.2.5 numpy=1.20.2 scipy=1.6.2 xgboost lightgbm

conda install -y scikit-learn-intelex

conda install -y -c conda-forge toytree python-kaleido

conda install -y -c plotly plotly

pip install sklearn-genetic-opt[all]

pip install scikeras

pip install tensorflow-cpu==2.10

