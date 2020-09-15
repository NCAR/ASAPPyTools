#!/bin/bash

set -e
set -eo pipefail

if [[ -z "${ENV}" ]]; then
  ENV=default
fi

conda config --set always_yes true --set changeps1 false --set quiet true
conda update -q conda
conda config --set pip_interop_enabled True # Enable pip interoperability
conda config --add channels conda-forge
conda env create -f ci/env-${ENV}.yml --name=asaptools-${ENV} --quiet
conda env list
source activate asaptools-${ENV}
pip install pip --upgrade
pip install --no-deps --quiet -e .
conda list -n asaptools-${ENV}
