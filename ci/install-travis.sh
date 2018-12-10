#!/bin/bash

set -e
echo "[install-travis]"

MINICONDA_DIR="$HOME/miniconda3"
time wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh || exit 1
time bash miniconda.sh -b -p "$MINICONDA_DIR" || exit 1

echo
echo "[show conda]"
which conda

echo
echo "[update conda]"
conda config --set always_yes true --set changeps1 false || exit 1

echo
echo "[install dependencies]"
conda create -q -n asaptools-dev python=${PYTHON}
conda env update -f environment-dev.yml
source activate asaptools-dev
conda list

echo
echo "[install asaptools]"
pip install --no-deps -e .

echo "[finished install]"

exit 0