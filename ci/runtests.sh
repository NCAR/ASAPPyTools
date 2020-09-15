#!/bin/bash

set -e
set -eo pipefail

if [[ -z "${ENV}" ]]; then
  ENV=default
fi

if [[ -z "${COV}" ]]; then
  CMD="python -m unittest"
else
  CMD="coverage run -p -m"
fi

source activate asaptools-${ENV}
${CMD} asaptools.tests.vprinterTests
${CMD} asaptools.tests.timekeeperTests
${CMD} asaptools.tests.partitionTests
${CMD} asaptools.tests.partitionArrayTests
${CMD} asaptools.tests.simpleCommP1STests
mpirun -np 4 ${CMD} asaptools.tests.simpleCommParTests
mpirun -np 4 ${CMD} asaptools.tests.simpleCommParDivTests

if [[ ! -z ${COV} ]]; then
  coverage combine
  apt-get update
  apt-get --yes install curl
  source activate asaptools-${ENV}
  curl -s https://codecov.io/bash > .codecov.sh
  chmod +x .codecov.sh
  ./.codecov.sh
fi
