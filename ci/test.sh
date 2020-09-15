#!/bin/bash

set -e
set -eo pipefail

source activate ${ENV_NAME}
python -m unittest asaptools.tests.vprinterTests
python -m unittest asaptools.tests.timekeeperTests
python -m unittest asaptools.tests.partitionTests
python -m unittest asaptools.tests.partitionArrayTests
python -m unittest asaptools.tests.simpleCommP1STests
mpirun -np 4 python -m unittest asaptools.tests.simpleCommParTests
mpirun -np 4 python -m unittest asaptools.tests.simpleCommParDivTests
