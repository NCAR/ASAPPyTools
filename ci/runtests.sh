#!/bin/bash

set -e
set -eo pipefail

coverage run -p -m asaptools.tests.vprinterTests
coverage run -p -m asaptools.tests.timekeeperTests
coverage run -p -m asaptools.tests.partitionTests
coverage run -p -m asaptools.tests.partitionArrayTests
coverage run -p -m asaptools.tests.simpleCommP1STests
mpirun -np 4 coverage run -p -m asaptools.tests.simpleCommParTests
mpirun -np 4 coverage run -p -m asaptools.tests.simpleCommParDivTests
coverage combine
coverage xml
