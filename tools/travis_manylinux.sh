#!/usr/bin/env bash

# Echo each command.
set -x

# Exit on error.
set -e

export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
export PATH="$deps_dir/bin:$PATH"

docker pull ${DOCKER_IMAGE};
docker run --rm -e TWINE_PASSWORD -e PYGMO_BUILD_TYPE -e TRAVIS_TAG -v `pwd`:/pygmo2 $DOCKER_IMAGE bash /pygmo2/tools/docker_manylinux.sh

set +e
set +x
