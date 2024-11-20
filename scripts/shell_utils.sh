#!/bin/bash
# shell_utils.sh
# Created by: Jonathan Mikler on 20/Nov/24


check_conda_env() {
  CONDA_ENV_REQ=lgvit
  # check that 'lgvit' conda environment is activated
  CONDA_ENV=$(conda info --envs | grep "*" | awk '{print $1}')
  if [ "$CONDA_ENV" != $CONDA_ENV_REQ ]; then
    echo -e "ERROR: Conda environment '${CONDA_ENV_REQ}' is not activated. Please activate it before running the script."
    exit 1
  fi
}