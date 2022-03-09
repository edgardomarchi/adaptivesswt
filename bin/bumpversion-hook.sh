#!/bin/bash

if [[ $BUMPVERSION_NEW_VERSION ]]; then
  echo "BUMPVERSION_NEW_VERSION is set to '$BUMPVERSION_NEW_VERSION'"
  make dist
  git add setup.py
  exit 0
fi
