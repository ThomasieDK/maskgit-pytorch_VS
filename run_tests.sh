#!/bin/bash
set -e
python -m py_compile tamg/*.py models/stage2/*.py train.py
python -m unittest discover -s tests
