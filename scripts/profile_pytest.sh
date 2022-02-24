#!/bin/bash
python3 -m cProfile -o profile -m pytest $@
