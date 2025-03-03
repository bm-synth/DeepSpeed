# tests directory-specific settings - this file is run automatically by pytest before any tests are run

import sys
import pytest
import os
from os.path import abspath, dirname, join

# Set this environment variable for the T5 inference unittest(s) (e.g. google/t5-v1_1-small)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(dirname(__file__)), "src"))
sys.path.insert(1, git_repo_path)
