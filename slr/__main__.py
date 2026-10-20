"""Entry point for ``python -m slr``.

Python runs this file when you type ``python -m slr ...`` in a terminal. All it does is hand
over to the real command-line program in ``slr/cli.py``. (Installing the project with
``pip install -e .`` also creates a shorter ``slr`` command that does the same thing; see
pyproject.toml.)
"""

# Bring in the main() function from cli.py ...
from slr.cli import main

# ... and run it. It reads the words you typed after "slr" and decides what to do.
main()
