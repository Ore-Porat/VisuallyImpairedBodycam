"""SLR SLR: Single Lens Reflex Sightless Label Reader.

This file marks the ``slr`` folder as a Python "package" (a folder of related code files that
Python can import by name, e.g. ``import slr.cli``). It is run automatically the first time
anything from ``slr`` is imported.

The package holds every piece of software for the product:

- ``slr/common``  things shared by both halves (config loading, the wire protocol, small types)
- ``slr/server``  the laptop side (object detection, guidance logic, speech clips, debug view)
- ``slr/device``  the Raspberry Pi side (camera, buttons, volume dial, headphones)
- ``slr/cli.py``  the ``slr`` command you type in a terminal
- ``slr/sim.py``  runs server + device together on one laptop for testing
- ``slr/bench.py`` measures speed and reliability

See README.md for the big picture.
"""

# The version number of this software. It is sent to the server in the "hello" message so the
# logs show which build the device was running.
__version__ = "0.1.0"
