"""Marks the ``slr/common`` folder as a Python package.

"Common" means code that BOTH halves of the product use: the laptop server and the Raspberry Pi
device. It holds the configuration loader (config.py), the wire protocol that the two sides
speak over the network (protocol.py), network discovery (discovery.py) and a few small shared
data types (types.py). The file itself is intentionally empty.
"""
