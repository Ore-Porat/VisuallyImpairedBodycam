"""This (otherwise empty) file tells Python that the ``slr/device`` folder is a "package".

A package is just a folder of Python files that can be imported with dotted names, for example
``from slr.device.camera import open_camera``. Without this file Python would not look inside
the folder.

The ``slr.device`` package holds everything that runs ON THE CAMERA DEVICE (the Raspberry Pi
Zero 2 W hidden inside the SLR camera body, or a laptop pretending to be it in "sim" mode):
the camera, the physical buttons and mode switch, the volume dial, the headphone audio, and the
two network links to the laptop server (UDP for video frames, WebSocket for everything else).
"""
