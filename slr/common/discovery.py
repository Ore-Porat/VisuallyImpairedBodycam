"""mDNS service advertisement / discovery for _slrslr._tcp (FR-51, P1). Optional: needs zeroconf.

How it fits in
--------------
The Raspberry Pi device has to know the laptop's network address to connect to it. You can type
the address into config/device.yaml by hand, or set it to ``auto`` and let this file find the
laptop automatically. The laptop server shouts "I am an SLR server at this address" onto the
local network (``Advertiser``), and the device listens for that shout (``discover``).

FR-51 is the requirement number in the product spec, PRD.pdf; "P1" means it is a nice-to-have
priority, which is why the feature is optional: if the ``zeroconf`` library is not installed,
everything else still works and you just type the address in.

Words you will see
------------------
- mDNS (multicast DNS) / zeroconf / Bonjour: a way for devices on the same Wi-Fi to announce
  and find services by name, with no central server. Printers and smart TVs use it.
- service type: a label for "what kind of thing is this". Ours is ``_slrslr._tcp``.
- socket: the operating system's object for sending and receiving network data.
- IP address: the number that identifies a computer on the network, e.g. 192.168.137.1.
- port: a number (0..65535) that picks which program on that computer gets the data.
- ImportError: the error Python raises when a library is not installed. Caught here so the
  missing library just disables discovery instead of crashing.
- try / except / finally: run the ``try`` block; if it fails, run ``except``; run ``finally``
  no matter what (used here to always close sockets).
"""

from __future__ import annotations

import logging
import socket
import time

log = logging.getLogger(__name__)

# The name we announce / search for on the network. The trailing ".local." is required by mDNS.
SERVICE_TYPE = "_slrslr._tcp.local."


def _local_ip() -> str:
    """Find this computer's own IP address on the network it would use to reach others.

    Trick: open a UDP socket and "connect" it to a far-away address. No data is actually sent
    (UDP connect only picks a route), but afterwards the socket knows which local address it
    would use, and we read that back. Falls back to 127.0.0.1 if there is no network at all.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; picks the outward interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Advertiser:
    """Announces this laptop's server on the local network so the device can find it.

    Created by the server when it starts (if ``network.mdns`` is true in server.yaml) and
    closed when the server stops. The announcement carries the WebSocket port as the main
    port, and the UDP and HTTP ports as extra properties.
    """

    def __init__(self, ws_port: int, udp_port: int, http_port: int, name: str = "slr-server"):
        """Register the service. Any failure is logged and simply disables advertising.

        - ``ws_port``: the WebSocket (control channel) port.
        - ``udp_port``: the UDP (video frames) port.
        - ``http_port``: the shopping list web editor port.
        - ``name``: the instance name shown on the network.
        """
        # ``self.zc`` stays None if anything goes wrong; ``close`` checks it.
        self.zc = None
        try:
            # Imported inside the function so a missing library only affects this feature.
            from zeroconf import ServiceInfo, Zeroconf
        except ImportError:
            log.info("zeroconf not installed; mDNS advertisement disabled")
            return
        ip = _local_ip()
        # Describe the service: type, full name, address (as 4 raw bytes), main port, extras.
        self.info = ServiceInfo(
            SERVICE_TYPE, f"{name}.{SERVICE_TYPE}", addresses=[socket.inet_aton(ip)], port=ws_port,
            properties={"udp_port": str(udp_port), "http_port": str(http_port)},
        )
        try:
            self.zc = Zeroconf()
            self.zc.register_service(self.info)
            log.info("advertising %s on %s:%d", SERVICE_TYPE, ip, ws_port)
        except Exception as e:
            log.warning("mDNS advertisement failed: %s", e)
            self.zc = None

    def close(self) -> None:
        """Stop announcing and release the network resources. Safe to call if never started."""
        if self.zc:
            try:
                self.zc.unregister_service(self.info)
            finally:
                self.zc.close()


def discover(timeout_s: float = 5.0) -> tuple[str, int, int, int] | None:
    """Returns (host, ws_port, udp_port, http_port) of the first server found, or None.

    Called by the device when ``server.host`` is ``auto`` in device.yaml. It listens on the
    network for up to ``timeout_s`` seconds. Returns None if zeroconf is not installed or no
    server answered in time.
    """
    try:
        from zeroconf import ServiceBrowser, Zeroconf
    except ImportError:
        return None
    # Results are appended to this list by the listener below (which runs on zeroconf's own
    # thread), and the loop at the bottom waits until it is non-empty.
    found: list = []

    class Listener:
        """The three methods zeroconf calls as services appear, change or disappear."""

        def add_service(self, zc, type_, name):
            """A matching service was announced: look up its details and record them."""
            info = zc.get_service_info(type_, name)
            if info and info.addresses:
                # Properties arrive as raw bytes; turn them into normal strings.
                props = {k.decode(): v.decode() for k, v in (info.properties or {}).items() if v}
                # Default ports are used if the server did not include them.
                found.append((socket.inet_ntoa(info.addresses[0]), info.port,
                              int(props.get("udp_port", 8766)), int(props.get("http_port", 8767))))

        def update_service(self, *a):
            """Required by zeroconf's interface; we do not care about updates."""
            pass

        def remove_service(self, *a):
            """Required by zeroconf's interface; we do not care about removals."""
            pass

    zc = Zeroconf()
    try:
        # Start listening; the browser keeps running in the background until zc is closed.
        ServiceBrowser(zc, SERVICE_TYPE, Listener())
        # Poll every 0.1 s until something is found or the deadline passes.
        # time.monotonic() is a clock that only ever goes forward (never affected by the user
        # changing the system time), which makes it the right choice for timeouts.
        end = time.monotonic() + timeout_s
        while not found and time.monotonic() < end:
            time.sleep(0.1)
    finally:
        zc.close()
    return found[0] if found else None
