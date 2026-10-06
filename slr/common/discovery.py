"""mDNS service advertisement / discovery for _slrslr._tcp (FR-51, P1). Optional: needs zeroconf."""

from __future__ import annotations

import logging
import socket
import time

log = logging.getLogger(__name__)

SERVICE_TYPE = "_slrslr._tcp.local."


def _local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; picks the outward interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Advertiser:
    def __init__(self, ws_port: int, udp_port: int, http_port: int, name: str = "slr-server"):
        self.zc = None
        try:
            from zeroconf import ServiceInfo, Zeroconf
        except ImportError:
            log.info("zeroconf not installed; mDNS advertisement disabled")
            return
        ip = _local_ip()
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
        if self.zc:
            try:
                self.zc.unregister_service(self.info)
            finally:
                self.zc.close()


def discover(timeout_s: float = 5.0) -> tuple[str, int, int, int] | None:
    """Returns (host, ws_port, udp_port, http_port) of the first server found, or None."""
    try:
        from zeroconf import ServiceBrowser, Zeroconf
    except ImportError:
        return None
    found: list = []

    class Listener:
        def add_service(self, zc, type_, name):
            info = zc.get_service_info(type_, name)
            if info and info.addresses:
                props = {k.decode(): v.decode() for k, v in (info.properties or {}).items() if v}
                found.append((socket.inet_ntoa(info.addresses[0]), info.port,
                              int(props.get("udp_port", 8766)), int(props.get("http_port", 8767))))

        def update_service(self, *a):
            pass

        def remove_service(self, *a):
            pass

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, SERVICE_TYPE, Listener())
        end = time.monotonic() + timeout_s
        while not found and time.monotonic() < end:
            time.sleep(0.1)
    finally:
        zc.close()
    return found[0] if found else None
