"""TLV320DAC3100 setup for the Pi's I2S headphone output.

The Adafruit TLV320DAC3100 breakout needs I2C configuration before it will
produce sound.  It is wired to the Pi's I2S pins (BCK/WSEL/DIN) plus I2C
(SDA/SCL) and a reset pin.  No MCLK connection is needed: the codec derives its
PLL clock from BCLK.
"""

from __future__ import annotations

import time
from typing import Any

DEFAULT_ADDRESS = 0x18
DEFAULT_RESET_PIN = 4


class Tlv320Dac:
    """Reset and configure one TLV320DAC3100 for stereo headphone playback."""

    def __init__(self, sample_rate: int, reset_pin: int = DEFAULT_RESET_PIN,
                 address: int = DEFAULT_ADDRESS):
        # Imported here so laptops without GPIO/I2C support can still import this module.
        from gpiozero import DigitalOutputDevice
        import adafruit_tlv320
        import board
        import busio

        self._dac: Any = None
        self._i2c: Any = None
        self._reset = DigitalOutputDevice(reset_pin, initial_value=False)
        try:
            time.sleep(0.1)
            self._reset.on()
            time.sleep(0.1)

            self._i2c = busio.I2C(board.SCL, board.SDA)
            self._dac = adafruit_tlv320.TLV320DAC3100(self._i2c, address=address)

            self._dac.configure_clocks(sample_rate=sample_rate, bit_depth=16)
            self._dac.speaker_output = False
            self._dac.headphone_output = True

            # Leave the codec near unity gain; the project's software volume does the rest.
            self._dac.dac_volume = -3.0
            self._dac.headphone_volume = 0.0
            self._dac.headphone_left_gain = 0
            self._dac.headphone_right_gain = 0
            time.sleep(0.35)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Power down the headphone output and release GPIO/I2C resources."""
        if self._dac is not None:
            try:
                self._dac.headphone_output = False
            except Exception:
                pass
            self._dac = None
        if self._i2c is not None:
            try:
                self._i2c.deinit()
            except Exception:
                pass
            self._i2c = None
        if self._reset is not None:
            try:
                self._reset.close()
            except Exception:
                pass
            self._reset = None
