#!/usr/bin/env bash
# Install SLR SLR device software on a Raspberry Pi Zero 2 W (Raspberry Pi OS Lite 64-bit).
# Run from the repo root on the Pi:   sudo ./deploy/install_pi.sh
#
# Never installs an ML runtime (PRD rule 4): no torch, ultralytics or OpenCV DNN on the Pi.
# OpenCV itself is installed only to read the USB camera; detection stays on the laptop.
#
# What this script does, in order
#   1. installs the system packages the device needs (camera, GPIO, audio libraries)
#   2. turns on the SPI bus (volume dial), I2C (TLV320 DAC setup) and I2S audio output
#   3. creates a Python "virtual environment" and installs this project into it
#   4. gives the normal user permission to use GPIO, SPI, I2C, audio and the camera
#   5. lets the device power itself off without a password (for the shutdown button combo)
#   6. installs a systemd service so the software starts on every boot (FR-50)
# FR-xx / PRD references point to the product spec, PRD.pdf.
#
# Shell basics used below
#   Lines starting with # are comments. "$VAR" reads a variable. `a || b` runs b only if a
#   fails; `a && b` runs b only if a succeeds. `>>` appends to a file. `-y` answers "yes" to
#   prompts so the script can run unattended.

# Safety settings: stop at the first error (-e), treat unset variables as errors (-u), and
# make a failure anywhere in a pipeline (a | b) count as a failure (-o pipefail).
set -euo pipefail

# Root (administrator) rights are needed for apt, config.txt, sudoers and systemd.
# $EUID is the numeric id of the user running the script; 0 means root.
if [[ $EUID -ne 0 ]]; then
  echo "run with sudo" >&2
  exit 1
fi

# Work out where the repo is (the folder above this script), which user to install for
# (the one who typed sudo, or "pi"), and where the boot config file lives (its path moved
# between Raspberry Pi OS versions, so check both).
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_USER="${SUDO_USER:-pi}"
CONFIG=/boot/firmware/config.txt
[[ -f $CONFIG ]] || CONFIG=/boot/config.txt

# apt is the package manager on Raspberry Pi OS (Debian). "update" refreshes the package
# list; "install" fetches: Python venv/pip tooling, OpenCV for the USB camera, lgpio for
# GPIO access, I2C tools for the DAC, PortAudio + ALSA utilities for sound, numpy, and
# simplejpeg for fast JPEG compression. OpenCV and the other binary packages only come
# from apt, which is why the venv below can see system packages.
echo "== apt packages"
apt-get update
apt-get install -y python3-venv python3-pip python3-opencv python3-lgpio \
  libportaudio2 alsa-utils i2c-tools python3-smbus python3-numpy python3-simplejpeg v4l-utils

# The Pi's hardware features are switched on in config.txt, read by the firmware at boot.
#   raspi-config nonint do_spi 0  -> enable SPI (the MCP3008 volume dial chip)
#   raspi-config nonint do_i2c 0  -> enable I2C (the TLV320 DAC's control bus)
#   dtoverlay=i2s-dac             -> generic I2S output; correct for the TLV320DAC3100
#     (only appended if the line is not already there, so re-running is safe)
echo "== enable SPI (MCP3008), I2C (TLV320), and the I2S DAC (headphone jack)"
raspi-config nonint do_spi 0 || true
raspi-config nonint do_i2c 0 || true
# Remove the old PCM5102A/HifiBerry overlay if a previous install added it.
sed -i '/^dtoverlay=hifiberry-dac/d' "$CONFIG"
grep -q '^dtoverlay=i2s-dac' "$CONFIG" || echo 'dtoverlay=i2s-dac' >> "$CONFIG"
# the Zero 2 W has no analog audio; make sure the onboard HDMI audio doesn't grab card 0
sed -i 's/^dtparam=audio=on/dtparam=audio=off/' "$CONFIG"

# A virtual environment (venv) is a private Python install inside the repo (.venv/) so this
# project's packages do not interfere with the system's. --system-site-packages lets it
# also use the apt-installed libraries (OpenCV etc.). "sudo -u" runs the commands as the
# normal user so the files are owned by them, not by root.
# `pip install -e ".[device]"` installs this repo in "editable" mode with the "device"
# extras from pyproject.toml (no machine-learning packages).
echo "== Python venv (system site packages so apt libraries are visible)"
sudo -u "$RUN_USER" python3 -m venv --system-site-packages "$REPO/.venv"
sudo -u "$RUN_USER" "$REPO/.venv/bin/pip" install --upgrade pip
sudo -u "$RUN_USER" "$REPO/.venv/bin/pip" install -e "$REPO[device]"

# On Linux, access to hardware is controlled by group membership. Add the user to the
# groups that own the GPIO pins, the SPI/I2C buses, the sound card and the camera.
echo "== groups for GPIO / SPI / I2C / audio / camera"
usermod -aG gpio,spi,i2c,audio,video "$RUN_USER"

# "sudoers" is the file that says who may run what as root. This rule lets the user run
# ONLY the shutdown command without a password, so the shutter+advance combo can power
# the device off cleanly. chmod 440 = read-only, as sudo requires for its config files.
echo "== allow the device to power off safely (shutdown combo)"
echo "$RUN_USER ALL=(root) NOPASSWD: /sbin/shutdown, /usr/sbin/shutdown" > /etc/sudoers.d/slr-shutdown
chmod 440 /etc/sudoers.d/slr-shutdown

# systemd is the service manager that starts programs at boot. Copy our unit file
# (deploy/slr-device.service) into place, replacing the @REPO@ / @USER@ placeholders with
# the real values (sed does the text substitution). "daemon-reload" tells systemd to
# re-read its files; "enable" makes the service start automatically on boot.
echo "== systemd service (FR-50: start on boot)"
sed -e "s#@REPO@#$REPO#g" -e "s#@USER@#$RUN_USER#g" "$REPO/deploy/slr-device.service" \
  > /etc/systemd/system/slr-device.service
systemctl daemon-reload
systemctl enable slr-device.service

# The device plays a few built-in clips ("Starting up", "Volume 5" ...) that are rendered on
# the laptop. Warn if they are missing; the device would otherwise be silent until connected.
if ! ls "$REPO"/device_assets/system_clips/*.wav >/dev/null 2>&1; then
  echo "WARNING: no system clips. On the laptop run 'slr clips build-system' and copy device_assets/." >&2
fi

# Print the remaining manual steps.
echo
echo "Done. Next:"
echo "  1. set server.host in $REPO/config/device.yaml to the laptop's hotspot address"
echo "  2. reboot (needed for the overlays), then check:"
echo "       aplay -l ; ls /dev/spidev0.0 ; i2cdetect -y 1 ; v4l2-ctl --list-devices"
echo "  3. slr device test-controls      # verify buttons/switch/dial wiring"
echo "  4. slr device test-camera        # open the USB camera and save one frame"
echo "  5. slr device calibrate-volume   # FR-44"
echo "  6. sudo systemctl start slr-device ; journalctl -u slr-device -f"
