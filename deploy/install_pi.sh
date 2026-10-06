#!/usr/bin/env bash
# Install SLR SLR device software on a Raspberry Pi Zero 2 W (Raspberry Pi OS Lite 64-bit).
# Run from the repo root on the Pi:   sudo ./deploy/install_pi.sh
#
# Never installs an ML runtime (PRD rule 4): no torch, ultralytics or OpenCV DNN on the Pi.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "run with sudo" >&2
  exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_USER="${SUDO_USER:-pi}"
CONFIG=/boot/firmware/config.txt
[[ -f $CONFIG ]] || CONFIG=/boot/config.txt

echo "== apt packages"
apt-get update
apt-get install -y python3-venv python3-pip python3-picamera2 python3-lgpio \
  libportaudio2 alsa-utils i2c-tools python3-numpy python3-simplejpeg

echo "== enable I2C (ADS1115 volume dial) and the I2S DAC (headphone jack)"
raspi-config nonint do_i2c 0 || true
grep -q '^dtoverlay=hifiberry-dac' "$CONFIG" || echo 'dtoverlay=hifiberry-dac' >> "$CONFIG"
# the Zero 2 W has no analog audio; make sure the onboard HDMI audio doesn't grab card 0
sed -i 's/^dtparam=audio=on/dtparam=audio=off/' "$CONFIG"

echo "== Python venv (system site packages so picamera2 from apt is visible)"
sudo -u "$RUN_USER" python3 -m venv --system-site-packages "$REPO/.venv"
sudo -u "$RUN_USER" "$REPO/.venv/bin/pip" install --upgrade pip
sudo -u "$RUN_USER" "$REPO/.venv/bin/pip" install -e "$REPO[device]"

echo "== groups for GPIO / I2C / audio"
usermod -aG gpio,i2c,audio,video "$RUN_USER"

echo "== allow the device to power off safely (shutdown combo)"
echo "$RUN_USER ALL=(root) NOPASSWD: /sbin/shutdown, /usr/sbin/shutdown" > /etc/sudoers.d/slr-shutdown
chmod 440 /etc/sudoers.d/slr-shutdown

echo "== systemd service (FR-50: start on boot)"
sed -e "s#@REPO@#$REPO#g" -e "s#@USER@#$RUN_USER#g" "$REPO/deploy/slr-device.service" \
  > /etc/systemd/system/slr-device.service
systemctl daemon-reload
systemctl enable slr-device.service

if ! ls "$REPO"/device_assets/system_clips/*.wav >/dev/null 2>&1; then
  echo "WARNING: no system clips. On the laptop run 'slr clips build-system' and copy device_assets/." >&2
fi

echo
echo "Done. Next:"
echo "  1. set server.host in $REPO/config/device.yaml to the laptop's hotspot address"
echo "  2. reboot (needed for the I2S overlay), then check: aplay -l ; i2cdetect -y 1 (expect 48)"
echo "  3. slr device test-controls      # verify wiring"
echo "  4. slr device calibrate-volume   # FR-44"
echo "  5. sudo systemctl start slr-device ; journalctl -u slr-device -f"
