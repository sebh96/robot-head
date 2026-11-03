#!/usr/bin/env bash
# Sets up robot-head on a Raspberry Pi 5 with Raspberry Pi OS Lite (64-bit), and starts it on every boot.
# Run it from the robot-head folder:  bash pi/install.sh
# Running it again is safe, for example after a git pull or after plugging in other sound devices.
set -euo pipefail
cd "$(dirname "$0")/.."
HERE="$(pwd)"
ME="$(id -un)"

step() { printf '\n== %s\n' "$1"; }

step "System packages"
sudo apt-get update
# PortAudio for sound, the libraries OpenCV needs, and tools for checking I2C and sound by hand.
sudo apt-get install -y git curl i2c-tools alsa-utils libportaudio2 libsndfile1 libgl1 libglib2.0-0

step "I2C for the servo board"
sudo raspi-config nonint do_i2c 0  # 0 means on
for group in i2c gpio audio video; do
    if getent group "$group" >/dev/null; then sudo usermod -aG "$group" "$ME"; fi
done

step "uv and the Python packages (the first time takes a few minutes)"
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
"$UV" sync --frozen --extra pi

step "Settings"
if [ ! -f .env ]; then
    # Piper instead of Kokoro (too slow on a Pi), and the real servos.
    "$UV" run --frozen --extra pi python - <<'PY'
from robot_head.devices import save_settings
save_settings({"TTS_ENGINE": "piper", "TTS_VOICE": "", "TTS_SPEED": "1.0", "MOTORS": "pca9685"})
PY
    echo "Created .env for the Pi: Piper voice, real servos. Paste your OpenRouter key under Settings in the control panel."
else
    echo "Keeping your .env as it is."
fi
web_host="$(sed -n 's/^WEB_HOST=//p' .env | tr -d '"')"
if [ "$web_host" != "0.0.0.0" ]; then
    echo
    echo "The control panel shows all settings, your API key included."
    read -r -p "Open it to your home network, so you can reach it from your PC or phone? Only on a network you trust. [y/N] " answer || answer=""
    if [[ "$answer" =~ ^[YyJj] ]]; then
        "$UV" run --frozen --extra pi python -c 'from robot_head.devices import save_settings; save_settings({"WEB_HOST": "0.0.0.0"})'
        web_host="0.0.0.0"
    fi
fi

step "Sound"
# The robot talks to ALSA's "default" device. Point it at a USB speaker and a USB microphone (a webcam's
# counts), with automatic sample rate conversion, so any USB sound device works without further setup.
pick() {  # $1: aplay or arecord, $2: what it's for. Prints "card,device" or nothing.
    local options
    mapfile -t options < <("$1" -l 2>/dev/null | sed -n 's/^card [0-9]*: \([^ ]*\) \[\(.*\)\], device \([0-9]*\):.*/\1,\3 (\2)/p' | grep -v '^vc4hdmi' || true)
    if [ "${#options[@]}" -eq 0 ]; then
        return
    elif [ "${#options[@]}" -eq 1 ]; then
        echo "${options[0]%% *}"
    else
        echo "Which one is the $2?" >&2
        local PS3="Number: " choice
        select choice in "${options[@]}"; do
            if [ -n "$choice" ]; then echo "${choice%% *}"; break; fi
        done
    fi
}
speaker="$(pick aplay speaker)"
microphone="$(pick arecord microphone)"
if [ -z "$speaker" ] || [ -z "$microphone" ]; then
    if [ -z "$speaker" ] && [ -z "$microphone" ]; then
        missing="speaker or microphone"
    elif [ -z "$speaker" ]; then
        missing="speaker"
    else
        missing="microphone"
    fi
    echo "Found no USB $missing. Plug it in and run bash pi/install.sh again; skipping the sound setup for now."
else
    sudo tee /etc/asound.conf >/dev/null <<EOF
# Written by robot-head's pi/install.sh: play on ${speaker%,*}, record from ${microphone%,*}.
pcm.!default {
    type asym
    playback.pcm "plughw:CARD=${speaker%,*},DEV=${speaker#*,}"
    capture.pcm "plughw:CARD=${microphone%,*},DEV=${microphone#*,}"
}
ctl.!default {
    type hw
    card ${speaker%,*}
}
EOF
    # Turn the volume up and the microphone on; the controls are named differently on every device.
    for control in PCM Speaker Master Headphone; do
        amixer -q -c "${speaker%,*}" sset "$control" 90% unmute 2>/dev/null || true
    done
    for control in Mic Capture; do
        amixer -q -c "${microphone%,*}" sset "$control" 80% cap 2>/dev/null || true
    done
    sudo alsactl store 2>/dev/null || true
    echo "Speaker: ${speaker%,*}, microphone: ${microphone%,*}."
    echo "Check them with: speaker-test -t wav -c 1 -l 1    and    arecord -d 3 test.wav && aplay test.wav"
fi

step "Start on boot"
sed -e "s|@USER@|$ME|g" -e "s|@DIR@|$HERE|g" -e "s|@UV@|$UV|g" pi/robot-head.service |
    sudo tee /etc/systemd/system/robot-head.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable robot-head

step "Done"
echo "Restart the Pi once with: sudo reboot"
echo "It starts by itself after that. The first start downloads the speech models (about 1 GB), so give it a few minutes."
echo "Watch what it's doing with: journalctl -u robot-head -f"
if [ "$web_host" = "0.0.0.0" ]; then
    echo "Control panel: http://$(hostname).local:8765"
else
    echo "Control panel: on your PC, run   ssh -L 8765:127.0.0.1:8765 $ME@$(hostname).local"
    echo "and open http://127.0.0.1:8765 while that connection is open."
fi
