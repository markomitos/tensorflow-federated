#!/usr/bin/env bash
set -e
exec 2>&1

PROJECT_DIR=$(pwd)

echo "Installing required python packages..."
python3 --version
python3 -m pip install --root-user-action=ignore -r $PROJECT_DIR/requirements.txt

if nvidia-smi &> /dev/null; then
  python3 -m pip install "tensorflow[and-cuda]>=2.16.0,==2.16.*"
fi
