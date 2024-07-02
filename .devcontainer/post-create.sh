#!/usr/bin/env bash
set -e
exec 2>&1

PROJECT_DIR=$(pwd)

echo "Installing required python packages..."
python3 --version
pip install --root-user-action=ignore -r $PROJECT_DIR/requirements.txt
