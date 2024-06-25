#!/usr/bin/env bash
set -e
exec 2>&1

PROJECT_DIR=$(pwd)

echo "Installing required python packages..."
python3 --version
pip install -r $PROJECT_DIR/requirements.txt

bazel_plugin=""
if [ -d "/mnt/jetbrains/system/ide-plugins/clwb" ]; then
  bazel_plugin="clwb"
elif [ -d "/mnt/jetbrains/system/ide-plugins/ijwb" ]; then
  bazel_plugin="ijwb"
fi

if [ -n "$bazel_plugin" ]; then
  echo "Found bazel plugin: $bazel_plugin"
  aspect_args="--aspects=@@intellij_aspect//:intellij_info_bundled.bzl%intellij_info_aspect --override_repository=intellij_aspect=/mnt/jetbrains/system/ide-plugins/${bazel_plugin}/aspect"
else
  echo "Bazel plugin not found"
fi

if [ -f /tmp/remote_bazel_cache/key.json ]; then
  echo "Found remote cache key, enabling upload of local results..."
  cache_args="--google_credentials=/tmp/remote_bazel_cache/key.json --remote_upload_local_results"
else
  echo "Remote cache key not found, using read-only cache..."
fi

echo "Building Bazel project..."
bazel build \
  --keep_going --color=no --progress_in_terminal_title=no \
  $aspect_args \
  $cache_args \
  //tensorflow_federated/...
