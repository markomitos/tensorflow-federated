#!/usr/bin/env bash
set -e
exec 2>&1

PROJECT_DIR=$(pwd)

echo "Installing IDE plugins..."
/mnt/jetbrains/system/ide/bin/remote-dev-server.sh \
  installPlugins \
  "$PROJECT_DIR" \
  "com.intellij.ml.llm" \
  "mobi.hsz.idea.gitignore" || true

bazel_plugin=""
if [ "$CODECANVAS_JETBRAINS_IDE" = "clion" ]; then
  bazel_plugin="clwb"
elif [ "$CODECANVAS_JETBRAINS_IDE" = "idea" ] || [ "$CODECANVAS_JETBRAINS_IDE" = "pycharm" ]; then
  bazel_plugin="ijwb"
elif [ -z "$CODECANVAS_JETBRAINS_IDE" ]; then
  echo "CODECANVAS_JETBRAINS_IDE is not set. Set it to 'clion', 'idea' or 'pycharm' to install the correct Bazel plugin."
else
  echo "CODECANVAS_JETBRAINS_IDE is set to an unknown value: $CODECANVAS_JETBRAINS_IDE. Set it to 'clion', 'idea' or 'pycharm' to install the correct Bazel plugin."
fi

if [ -n "$bazel_plugin" ]; then
  echo "Installing Bazel plugin $bazel_plugin..."
  /mnt/jetbrains/system/ide/bin/remote-dev-server.sh \
    installPlugins \
    "$PROJECT_DIR" \
    "com.google.idea.bazel.${bazel_plugin}" || true
fi
echo "Finished installing plugins."

# Removing alien_plugins.txt to avoid a blocking confirmation dialog
alien_plugins_files=$(find ~/.config/JetBrains -name alien_plugins.txt)
if [ -n "$alien_plugins_files" ]; then
  echo "Removing $alien_plugins_files"
  rm -f $alien_plugins_files
else
  echo "No alien_plugins.txt files found."
fi
