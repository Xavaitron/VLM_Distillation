#!/bin/bash
# Install autoattack and robustbench from offline packages
# Usage: bash install_offline.sh
# This extracts pre-packaged site-packages directly into your Python environment.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

echo "Installing to: $SITE_PACKAGES"

echo "Extracting autoattack..."
tar xzf "$SCRIPT_DIR/autoattack.tar.gz" -C "$SITE_PACKAGES"

echo "Extracting robustbench..."
tar xzf "$SCRIPT_DIR/robustbench.tar.gz" -C "$SITE_PACKAGES"

echo "Done! Verifying..."
python3 -c "from autoattack import AutoAttack; print('✓ autoattack imported successfully')"
python3 -c "import robustbench; print('✓ robustbench imported successfully')"
