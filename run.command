#!/bin/bash
# Double-click to start Wave to Elon.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "Run setup.command first."
  read -r -p "Press Enter to close..."
  exit 1
fi
.venv/bin/python wave_to_elon.py "$@"
