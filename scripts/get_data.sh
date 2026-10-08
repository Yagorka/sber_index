#!/usr/bin/env bash
# Совместимый старый вход: проверяет локальные данные, ничего не скачивает.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PY:-python}" scripts/check_inputs.py --profile full
