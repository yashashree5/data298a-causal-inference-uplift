#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "${NUPLAN_DATA_ROOT:-/data}" ]]; then
  echo "Dataset mount is missing: ${NUPLAN_DATA_ROOT:-/data}" >&2
  exit 1
fi

mkdir -p "${NUPLAN_EXP_ROOT:-/artifacts/nuplan}"
exec "$@"
