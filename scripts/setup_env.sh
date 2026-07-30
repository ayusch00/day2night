#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${PWD}}"
PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
DTBS_DIR="${REPO_ROOT}/third_party/DTBS"
DTBS_COMMIT="ea92f6910a1b36c12625a54789cdeb6a6e5dbff4"

if [[ ! -f "${REPO_ROOT}/requirements.txt" ]]; then
  echo "Run this script from the repository root or set REPO_ROOT." >&2
  exit 1
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      candidate_version="$("${candidate}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      if [[ "${candidate_version}" == "3.10" ]]; then
        PYTHON_BIN="$(command -v "${candidate}")"
        break
      fi
    fi
  done
fi

if [[ -z "${PYTHON_BIN}" ]] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python 3.10 was not found. Install it or set PYTHON_BIN=/path/to/python3.10." >&2
  exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.10" ]]; then
  echo "Python 3.10 is required; ${PYTHON_BIN} reports ${PYTHON_VERSION}." >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements.txt"

if [[ ! -d "${DTBS_DIR}/.git" ]]; then
  mkdir -p "${REPO_ROOT}/third_party"
  git clone https://github.com/hf618/DTBS.git "${DTBS_DIR}"
fi

git -C "${DTBS_DIR}" fetch origin "${DTBS_COMMIT}"
git -C "${DTBS_DIR}" checkout --detach "${DTBS_COMMIT}"

python "${REPO_ROOT}/scripts/check_environment.py"

echo "Environment ready: ${VENV_DIR}"
echo "DTBS is pinned to: ${DTBS_COMMIT}"
echo "Download the documented checkpoints before training or inference."
