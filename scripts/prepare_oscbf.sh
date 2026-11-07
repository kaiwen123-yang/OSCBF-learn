#!/usr/bin/env bash
# User-run installation. No robot SDK imports or hardware access.
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
mkdir -p src envs logs
exec > >(tee "logs/oscbf_prepare_$(date +%Y%m%d_%H%M%S).log") 2>&1
trap 'rc=$?; echo "OSCBF_PREPARE_FAILED line $LINENO (exit $rc)"; exit "$rc"' ERR
repo="$project_root/src/oscbf"
prefix="$project_root/envs/oscbf"
commit=6082329f8e1d61ee8e875a63a5cd80a14d513e12
if [[ ! -e "$repo/.git" ]]; then
  git clone --no-checkout https://github.com/StanfordASL/oscbf.git "$repo"
  git -C "$repo" checkout --detach "$commit"
fi
test "$(git -C "$repo" rev-parse HEAD)" = "$commit"
git -C "$repo" diff --quiet HEAD
git -C "$repo" log -1 --format='%H%n%cI%n%s' > logs/oscbf_source.txt
if [[ ! -x "$prefix/bin/python" ]]; then
  "${PIPER_PYTHON:-python3.10}" -m venv "$prefix"
fi
"$prefix/bin/python" -I -c 'import sys; assert sys.version_info[:2] == (3, 10); print(sys.executable)'
"$prefix/bin/python" -I -m pip install -r "$project_root/requirements/oscbf.txt"
"$prefix/bin/python" -I -m pip install --no-deps -e "$repo" -e "$project_root/src/pyAgxArm"
"$prefix/bin/python" -I -m pip check
"$prefix/bin/python" -I -m pip freeze > logs/oscbf_pip_freeze.txt
"$prefix/bin/python" -I -c 'import sys; print(sys.version); print(sys.executable)' > logs/oscbf_python.txt
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu \
  "$prefix/bin/python" -I "$project_root/scripts/check_oscbf.py"
echo 'OSCBF_ENV_OK: official velocity QP warmed up on CPU; GUI demo can now run.'
