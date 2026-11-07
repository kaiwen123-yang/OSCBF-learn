#!/usr/bin/env bash
# An interactive project session keeps WSL alive while the USB adapter is attached.
set -e
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV
source "$project_root/envs/oscbf/bin/activate"
export JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1
echo "OSCBF project: $project_root"
echo 'Project Python is active. Keep this WSL terminal open during robot work.'
exec bash --noprofile --norc -i
