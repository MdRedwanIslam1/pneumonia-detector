#!/usr/bin/env bash

# Activate the Linux virtual environment stored inside the WSL disk on D:.
source "$HOME/venvs/pneumonia-detector/bin/activate"

# Make the Windows GPU driver and pip-installed CUDA libraries visible to TensorFlow.
cuda_library_paths="$(python -c 'import glob, site; print(":".join(glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib")))')"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${cuda_library_paths}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Open the project directory shared from the Windows D: drive.
cd /mnt/d/MLProjects/pneumonia-detector

echo "Pneumonia detector WSL environment activated."
echo "Project: $(pwd)"
