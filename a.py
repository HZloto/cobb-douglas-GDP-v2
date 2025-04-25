#!/usr/bin/env python3
"""
install_numpy.py – Ensures pip is available and installs/updates NumPy.
Usage: python install_numpy.py
"""

import sys
import subprocess
import shutil

def run(cmd: list[str]) -> None:
    """Run a shell command and forward output/errors live."""
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main() -> None:
    python_exe = sys.executable
    pip_exe = shutil.which("pip") or shutil.which("pip3")

    # 1) Bootstrap pip if missing
    if pip_exe is None:
        print("pip not found – bootstrapping with ensurepip …")
        run([python_exe, "-m", "ensurepip", "--upgrade"])
        pip_exe = shutil.which("pip3") or shutil.which("pip")

    # 2) Upgrade pip itself (recommended)
    run([python_exe, "-m", "pip", "install", "--upgrade", "pip"])

    # 3) Install (or upgrade) NumPy
    run([python_exe, "-m", "pip", "install", "--upgrade", "numpy"])

    print("\n✅ NumPy installation complete.")
    # Optional sanity-check
    import numpy
    print(f"NumPy version: {numpy.__version__}")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"❌ Command failed with exit code {e.returncode}")
