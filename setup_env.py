#!/usr/bin/env python3
"""
Cross-platform environment setup script for mimiciv_backdoor_study.

This script helps set up the development environment on Windows, macOS, and Linux.
It creates a virtual environment, installs dependencies, and provides platform-specific
instructions for PyArrow installation.

Usage:
  python setup_env.py
"""
import os
import sys
import subprocess
import platform
from pathlib import Path
import argparse

def run_cmd(cmd, check=True, shell=False):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(cmd, check=check, shell=shell, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None

def detect_conda():
    """Check if conda is available."""
    try:
        result = run_cmd(["conda", "--version"])
        return result is not None
    except:
        return False

def create_venv(env_name="mimiciv_env"):
    """Create a virtual environment."""
    system = platform.system().lower()

    if system == "windows":
        python_cmd = "python"
        venv_cmd = [python_cmd, "-m", "venv", env_name]
    else:
        python_cmd = "python3"
        venv_cmd = [python_cmd, "-m", "venv", env_name]

    print(f"Creating virtual environment: {env_name}")
    result = run_cmd(venv_cmd)
    if result is None:
        print("Failed to create virtual environment")
        return False

    return True

def get_pip_path(env_name="mimiciv_env"):
    """Get the path to pip in the virtual environment."""
    system = platform.system().lower()
    if system == "windows":
        return Path(env_name) / "Scripts" / "pip.exe"
    else:
        return Path(env_name) / "bin" / "pip"

def install_dependencies(env_name="mimiciv_env"):
    """Install project dependencies."""
    pip_path = get_pip_path(env_name)

    print("Upgrading pip...")
    run_cmd([str(pip_path), "install", "--upgrade", "pip"])

    print("Installing dependencies from requirements.txt...")
    req_file = Path("mimiciv_backdoor_study/requirements.txt")
    if req_file.exists():
        result = run_cmd([str(pip_path), "install", "-r", str(req_file)])
        if result is None:
            print("Failed to install dependencies")
            return False
    else:
        print(f"Requirements file not found: {req_file}")
        return False

    return True

def setup_pyarrow():
    """Provide platform-specific PyArrow installation instructions."""
    system = platform.system().lower()

    print("\n" + "="*50)
    print("PyArrow Setup Instructions")
    print("="*50)

    if system == "darwin":  # macOS
        print("On macOS, PyArrow often requires conda-forge for pre-built wheels:")
        print("1. Install Miniconda or Anaconda if not already installed")
        print("2. Create conda environment:")
        print("   conda create -n mimiciv_env -c conda-forge python=3.11 pyarrow -y")
        print("3. Activate environment:")
        print("   conda activate mimiciv_env")
        print("4. Install remaining dependencies:")
        print("   pip install -r mimiciv_backdoor_study/requirements.txt")
    elif system == "linux":
        print("On Linux, PyArrow should install normally with pip.")
        print("If you encounter issues, try:")
        print("  pip install --no-cache-dir pyarrow")
    elif system == "windows":
        print("On Windows, PyArrow installation may require:")
        print("1. Microsoft Visual C++ Build Tools")
        print("2. Or use conda-forge:")
        print("   conda create -n mimiciv_env -c conda-forge python=3.11 pyarrow -y")
    else:
        print(f"Unknown platform: {system}")
        print("Please check PyArrow documentation for your platform.")

def main():
    parser = argparse.ArgumentParser(description="Set up mimiciv_backdoor_study environment")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force recreation of virtual environment if it exists")
    parser.add_argument("--env-name", default="mimiciv_env",
                       help="Name of the virtual environment to create")

    args = parser.parse_args()

    print("mimiciv_backdoor_study Environment Setup")
    print("="*50)

    env_name = args.env_name

    # Check if environment already exists
    if Path(env_name).exists():
        if args.force:
            # Check if we're currently running from within this environment
            current_exe = sys.executable
            env_python = Path(env_name) / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python")

            if env_python.resolve() == Path(current_exe).resolve():
                print(f"Cannot remove environment '{env_name}' while running from within it.")
                print("Please run this command from outside the environment, or manually delete the environment folder first.")
                return
            else:
                print(f"Removing existing environment '{env_name}'...")
                import shutil
                shutil.rmtree(env_name)
        else:
            print(f"Environment '{env_name}' already exists.")
            print("Use --force to recreate it, or activate it manually:")
            system = platform.system().lower()
            if system == "windows":
                print(f"  {env_name}\\Scripts\\activate.bat")
            else:
                print(f"  source {env_name}/bin/activate")
            return

    # Create virtual environment
    if not create_venv(env_name):
        return

    # Install dependencies
    if not install_dependencies(env_name):
        return

    print(f"\n✅ Setup completed successfully!")
    print(f"Activate the environment with:")
    system = platform.system().lower()
    if system == "windows":
        print(f"  {env_name}\\Scripts\\activate.bat")
    else:
        print(f"  source {env_name}/bin/activate")

    print("\nNext steps:")
    print("1. Activate the environment (see command above)")
    print("2. Run: python run_project.py baseline")
    print("   Or: run_project.bat baseline (Windows)")

    # PyArrow instructions
    setup_pyarrow()

if __name__ == "__main__":
    main()
