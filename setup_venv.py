#!/usr/bin/env python3
import subprocess
import sys
import os
import shutil

# Specify minimum required Python version
MIN_PYTHON_VERSION = (3, 10)

def check_python_version(python_exe):
    """Check if Python executable meets minimum version requirement."""
    try:
        result = subprocess.run([python_exe, "-c", 
                               "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"],
                              capture_output=True, text=True, check=True)
        version_str = result.stdout.strip()
        version_parts = [int(x) for x in version_str.split('.')]
        version_tuple = tuple(version_parts[:2])  # Only major.minor for comparison
        
        return version_tuple >= MIN_PYTHON_VERSION, version_str
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return False, None

def find_suitable_python():
    """Find a Python executable that meets the version requirement."""
    # Try different Python executables in order of preference
    candidates = [
        sys.executable,  # Current Python running this script
        "python3",
        "python",
        "python3.11",
        "python3.12",
        "python3.10",
        "/usr/bin/python3",
        "/opt/homebrew/bin/python3",
    ]
    
    for candidate in candidates:
        is_suitable, version = check_python_version(candidate)
        if is_suitable:
            print(f"Found suitable Python: {candidate} (version {version})")
            return candidate
    
    return None

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    print("Setting up Python environment...")
    
    # Find suitable Python executable
    python_exe = find_suitable_python()
    if not python_exe:
        min_version_str = f"{MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"
        print(f"Error: Python {min_version_str} or higher not found!")
        print(f"Please install Python {min_version_str}+ and make sure it's in your PATH.")
        sys.exit(1)
    
    # Remove existing venv if it exists
    if os.path.exists(".venv"):
        print("Removing existing virtual environment...")
        shutil.rmtree(".venv")
    
    # Create virtual environment with suitable Python version
    run_command([python_exe, "-m", "venv", ".venv"])
    
    # Determine pip path
    if os.name == 'nt':  # Windows
        pip_path = os.path.join(".venv", "Scripts", "pip")
    else:  # Unix-like
        pip_path = os.path.join(".venv", "bin", "pip")

    # Install requirements
    run_command([pip_path, "install", "-r", "requirements.txt"])

    # Create default folders needed for the GUI
    print("Creating default folders...")
    folders = [
        "data_raw",
        "data_trace",
        "data_trace/backup_trace",
        "data_trace/aperMap",
        "data_trace/aperMap/backup_aperMap",
        "data_trace/aperMap/slits",
        "data_trace/aperMap/trace_coefs"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created: {folder}")

    print("Setup complete!")
    if os.name == 'nt':
        print("To activate: .venv\\Scripts\\activate")
        print("To run GUI: .venv\\Scripts\\python ifum_apermap_maker_GUI.py")
    else:
        print("To activate:")
        print("  Bash/Zsh: source .venv/bin/activate")
        print("  Fish: source .venv/bin/activate.fish")
        print("To run GUI: .venv/bin/python ifum_apermap_maker_GUI.py")

    # Create a cross-platform run script
    if os.name == 'nt':  # Windows
        # Create batch file
        with open("run_gui.bat", "w") as f:
            f.write("@echo off\n")
            f.write(".venv\\Scripts\\python ifum_apermap_maker_GUI.py\n")
        # Create PowerShell script as run_gui (no extension)
        with open("run_gui", "w") as f:
            f.write("#!/usr/bin/env pwsh\n")
            f.write(".venv\\Scripts\\python ifum_apermap_maker_GUI.py\n")
        print("Run scripts created: run_gui.bat and run_gui")
        print("Usage: run_gui.bat or ./run_gui (with PowerShell)")
    else:  # Unix-like
        with open("run_gui", "w") as f:
            f.write("#!/bin/bash\n")
            f.write(".venv/bin/python ifum_apermap_maker_GUI.py\n")
        os.chmod("run_gui", 0o755)  # Make executable
        print("Run script created: ./run_gui")

if __name__ == "__main__":
    main()