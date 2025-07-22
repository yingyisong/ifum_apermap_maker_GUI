# ifum_apermap_maker_GUI

Tkinter GUI that creates the AperMap files for usage in the IFUM quicklook GUI.

Requirements: python>=3.10, [matplotlib](https://matplotlib.org/), [ccdproc](https://ccdproc.readthedocs.io/en/latest/install.html), [specutils](https://specutils.readthedocs.io/en/stable/installation.html).

<!-- [astropy](https://www.astropy.org/),  -->

## Download

```bash
git clone https://github.com/yingyisong/ifum_apermap_maker_GUI.git
cd ifum_apermap_maker_GUI
```

## Install Option 1: Conda

### Install

```bash
conda create -n ifum_gui python=3.12
conda activate ifum_gui
python install -r requirements.txt
```

### Run the GUI

```bash
conda activate ifum_gui

# navigate to the GUI folder
python ifum_apermap_maker_GUI.py
```

### Uninstall

```bash
conda env remove -n ifum_gui
```

## Install Option 2: Virtual Environment

### Install

```bash
python setup_venv.py
```

### Run the GUI

```bash
# navigate to the GUI folder
./run_gui
```

### Uninstall

Linux/macOS:

```bash
# navigate to the GUI folder
rm -rf .venv run_gui
```

Windows (Command Prompt):

```cmd
REM navigate to the GUI folder
rmdir /s .venv & del run_gui.bat run_gui
```

Windows (PowerShell):

```powershell
# navigate to the GUI folder
Remove-Item -Recurse -Force .venv, run_gui.bat, run_gui
```

<!--## Clone and intiatlize the GUI

```bash
git clone https://github.com/yingyisong/ifum_apermap_maker_GUI.git
cd ifum_apermap_maker_GUI
./create_folders
```

## Start the GUI using Docker

Require: [Docker](https://www.docker.com/get-started/), [XQuartz](https://www.xquartz.org/) for macOS

Steps:

1. (macOS) Setup XQuartz
   1. Start XQuartz
   1. Open Settings, go to Security and check both items
   1. Quit and restart XQuartz
1. Start Docker Desktop
1. Start the GUI using the terminal command:
   ```bash
   ./run_maker
   ```

## Alternatively, start the GUI without using Docker:

Require: [astropy](https://www.astropy.org/), [ccdproc](https://ccdproc.readthedocs.io/en/latest/install.html), [specutils](https://specutils.readthedocs.io/en/stable/installation.html)

Start the GUI using the terminal command:

```bash
conda activate pypeit
python ifum_apermap_maker_GUI.py
``` -->
