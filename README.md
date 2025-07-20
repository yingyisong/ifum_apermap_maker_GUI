# ifum_apermap_maker_GUI

Tkinter GUI that creates the AperMap files for usage in the IFUM quicklook GUI.

Requirements: python>=3.10, [matplotlib](https://matplotlib.org/), [ccdproc](https://ccdproc.readthedocs.io/en/latest/install.html), [specutils](https://specutils.readthedocs.io/en/stable/installation.html).

<!-- [astropy](https://www.astropy.org/),  -->

## Clone and install

```bash
git clone https://github.com/yingyisong/ifum_apermap_maker_GUI.git
cd ifum_apermap_maker_GUI
python setup.py
```

## Run the GUI:

```bash
# navigate to the GUI folder
./run_gui
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
