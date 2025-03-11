# ifum_apermap_maker_GUI
Tkinter GUI that creates the AperMap for the IFUM quicklook GUI

## Start the GUI using Docker
Require: [Docker](https://www.docker.com/get-started/), [XQuartz](https://www.xquartz.org/) for macOS

Steps:
1. (macOS) Setup XQuartz
   a. Start XQuartz
   b. Open Settings, go to Security and check both items
   c. Quit and restart XQuartz
2. Clone the repo using the terminal command:
   ```bash
   git clone https://github.com/yingyisong/ifum_apermap_maker_GUI.git
   ```
3. Start Docker Desktop
4. Start the GUI using the terminal command:
   ```bash
   ./run_maker
   ```

## Alternatively, start the GUI without using Docker:
Require: [astropy](https://www.astropy.org/), [ccdproc](https://ccdproc.readthedocs.io/en/latest/install.html), [specutils](https://specutils.readthedocs.io/en/stable/installation.html)

Start the GUI using the terminal command:
```bash
python ifum_apermap_maker_GUI.py
```