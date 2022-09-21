# ifum_apermap_maker_GUI
Tkinter GUI that creates the AperMap for the IFUM quicklook GUI

# ifum_aperMap_maker

Require: PypeIt_m2fs (forked from https://pypeit.readthedocs.io/en/release/index.html, v1.8.2dev)

Install PypeIt_m2fs:
```sh
# Setup a clean python environment (recommend)
conda create -n pypeit_m2fs python=3.9
conda activate pypeit_m2fs

# In the directory to install PypeIt_m2fs
git clone https://github.com/yingyisong/PypeIt_m2fs.git
cd PypeIt_m2fs
pip install -e ".[dev,pyqt5]"
```

Usage:
1) Edit `input_sample.txt`
2) Run as `python make_aperMap.py input_sample.txt`
