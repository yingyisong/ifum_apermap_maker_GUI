#!/usr/bin/env python
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter.messagebox import showinfo
import customtkinter as ctk

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from datetime import datetime

from astropy.io import fits
import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit

from utils_io import IFUM_UNIT, pack_4fits_simple, func_parabola, readFloat_space, write_pypeit_file, write_trace_file, cut_apermap, cached_fits_open
from utils_trace import load_trace, reshape_trace_by_curvature, do_trace_v3, create_apermap

import subprocess
#from multiprocessing import Process

# window geometry
# window_width = 1800
# window_height = 930
CTRL_WIDTH  = 650       # control panel width
CTRL_HEIGHT = 820       # control panel height
IMG_WIDTH   = 800       # each image window width
IMG_HEIGHT  = 800       # each image window height
img_figsize = (6, 6)


def main():
    #### Create the entire GUI program
    program = IFUM_AperMap_Maker()

    #### Start the GUI event loop
    program.window.mainloop()


# def check_appearance():
#     """Checks DARK/LIGHT mode of macos."""
#     """True=DARK, False=LIGHT"""
#     cmd = 'defaults read -g AppleInterfaceStyle'
#     p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                          stderr=subprocess.PIPE, shell=True)
#     return bool(p.communicate()[0])


# #print(check_appearance())
# if check_appearance():
#     STEP_LABEL_COLOR = 'yellow'
#     LABEL_COLOR = 'mediumseagreen'
#     BG_COLOR = 'black'
# else:
#     STEP_LABEL_COLOR = 'blue'
#     LABEL_COLOR = 'black'
#     BG_COLOR = 'lightgray'

def check_appearance():
    """Checks DARK/LIGHT mode of macOS.  True = DARK, False = LIGHT."""
    cmd = 'defaults read -g AppleInterfaceStyle'
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    return bool(p.communicate()[0])


# ── CTk global appearance ────────────────────────────────────────────────────
if check_appearance():
    ctk.set_appearance_mode("dark")
    #STEP_LABEL_COLOR = '#ffd600'       # yellow on dark bg
    STEP_LABEL_COLOR = '#1565c0'       # blue on light bg
else:
    ctk.set_appearance_mode("light")
    STEP_LABEL_COLOR = '#1565c0'       # blue on light bg

ctk.set_default_color_theme("blue")

# Badge colours for the sunken file-name indicators
BADGE_ACTIVE  = ('#FFD700', '#8B6914')   # gold when active
BADGE_NEUTRAL = ('gray80', 'gray30')     # neutral resting state

class IFUM_AperMap_Maker:

    def __init__(self):
        # Setup CustomTkinter Window
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # ── Main control window ──────────────────────────────────────────────
        self.window = ctk.CTk()
        self.window.title("IFUM AperMap Maker")
        self.window.geometry(f"{CTRL_WIDTH}x{CTRL_HEIGHT}+0+0")
        self.window.resizable(True, True)
        self.window.protocol("WM_DELETE_WINDOW", self._on_main_close)

        # ── b-side image window ──────────────────────────────────────────────
        self.win_b = ctk.CTkToplevel(self.window)
        self.win_b.title("b-side Image")
        self.win_b.geometry(f"{IMG_WIDTH}x{IMG_HEIGHT}+{CTRL_WIDTH+10}+0")
        self.win_b.resizable(True, True)
        self.win_b.protocol("WM_DELETE_WINDOW", lambda: None)   # keep open

        # ── r-side image window ──────────────────────────────────────────────
        self.win_r = ctk.CTkToplevel(self.window)
        self.win_r.title("r-side Image")
        self.win_r.geometry(f"{IMG_WIDTH}x{IMG_HEIGHT}+{CTRL_WIDTH+IMG_WIDTH+20}+0")
        self.win_r.resizable(True, True)
        self.win_r.protocol("WM_DELETE_WINDOW", lambda: None)

        # initialize the rest of the GUI
        self.initialize_gui()

    def _on_main_close(self):
        self.window.quit()
        self.window.destroy()

    def initialize_gui(self):
        """Initialize the GUI components."""
        #### menubar
        self.menubar = tk.Menu(self.window)
        self.window.config(menu=self.menubar)
        self.menu_file = tk.Menu(self.menubar, tearoff=0)
        self.menu_file.add_command(label="Load Curve", command=self.load_curve_file)
        self.menu_file.add_command(label="Save Curve", command=self.save_curve_file)
        #self.menu_file.add_command(label="Save as", command=self.save_file_as)
        self.menu_file.add_command(label="Exit", command=self._on_main_close)
        self.menubar.add_cascade(label="File", menu=self.menu_file)

        # Control panel (fills the main window, no scroll)
        self.frame1 = ctk.CTkFrame(self.window)
        self.frame1.pack(fill="both", expand=True, padx=6, pady=6)

        # Column weight so entry fields expand
        for c in range(1, 7):
            self.frame1.columnconfigure(c, weight=1)

        # Image frames inside their dedicated windows
        self.frame2 = ctk.CTkFrame(self.win_b)   # b-side canvas host
        self.frame2.pack(fill="both", expand=True)

        self.frame3 = ctk.CTkFrame(self.win_r)   # r-side canvas host
        self.frame3.pack(fill="both", expand=True)

        # initialize other widgets and values
        self.initialize_widgets()

    def initialize_widgets(self):
        """Initialize all widgets."""
        #### IFUM units
        self.LSB = IFUM_UNIT('LSB')
        self.STD = IFUM_UNIT('STD')
        self.HR = IFUM_UNIT('HR')
        self.M2FS = IFUM_UNIT('M2FS')
        self.UNKNOWN = IFUM_UNIT('unknown')
        self.ifu_type = self.UNKNOWN

        #### global widgets
        self.ent_folder = None
        self.box_files = None
        self.lbl_file_curve = None

        #### backend values
        self.data_full = np.ones((4048, 4048), dtype=np.int32)
        self.data_full2 = np.ones((4048, 4048), dtype=np.int32)
        self.file_current = "0000"

        self.folder_rawdata = "./data_raw/"
        self.folder_trace   = "./data_trace/"
        self.folder_curve   = "./curve_files/"
        self.path_MasterSlits = ' '
        self.labelname_mono   = "band_name"

        self.points = []
        self.x_last, self.y_last = -1., -1.
        self.curve_points = np.array([])
        self.param_curve_b = np.array([1.22771242e-05, 2.28414233e+03, 7.87506089e+02]) #np.zeros(3)
        self.param_curve_r = np.array([1.53314740e-05, 2.12797487e+03, 6.75423701e+02]) #np.zeros(3)
        self.param_edges_b = np.array([418., 1250., 1250.-418.])#np.zeros(2)
        self.param_edges_r = np.array([426., 1258., 1258.-426.])#np.zeros(2)
        self.param_edges_offset = self.param_edges_r[0]-self.param_edges_b[0]
        #self.param_curve_b = np.array([-1.67977832e-05, 1.86465356e+03, 9.65048123e+02]) #np.zeros(3) for STD_b0107 Triplet
        #self.param_curve_r = np.array([-1.76051975e-05, 2.19290161e+03, 1.29282266e+03]) #np.zeros(3)
        #self.param_edges_b = np.array([603., 1571., 968.])#np.zeros(2)
        #self.param_edges_r = np.array([600., 1568., 968.])#np.zeros(2)
        self.param_smash_range = '0.45,0.55'

        #### initialize tk variables
        self.fit_files = ctk.StringVar()
        self.shoe = ctk.StringVar()
        self.pca = ctk.StringVar()
        self.state_edge_lock_r = ctk.IntVar()
        self.state_edge_lock_b = ctk.IntVar()

        self.txt_param_curve_A_b = ctk.StringVar(value=['%.3e'%self.param_curve_b[0]])
        self.txt_param_curve_B_b = ctk.StringVar(value=['%.1f'%self.param_curve_b[1]])
        self.txt_param_curve_C_b = ctk.StringVar(value=['%.1f'%self.param_curve_b[2]])
        self.txt_param_curve_A_r = ctk.StringVar(value=['%.3e'%self.param_curve_r[0]])
        self.txt_param_curve_B_r = ctk.StringVar(value=['%.1f'%self.param_curve_r[1]])
        self.txt_param_curve_C_r = ctk.StringVar(value=['%.1f'%self.param_curve_r[2]])
        self.txt_param_edges_X1_b = ctk.StringVar(value=['%.0f'%self.param_edges_b[0]])
        self.txt_param_edges_X2_b = ctk.StringVar(value=['%.0f'%self.param_edges_b[1]])
        self.txt_param_edges_dX_b = ctk.StringVar(value=['%.0f'%self.param_edges_b[2]])
        self.txt_param_edges_X1_r = ctk.StringVar(value=['%.0f'%self.param_edges_r[0]])
        self.txt_param_edges_X2_r = ctk.StringVar(value=['%.0f'%self.param_edges_r[1]])
        self.txt_param_edges_dX_r = ctk.StringVar(value=['%.0f'%self.param_edges_r[2]])
        self.txt_param_edges_offset = ctk.StringVar(
            value=['%.0f'%(self.param_edges_r[0]-self.param_edges_b[0])]
        )

        self.txt_folder_trace = ctk.StringVar(value=[self.folder_trace])
        self.txt_smash_range = ctk.StringVar(value=[self.param_smash_range])
        self.txt_labelname_mono = ctk.StringVar(value=[self.labelname_mono])

        #### create all widgets
        #self.my_counter = None  # All attributes should be initialize in init
        self.create_widgets_files(line_start=0, line_num=2)  # step 0
        self.create_widgets_curve(line_start=2, line_num=4)  # step 1
        self.create_widgets_edges(line_start=6, line_num=5)  # step 2
        self.create_widgets_trace(line_start=11, line_num=3)  # step 3
        self.create_widgets_pypeit(line_start=14, line_num=4) # step 4
        self.create_widgets_mono(line_start=18, line_num=4)   # step 6 (optional)
        self.create_widgets_add_slits(line_start=22, line_num=4)   # step 5 (obsolete)
        self.bind_widgets()

        #### initialize widgets
        self.shoe.set('r')
        self.pca.set('on')
        self.state_edge_lock_r.set(0)
        self.state_edge_lock_b.set(0)
        self.refresh_folder()
        self.init_image1()
        self.init_image2()

        # get the default FG color of the entry
        # self.DEFAULT_FG = self.ent_folder.cget("fg")
        # self.DEFAULT_FG_DISABLED = self.ent_folder.cget("disabledforeground")
        self.DEFAULT_FG = self.ent_folder.cget("text_color")
        self.DEFAULT_FG_DISABLED = ("gray50", "gray50") 

        # # Force the content frame to maintain minimum size
        # self.content_frame.configure(width=CTRL_WIDTH, height=CTRL_HEIGHT)
        # # Force update of scrollbars after all widgets are created
        # self.content_frame.update_idletasks()

    # =========================================================================
    # WIDGET CREATION METHODS
    # =========================================================================

    def _step_label(self, row, tag, desc):
        """Helper: place a coloured step header across frame1."""
        ctk.CTkLabel(self.frame1, text=tag, text_color=STEP_LABEL_COLOR,
                     font=ctk.CTkFont(weight="bold")
                     ).grid(row=row, column=0, sticky="w", padx=(6, 2), pady=2)
        ctk.CTkLabel(self.frame1, text=desc, text_color=STEP_LABEL_COLOR,
                     font=ctk.CTkFont(weight="bold")
                     ).grid(row=row, column=1, columnspan=6, sticky="w", padx=2, pady=2)

    def _file_badge(self, row, col, text="0000", colspan=1):
        """Helper: sunken file-name badge using a rounded CTkLabel."""
        lbl = ctk.CTkLabel(self.frame1, text=text,
                           fg_color=BADGE_NEUTRAL, corner_radius=4,
                           padx=6, pady=2)
        lbl.grid(row=row, column=col, columnspan=colspan,
                 sticky="e", padx=4, pady=2)
        return lbl

    def load_curve_file(self):
        '''  load a curve file and update param_curve '''

        #### open a txt file containing the curve parameters
        pathname = filedialog.askopenfilename(initialdir=self.folder_curve, title="Select file", filetypes=(("txt files", "*.txt"), ("all files", "*.*")))
        dirname, filename = os.path.split(pathname)
        if os.path.isfile(pathname) and filename.startswith("curve") and filename.endswith(".txt"):
            file = np.loadtxt(pathname, dtype='str')
            mask_bside = file[:, 0] == 'b'

            #### update param_curve
            popt = file[mask_bside, 1:4].astype(np.float64).flatten()
            self.param_curve_b = np.array(popt)

            popt = file[~mask_bside, 1:4].astype(np.float64).flatten()
            self.param_curve_r = np.array(popt)

            self.renew_param_curve()

            #### update param_edges
            temp = file[mask_bside, 4:6].astype(np.float64).flatten()
            self.param_edges_b = np.array([temp[0], temp[0]+temp[1], temp[1]])

            temp = file[~mask_bside, 4:6].astype(np.float64).flatten()
            self.param_edges_r = np.array([temp[0], temp[0]+temp[1], temp[1]])

            self.param_edges_offset = self.param_edges_r[0]-self.param_edges_b[0]

            self.renew_param_edges()

            #### show message
            info_temp = 'Steps 1 & 2 parameters loaded!\n\n Location: %s'%(pathname)
            self.popup_showinfo('File loaded', info_temp) 

        self.window.focus_force()

    def save_curve_file(self):
        '''  save the curve parameters '''

        today_temp = datetime.today().strftime('%y%m%d')
        filename = "curve_%s_%s_%s_%s_%s_%s_%s.txt"%(
            self.ifu_type.label, 
            self.HDR_BINNING,
            self.HDR_SLIDE,
            self.HDR_SLITNAME, 
            self.HDR_CONFIG, 
            self.lbl_file_trace.cget("text"), 
            today_temp)
        pathname = os.path.join(self.folder_curve, filename)
        file = open(pathname, 'w')
        file.write("#side A B C X1 dX\n")
        file.write("b %.3e %.1f %.1f %.0f %.0f\n"%(
            self.param_curve_b[0], self.param_curve_b[1], self.param_curve_b[2], 
            self.param_edges_b[0], self.param_edges_b[2]))
        file.write("r %.3e %.1f %.1f %.0f %.0f\n"%(
            self.param_curve_r[0], self.param_curve_r[1], self.param_curve_r[2], 
            self.param_edges_r[0], self.param_edges_r[2]))
        file.close()

        #### show message
        info_temp = 'Steps 1 & 2 parameters saved!\n\n Location: %s'%pathname
        self.popup_showinfo('File saved', info_temp)

        self.window.focus_force()

    def create_widgets_files(self, line_start, line_num):
        """ step 0 list raw data files """
        start, lines = line_start, line_num
        rows = np.arange(start, start+lines)
        #### folder
        lbl_folder = ctk.CTkLabel(self.frame1, text="Folder")
        lbl_folder.grid(row=rows[0], column=0, sticky="w", padx=(5, 2), pady=1)

        self.ent_folder = ctk.CTkEntry(self.frame1, textvariable=ctk.StringVar(value=[self.folder_rawdata]))
        self.ent_folder.grid(row=rows[0], column=1, columnspan=5, sticky="ew", padx=2, pady=1)

        self.btn_folder = ctk.CTkButton(self.frame1, width=80, text="Raw DIR...", command=self.open_folder)
        self.btn_folder.grid(row=rows[0], column=6, sticky="e", padx=2, pady=1)

        self.btn_refresh = ctk.CTkButton(self.frame1, width=80, text="Refresh", command=self.refresh_folder)
        self.btn_refresh.grid(row=rows[0], column=7, sticky="e", padx=2, pady=1)

        #### file list
        #lbl_files = ctk.CTkLabel(self.frame1, text="Files:", fg=LABEL_COLOR, bg=BG_COLOR)
        #lbl_files.grid(row=rows[2], column=0, sticky="w", padx=(5, 2), pady=1)

        # Using standard Tkinter Listbox because CustomTkinter doesn't have a native Listbox widget
        # self.box_files = tk.Listbox(self.frame1, listvariable=self.fit_files, height=5, 
        #                             bg="gray20", fg="white", selectbackground="#1f538d")
        # self.box_files.grid(row=rows[1], column=0, columnspan=8, sticky="nsew", padx=5, pady=1)
        _mode_idx = 1 if ctk.get_appearance_mode() == "Dark" else 0
        _frame_fg  = ctk.ThemeManager.theme["CTkFrame"]["fg_color"][_mode_idx]
        _entry_fg  = ctk.ThemeManager.theme["CTkEntry"]["fg_color"][_mode_idx]
        _text_col  = ctk.ThemeManager.theme["CTkLabel"]["text_color"][_mode_idx]
        _sel_col   = ctk.ThemeManager.theme["CTkButton"]["fg_color"][_mode_idx]
        self.box_files = tk.Listbox(
            self.frame1, listvariable=self.fit_files, height=5,
            bg=_entry_fg, fg=_text_col,
            selectbackground=_sel_col, selectforeground="white",
            relief="flat", borderwidth=0, highlightthickness=0)
        self.box_files.grid(row=rows[1], column=0, columnspan=8,
                            sticky="nsew", padx=6, pady=2)

        # # scroll bar style
        # style = ttk.Style()
        # style.theme_use('clam')
        # style.configure("Vertical.TScrollbar", gripcount=0,
        #         background=BG_COLOR, darkcolor=LABEL_COLOR, lightcolor=LABEL_COLOR,
        #         troughcolor="gray", bordercolor=BG_COLOR, arrowcolor="gray")

        # box_scrollbar = ttk.Scrollbar(self.frame1, orient="vertical")
        # box_scrollbar.grid(row=rows[1], column=7, sticky='nse')
        box_scrollbar = ctk.CTkScrollbar(self.frame1, orientation="vertical", command=self.box_files.yview, height=60)
        box_scrollbar.grid(row=rows[1], column=7, sticky='nse', padx=(0, 5), pady=1)

        self.box_files.config(yscrollcommand=box_scrollbar.set)
        # box_scrollbar.config(command=self.box_files.yview) 

    def create_widgets_curve(self, line_start, line_num):
        """ step 1 fit curvature using an arc or twilight file """
        start, lines = line_start, line_num
        rows = np.arange(start, start+lines)

        # lbl_step1 = ctk.CTkLabel(self.frame1, text="Step 1:")
        # lbl_step1.grid(row=rows[0], column=0, sticky="w", padx=(5, 2), pady=1)
        # lbl_step1_desc = ctk.CTkLabel(self.frame1, text="Fit curvature of a constant wavelength (load an ARC/solar file)")
        # lbl_step1_desc.grid(row=rows[0], column=1, columnspan=6, sticky="w")

        # self.lbl_file_curve = ctk.CTkLabel(self.frame1, text="0000", corner_radius=4, fg_color=("gray80", "gray20"))
        # self.lbl_file_curve.grid(row=rows[0], column=6, sticky="e", padx=2)

        self._step_label(rows[0], "Step 1:",
                         "Fit curvature of a constant wavelength (load an ARC/solar file)")
        self.lbl_file_curve = self._file_badge(rows[0], col=6, text="0000")

        self.btn_load_curve = ctk.CTkButton(self.frame1, width=60, text="Load", command=self.load_4fits_curve)
        self.btn_load_curve.grid(row=rows[0], column=7, sticky="e", padx=2, pady=1)

        #### pick points
        lbl_note_curve = ctk.CTkLabel(self.frame1, text="Hint: Select 7 points to fit func: x-C = A*(y-B)^2")
        lbl_note_curve.grid(row=rows[1], column=1, columnspan=5, sticky="w")

        self.btn_select_curve_r = ctk.CTkButton(
            self.frame1, width=60, text="Select (r)", 
            command=lambda: self.pick_points('r'), state='disabled')
        self.btn_select_curve_r.grid(row=rows[1], column=6, sticky="e", padx=2, pady=1)

        self.btn_select_curve_b = ctk.CTkButton(
            self.frame1, width=60, text="Select (b)", 
            command=lambda: self.pick_points('b'), state='disabled')
        self.btn_select_curve_b.grid(row=rows[1], column=7, sticky="e", padx=2, pady=1)

        #### curve parameters (r-side)
        lbl_param_curve_A_r = ctk.CTkLabel(self.frame1, text="r-side:  A =")
        lbl_param_curve_A_r.grid(row=rows[2], column=1, sticky="e", padx=2)
        self.ent_param_curve_A_r = ctk.CTkEntry(self.frame1, width=80, textvariable=self.txt_param_curve_A_r)
        self.ent_param_curve_A_r.grid(row=rows[2], column=2, sticky="ew", padx=1)

        lbl_param_curve_B_r = ctk.CTkLabel(self.frame1, text="B =")
        lbl_param_curve_B_r.grid(row=rows[2], column=3, sticky="e", padx=2)
        self.ent_param_curve_B_r = ctk.CTkEntry(self.frame1, width=80, textvariable=self.txt_param_curve_B_r)
        self.ent_param_curve_B_r.grid(row=rows[2], column=4, sticky="ew", padx=1)

        lbl_param_curve_C_r = ctk.CTkLabel(self.frame1, text="C =")
        lbl_param_curve_C_r.grid(row=rows[2], column=5, sticky="e", padx=2)
        self.ent_param_curve_C_r = ctk.CTkEntry(self.frame1, width=80, textvariable=self.txt_param_curve_C_r)
        self.ent_param_curve_C_r.grid(row=rows[2], column=6, sticky="ew", padx=1)

        self.btn_plot_curve_r = ctk.CTkButton(
            self.frame1, width=60, text='Plot (r)', 
            command=lambda: self.update_curve(None, 'r'))
        self.btn_plot_curve_r.grid(row=rows[2], column=7, sticky="e", padx=2, pady=1)

        #### curve parameters (b-side)
        lbl_param_curve_A_b = ctk.CTkLabel(self.frame1, text="b-side:  A =")
        lbl_param_curve_A_b.grid(row=rows[3], column=1, sticky="e", padx=2)
        self.ent_param_curve_A_b = ctk.CTkEntry(self.frame1, width=80, textvariable=self.txt_param_curve_A_b)
        self.ent_param_curve_A_b.grid(row=rows[3], column=2, sticky="ew", padx=1)

        lbl_param_curve_B_b = ctk.CTkLabel(self.frame1, text="B =")
        lbl_param_curve_B_b.grid(row=rows[3], column=3, sticky="e", padx=2)
        self.ent_param_curve_B_b = ctk.CTkEntry(self.frame1, width=80, textvariable=self.txt_param_curve_B_b)
        self.ent_param_curve_B_b.grid(row=rows[3], column=4, sticky="ew", padx=1)

        lbl_param_curve_C_b = ctk.CTkLabel(self.frame1, text="C =")
        lbl_param_curve_C_b.grid(row=rows[3], column=5, sticky="e", padx=2)
        self.ent_param_curve_C_b = ctk.CTkEntry(self.frame1, width=80, textvariable=self.txt_param_curve_C_b)
        self.ent_param_curve_C_b.grid(row=rows[3], column=6, sticky="ew", padx=1)

        self.btn_plot_curve_b = ctk.CTkButton(
            self.frame1, width=60, text='Plot (b)', 
            command=lambda: self.update_curve(None, 'b'))
        self.btn_plot_curve_b.grid(row=rows[3], column=7, sticky="e", padx=2, pady=1)

    def create_widgets_edges(self, line_start, line_num):
        """ step 2 select edges using a science or twilight file """
        start, lines = line_start, line_num
        rows = np.arange(start, start+lines)

        # lbl_step2 = ctk.CTkLabel(self.frame1, text="Step 2:")
        # lbl_step2.grid(row=rows[0], column=0, sticky="w", padx=(5, 2), pady=1)
        # lbl_step2_desc = ctk.CTkLabel(self.frame1, text="Determine spectral region edges (load a solar/SCI file)")
        # lbl_step2_desc.grid(row=rows[0], column=1, columnspan=6, sticky="w")

        # self.lbl_file_edges = ctk.CTkLabel(self.frame1, text="0000", corner_radius=4, fg_color=("gray80", "gray20"))
        # self.lbl_file_edges.grid(row=rows[0], column=6, sticky="e", padx=2)

        self._step_label(rows[0], "Step 2:",
                         "Determine spectral region edges (load a solar/SCI file)")
        self.lbl_file_edges = self._file_badge(rows[0], col=6, text="0000")

        self.btn_load_edges = ctk.CTkButton(self.frame1, width=60, text="Load", command=self.load_4fits_edges, state='normal')
        self.btn_load_edges.grid(row=rows[0], column=7, sticky="e", padx=2, pady=1)

        #### pick edges
        lbl_note_edges = ctk.CTkLabel(self.frame1, text="Hint: Select 2 points along y-axis middle line")
        lbl_note_edges.grid(row=rows[1], column=1, columnspan=5, sticky="w")

        self.btn_select_edges_r = ctk.CTkButton(
            self.frame1, width=60, text="Select (r)", 
            command=lambda: self.pick_edges('r'), state='disabled')
        self.btn_select_edges_r.grid(row=rows[1], column=6, sticky="e", padx=2, pady=1)

        self.btn_select_edges_b = ctk.CTkButton(
            self.frame1, width=60, text="Select (b)", 
            command=lambda: self.pick_edges('b'), state='disabled')
        self.btn_select_edges_b.grid(row=rows[1], column=7, sticky="e", padx=2, pady=1)

        #### edge parameters (r-side)
        lbl_param_edges_X1_r = ctk.CTkLabel(self.frame1, text="r-side: X1 =")
        lbl_param_edges_X1_r.grid(row=rows[2], column=1, sticky="e", padx=2)
        self.ent_param_edges_X1_r = ctk.CTkEntry(self.frame1, width=80, textvariable=self.txt_param_edges_X1_r)
        self.ent_param_edges_X1_r.grid(row=rows[2], column=2, sticky="ew", padx=1)

        lbl_param_edges_X2_r = ctk.CTkLabel(self.frame1, text="X2 =")
        lbl_param_edges_X2_r.grid(row=rows[2], column=3, sticky="e", padx=2)
        self.ent_param_edges_X2_r = ctk.CTkEntry(self.frame1, width=80, textvariable=self.txt_param_edges_X2_r, state='disabled', text_color=("gray50", "gray50"))
        self.ent_param_edges_X2_r.grid(row=rows[2], column=4, sticky="ew", padx=1)

        lbl_param_edges_dX_r = ctk.CTkLabel(self.frame1, text="dX =")
        lbl_param_edges_dX_r.grid(row=rows[2], column=5, sticky="e", padx=2)
        self.ent_param_edges_dX_r = ctk.CTkEntry(self.frame1, width=80, textvariable=self.txt_param_edges_dX_r)
        self.ent_param_edges_dX_r.grid(row=rows[2], column=6, sticky="ew", padx=1)

        self.btn_plot_edges_r = ctk.CTkButton(
            self.frame1, width=60, text='Plot (r)', 
            command=lambda: self.update_edges(None, 'r'))
        self.btn_plot_edges_r.grid(row=rows[2], column=7, sticky="e", padx=2, pady=1)

        #### edge parameters (b-side)
        lbl_param_edges_X1_b = ctk.CTkLabel(self.frame1, text="b-side: X1 =")
        lbl_param_edges_X1_b.grid(row=rows[3], column=1, sticky="e", padx=2)
        self.ent_param_edges_X1_b = ctk.CTkEntry(self.frame1, width=80, textvariable=self.txt_param_edges_X1_b)
        self.ent_param_edges_X1_b.grid(row=rows[3], column=2, sticky="ew", padx=1)

        lbl_param_edges_X2_b = ctk.CTkLabel(self.frame1, text="X2 =")
        lbl_param_edges_X2_b.grid(row=rows[3], column=3, sticky="e", padx=2)
        self.ent_param_edges_X2_b = ctk.CTkEntry(self.frame1, width=80, textvariable=self.txt_param_edges_X2_b, state='disabled', text_color=("gray50", "gray50"))
        self.ent_param_edges_X2_b.grid(row=rows[3], column=4, sticky="ew", padx=1)

        lbl_param_edges_dX_b = ctk.CTkLabel(self.frame1, text="dX =")
        lbl_param_edges_dX_b.grid(row=rows[3], column=5, sticky="e", padx=2)
        self.ent_param_edges_dX_b = ctk.CTkEntry(self.frame1, width=80, textvariable=self.txt_param_edges_dX_b)
        self.ent_param_edges_dX_b.grid(row=rows[3], column=6, sticky="ew", padx=1)

        self.btn_plot_edges_b = ctk.CTkButton(
            self.frame1, width=60, text='Plot (b)', 
            command=lambda: self.update_edges(None, 'b'))
        self.btn_plot_edges_b.grid(row=rows[3], column=7, sticky="e", padx=2, pady=1)

        #### offset parameters
        lbl_param_edges_offset = ctk.CTkLabel(self.frame1, text="dX1 (r - b) =")
        lbl_param_edges_offset.grid(row=rows[4], column=1, sticky="e", padx=2)
        self.ent_param_edges_offset = ctk.CTkEntry(
            self.frame1, width=80, textvariable=self.txt_param_edges_offset)
        self.ent_param_edges_offset.grid(row=rows[4], column=2, sticky="ew", padx=1)

        #### lock on one side of the edges
        lbl_edge_lock = ctk.CTkLabel(self.frame1, text="Sync")
        lbl_edge_lock.grid(row=rows[4], column=3, sticky="e", padx=2)
        self.cbtn_edge_lock_r = ctk.CTkCheckBox(self.frame1, text='r-side', 
                variable=self.state_edge_lock_r, onvalue=1, offvalue=0, 
                command=lambda: self.lock_edge('r'), text_color='red')
        self.cbtn_edge_lock_r.grid(row=rows[4], column=4, sticky="e", padx=2)
        
        self.cbtn_edge_lock_b = ctk.CTkCheckBox(self.frame1, text='b-side', 
                variable=self.state_edge_lock_b, onvalue=1, offvalue=0, 
                command=lambda: self.lock_edge('b'), text_color='cyan')
        self.cbtn_edge_lock_b.grid(row=rows[4], column=5, columnspan=2, sticky="w", padx=2)

        #### button to load the curve profile
        self.btn_load_all_param = ctk.CTkButton(
            self.frame1, width=120, text="Load curve param.", 
            command=self.load_curve_file, state='normal')
        self.btn_load_all_param.grid(row=rows[4], 
                                     column=6, columnspan=2,
                                     sticky="e", padx=2, pady=1)

    def create_widgets_trace(self, line_start, line_num):
        """ step 3 check and make a masked LED fits file for tracing """
        start, lines = line_start, line_num
        rows = np.arange(start, start+lines)

        # lbl_step3 = ctk.CTkLabel(self.frame1, text="Step 3:")
        # lbl_step3.grid(row=rows[0], column=0, sticky="w", padx=(5, 2), pady=1)
        # lbl_step3_desc = ctk.CTkLabel(self.frame1, text="Make TRACE files (load an LED file)")
        # lbl_step3_desc.grid(row=rows[0], column=1, columnspan=6, sticky="w")

        # self.lbl_file_trace = ctk.CTkLabel(self.frame1, text="0000", corner_radius=4, fg_color=("gray80", "gray20"))
        # self.lbl_file_trace.grid(row=rows[0], column=6, sticky="e", padx=2)

        self._step_label(rows[0], "Step 3:", "Make TRACE files (load an LED file)")
        self.lbl_file_trace = self._file_badge(rows[0], col=6, text="0000")

        self.btn_load_trace = ctk.CTkButton(self.frame1, width=60, text="Load", command=self.load_4fits_trace, state='normal')
        self.btn_load_trace.grid(row=rows[0], column=7, sticky="e", padx=2, pady=1)

        ####
        self.ent_folder_trace = ctk.CTkEntry(self.frame1, textvariable=self.txt_folder_trace, state='normal')
        self.ent_folder_trace.grid(row=rows[1], column=1, columnspan=5, sticky="ew", padx=2)

        self.btn_folder_trace = ctk.CTkButton(self.frame1, width=80, text="Output DIR...", command=self.open_folder_trace, state='normal')
        self.btn_folder_trace.grid(row=rows[1], column=6, sticky="ew", pady=1)

        self.btn_make_trace = ctk.CTkButton(self.frame1, width=60, text="Make", command=self.make_file_trace, state='disabled')
        self.btn_make_trace.grid(row=rows[1], column=7, sticky="e", padx=2, pady=1)

        # add a bottom dashed line to this widget
        lbl_line = ctk.CTkLabel(self.frame1, text="-"*80, text_color='gray')
        lbl_line.grid(row=rows[2], column=0, columnspan=8, sticky="w", padx=5)

    def create_widgets_pypeit(self, line_start, line_num):
        """ step 4 run pypeit for tracing and making the AperMap """
        start, lines = line_start, line_num
        rows = np.arange(start, start+lines)

        # lbl_step4 = ctk.CTkLabel(self.frame1, text="Step 4:")
        # lbl_step4.grid(row=rows[0], column=0, sticky="w", padx=(5, 2), pady=1)
        # lbl_step4_desc = ctk.CTkLabel(self.frame1, text="Make AperMap files (open a TRACE file)")
        # lbl_step4_desc.grid(row=rows[0], column=1, columnspan=5, sticky="w")

        # self.lbl_file_pypeit = ctk.CTkLabel(self.frame1, text="0000_trace", corner_radius=4, fg_color=("gray80", "gray20"))
        # self.lbl_file_pypeit.grid(row=rows[0], column=5, columnspan=2, sticky="e", padx=2)

        self._step_label(rows[0], "Step 4:", "Make AperMap files (open a TRACE file)")
        self.lbl_file_pypeit = self._file_badge(rows[0], col=5, text="0000_trace",
                                                colspan=2)
        self.btn_load_pypeit = ctk.CTkButton(self.frame1, width=60, text="Open", command=self.open_fits_trace, state='normal')
        self.btn_load_pypeit.grid(row=rows[0], column=7, sticky="e", padx=2, pady=1)

        ##### step 4a make a PypeIt file
        #lbl_step4a = tk.Label(self.frame1, text="4a. Make PypeIt files", fg=LABEL_COLOR, bg=BG_COLOR)
        #lbl_step4a.grid(row=rows[1], column=1, columnspan=2, sticky="w")
        ##lbl_smash = tk.Label(self.frame1, text="smash range =", fg=LABEL_COLOR, bg=BG_COLOR)
        ##lbl_smash.grid(row=rows[1], column=4, columnspan=2, sticky="e")

        #lbl_pca = tk.Label(self.frame1, text="PCA:", fg=LABEL_COLOR, bg=BG_COLOR)
        #lbl_pca.grid(row=rows[1], column=4, sticky="w")

        #self.pca1 = tk.Radiobutton(self.frame1, text='off', variable=self.pca, value='off', fg=LABEL_COLOR, bg=BG_COLOR)
        #self.pca1.grid(row=rows[1], column=4, sticky='e')
        #self.pca2 = tk.Radiobutton(self.frame1, text='on', variable=self.pca, value='on', fg=LABEL_COLOR, bg=BG_COLOR)
        #self.pca2.grid(row=rows[1], column=5, sticky='w')

        #self.ent_smash_range = tk.Entry(self.frame1, width=6, textvariable=self.txt_smash_range, state='normal')
        #self.ent_smash_range.grid(row=rows[1], column=6, sticky="ew")

        #self.btn_make_pypeit = tk.Button(self.frame1, width=6, text='Make', command=self.make_file_pypeit, state='disabled', highlightbackground=BG_COLOR)
        #self.btn_make_pypeit.grid(row=rows[1], column=7, sticky='e', padx=2, pady=1)

        #### step 4b run PypeIt
        #lbl_step4b = tk.Label(self.frame1, text="4b. Run PypeIt to trace slits", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_step4b = ctk.CTkLabel(self.frame1, text="Choose a side to make:")
        lbl_step4b.grid(row=rows[2], column=1, columnspan=3, sticky="w")


        #### select shoe side
        #lbl_shoe = tk.Label(self.frame1, text="Shoe:", fg=LABEL_COLOR, bg=BG_COLOR)
        #lbl_shoe.grid(row=rows[2], column=4, sticky="w")

        self.shoe2 = ctk.CTkRadioButton(self.frame1, text='r-side', variable=self.shoe, value='r', text_color="red")
        self.shoe2.grid(row=rows[2], column=4, sticky='e', padx=2)

        self.shoe1 = ctk.CTkRadioButton(self.frame1, text='b-side', variable=self.shoe, value='b', text_color="cyan")
        self.shoe1.grid(row=rows[2], column=5, columnspan=2, sticky='w', padx=2)

        self.btn_run_pypeit = ctk.CTkButton(self.frame1, width=60, text='Make', command=self.run_trace, state='disabled')
        self.btn_run_pypeit.grid(row=rows[2], column=7, sticky='e', padx=2, pady=1)

        #### step 4c save the AperMap
        #lbl_save = tk.Label(self.frame1, text="4c. Save the AperMap file")
        #lbl_save.grid(row=rows[1], column=1, columnspan=4, sticky="w")
        #self.lbl_slitnum = tk.Label(self.frame1, relief=tk.SUNKEN, text="N_slits = 000", fg=LABEL_COLOR)
        #self.lbl_slitnum.grid(row=rows[3], column=5, columnspan=2, sticky="e")

        # add a bottom divider line to this widget
        # lbl_line = tk.Label(self.frame1, text=" "*80, fg='gray', bg=BG_COLOR)
        # lbl_line.grid(row=rows[3], column=0, columnspan=8, sticky="w")
        self.divider = ctk.CTkFrame(self.frame1, height=2, fg_color='gray')
        self.divider.grid(row=rows[3], column=0, columnspan=8, sticky="ew", padx=5, pady=5)

    def create_widgets_add_slits(self, line_start, line_num):
        """ step 5 add bad/missing slits """
        start, lines = line_start, line_num
        rows = np.arange(start, start+lines)

        # lbl_step5 = ctk.CTkLabel(self.frame1, text="Add. 2:")
        # lbl_step5.grid(row=rows[0], column=0, sticky="w", padx=(5, 2), pady=1)
        # lbl_step5_desc = ctk.CTkLabel(self.frame1, text="(obsolete) Add missing slits to an AperMap")
        # lbl_step5_desc.grid(row=rows[0], column=1, columnspan=5, sticky="w")

        # self.lbl_file_apermap = ctk.CTkLabel(self.frame1, text="apx0000_0000", corner_radius=4, fg_color=("gray80", "gray20"))
        # self.lbl_file_apermap.grid(row=rows[0], column=5, columnspan=2, sticky="e", padx=2)

        self._step_label(rows[0], "Add. 2:",
                         "(obsolete) Add missing slits to an AperMap")
        self.lbl_file_apermap = self._file_badge(rows[0], col=5,
                                                 text="apx0000_0000", colspan=2)

        self.btn_load_apermap = ctk.CTkButton(self.frame1, width=60, text="Open", command=self.open_fits_apermap, state='normal')
        self.btn_load_apermap.grid(row=rows[0], column=7, sticky="e", padx=2, pady=1)

        #### Notes
        lbl_general_note = ctk.CTkLabel(self.frame1, text='Hint: Select along x-axis middle line; ESC to finish')
        lbl_general_note.grid(row=rows[1], column=1, columnspan=5, sticky='w')

        #### select y positions of bundle centers (LSB, STD and HR select 5, 6 and 8, respectively)
        lbl_bundles_note = ctk.CTkLabel(self.frame1, text='Opt. A: By bundle centers (pick 5, 6 or 8 points)')
        lbl_bundles_note.grid(row=rows[2], column=1, columnspan=5, sticky='w')

        self.btn_select_bundles = ctk.CTkButton(self.frame1, width=60, text="Select", command=self.pick_bundles, state='disabled')
        self.btn_select_bundles.grid(row=rows[2], column=6, sticky="e", padx=2, pady=1)

        self.btn_make_apermap_bundles = ctk.CTkButton(self.frame1, width=60, text='Run', command=self.make_file_apermap_fix2_v2, state='disabled')
        self.btn_make_apermap_bundles.grid(row=rows[2], column=7, sticky='e', padx=2, pady=1)

        #### select y positions of all missing slits
        lbl_slits_note = ctk.CTkLabel(self.frame1, text='Opt. B: By exact slit positions (pick N missing points)')
        lbl_slits_note.grid(row=rows[3], column=1, columnspan=5, sticky='w')

        self.btn_select_slits = ctk.CTkButton(self.frame1, width=60, text="Select", command=self.pick_slits, state='disabled')
        self.btn_select_slits.grid(row=rows[3], column=6, sticky="e", padx=2, pady=1)

        self.btn_make_apermap_slits = ctk.CTkButton(self.frame1, width=60, text='Run', command=self.make_file_apermap_slits_v2, state='disabled')
        self.btn_make_apermap_slits.grid(row=rows[3], column=7, sticky='e', padx=2, pady=1)

    def create_widgets_mono(self, line_start, line_num):
        """ step 6 make monochromatic apermap """
        start, lines = line_start, line_num
        rows = np.arange(start, start+lines)

        # lbl_step6 = ctk.CTkLabel(self.frame1, text="Add. 1:")
        # lbl_step6.grid(row=rows[0], column=0, sticky="w", padx=(5, 2), pady=1)
        # lbl_step6_desc = ctk.CTkLabel(self.frame1, text="(optional) Make monochromatic / narrow-band AperMap files")
        # lbl_step6_desc.grid(row=rows[0], column=1, columnspan=6, sticky="w")

        self._step_label(rows[0], "Add. 1:",
                         "(optional) Make monochromatic / narrow-band AperMap files")

        #### step 6a
        lbl_step6a = ctk.CTkLabel(self.frame1, text="a. Use Step 2 to determine the new edges (hint: load a SCI file)")
        lbl_step6a.grid(row=rows[1], column=1, columnspan=6, sticky="w")

        #### step 6b
        lbl_step6b = ctk.CTkLabel(self.frame1, text="b. Load the original AperMap files (i.e., ap*.fits)")
        lbl_step6b.grid(row=rows[2], column=1, columnspan=5, sticky="w")

        self.lbl_file_mono = ctk.CTkLabel(self.frame1, text="XXX_0000", corner_radius=4, fg_color=("gray80", "gray20"))
        self.lbl_file_mono.grid(row=rows[2], column=4, columnspan=3, sticky="e", padx=2)

        self.btn_load_mono = ctk.CTkButton(self.frame1, width=60, text="Open", command=self.open_fits_apermap2, state='normal')
        self.btn_load_mono.grid(row=rows[2], column=7, sticky="e", padx=2, pady=1)

        #### step 6c
        lbl_step6c = ctk.CTkLabel(self.frame1, text="c. Make new files")
        lbl_step6c.grid(row=rows[3], column=1, columnspan=2, sticky="w")

        self.ent_labelname_mono = ctk.CTkEntry(self.frame1, textvariable=self.txt_labelname_mono, state='normal')
        self.ent_labelname_mono.grid(row=rows[3], column=3, columnspan=4, sticky="ew", padx=2)

        self.btn_make_apermap_mono = ctk.CTkButton(self.frame1, width=60, text="Make", command=self.make_file_apermap_mono, state='disabled')
        self.btn_make_apermap_mono.grid(row=rows[3], column=7, sticky="e", padx=2, pady=1)

    def bind_widgets(self):
        """ event bindings """
        self.ent_folder.bind('<Return>', self.refresh_folder)
        self.ent_folder.bind('<KP_Enter>', self.refresh_folder) # unique for macOS

        self.ent_param_curve_A_b.bind('<Return>', lambda event: self.update_curve(event, 'b'))
        self.ent_param_curve_A_b.bind('<KP_Enter>', lambda event: self.update_curve(event, 'b')) # unique for macOS
        self.ent_param_curve_B_b.bind('<Return>', lambda event: self.update_curve(event, 'b'))
        self.ent_param_curve_B_b.bind('<KP_Enter>', lambda event: self.update_curve(event, 'b')) # unique for macOS
        self.ent_param_curve_C_b.bind('<Return>', lambda event: self.update_curve(event, 'b'))
        self.ent_param_curve_C_b.bind('<KP_Enter>', lambda event: self.update_curve(event, 'b')) # unique for macOS
        self.ent_param_curve_A_r.bind('<Return>', lambda event: self.update_curve(event, 'r'))
        self.ent_param_curve_A_r.bind('<KP_Enter>', lambda event: self.update_curve(event, 'r')) # unique for macOS
        self.ent_param_curve_B_r.bind('<Return>', lambda event: self.update_curve(event, 'r'))
        self.ent_param_curve_B_r.bind('<KP_Enter>', lambda event: self.update_curve(event, 'r')) # unique for macOS
        self.ent_param_curve_C_r.bind('<Return>', lambda event: self.update_curve(event, 'r'))
        self.ent_param_curve_C_r.bind('<KP_Enter>', lambda event: self.update_curve(event, 'r')) # unique for macOS

        self.ent_param_edges_X1_b.bind('<Return>', lambda event: self.update_edges(event, 'b'))
        self.ent_param_edges_X1_b.bind('<KP_Enter>', lambda event: self.update_edges(event, 'b')) # unique for macOS
        self.ent_param_edges_dX_b.bind('<Return>', lambda event: self.update_edges(event, 'b'))
        self.ent_param_edges_dX_b.bind('<KP_Enter>', lambda event: self.update_edges(event, 'b')) # unique for macOS
        self.ent_param_edges_X1_r.bind('<Return>', lambda event: self.update_edges(event, 'r'))
        self.ent_param_edges_X1_r.bind('<KP_Enter>', lambda event: self.update_edges(event, 'r')) # unique for macOS
        self.ent_param_edges_dX_r.bind('<Return>', lambda event: self.update_edges(event, 'r'))
        self.ent_param_edges_dX_r.bind('<KP_Enter>', lambda event: self.update_edges(event, 'r')) # unique for macOS
        self.ent_param_edges_offset.bind('<Return>', lambda event: self.update_edges(event, 'both'))
        self.ent_param_edges_offset.bind('<KP_Enter>', lambda event: self.update_edges(event, 'both')) # unique for macOS

        self.ent_folder_trace.bind('<Return>', self.refresh_folder_trace)
        self.ent_folder_trace.bind('<KP_Enter>', self.refresh_folder_trace) # unique for macOS

        #self.ent_smash_range.bind('<Return>', self.refresh_smash_range)
        #self.ent_smash_range.bind('<KP_Enter>', self.refresh_smash_range) # unique for macOS

        self.ent_labelname_mono.bind('<Return>', self.refresh_labelname_mono)
        self.ent_labelname_mono.bind('<KP_Enter>', self.refresh_labelname_mono) # unique for macOS

        #self.window.bind_all('<1>', lambda event: event.widget.focus_set())

    def refresh_smash_range(self, *args):
        self.window.focus_force()

    def refresh_labelname_mono(self, *args):
        self.window.focus_force()

    def update_curve(self, event, shoe, *args):
        """ update the curve parameters """
        self.refresh_param_curve(shoe)
        self.clear_image(shoe=shoe)
        self.plot_curve(shoe=shoe)
        self.window.focus_force()

    def update_edges(self, event, shoe, *args):
        """ update the edge parameters """
        self.refresh_param_edges(shoe)

        if self.state_edge_lock_r.get() == 0 and self.state_edge_lock_b.get() == 0:
            self.clear_image(shoe=shoe)
            self.plot_edges(shoe=shoe)
        else:
            self.clear_image(shoe='both')
            self.plot_edges(shoe='both')

        self.window.focus_force()

    def refresh_param_curve(self, shoe, *args):
        if shoe=='b':
            self.param_curve_b[0] = np.float32(self.ent_param_curve_A_b.get())
            self.param_curve_b[1] = np.float32(self.ent_param_curve_B_b.get())
            self.param_curve_b[2] = np.float32(self.ent_param_curve_C_b.get())
        elif shoe=='r':
            self.param_curve_r[0] = np.float32(self.ent_param_curve_A_r.get())
            self.param_curve_r[1] = np.float32(self.ent_param_curve_B_r.get())
            self.param_curve_r[2] = np.float32(self.ent_param_curve_C_r.get())

    def refresh_param_edges(self, shoe, *args):
        if self.state_edge_lock_r.get() == 0 and self.state_edge_lock_b.get() == 0:
            # offset is not locked
            if shoe=='b':
                self.param_edges_b[0] = np.float32(self.ent_param_edges_X1_b.get())
                self.param_edges_b[2] = np.float32(self.ent_param_edges_dX_b.get())
                self.param_edges_b[1] = self.param_edges_b[0] + self.param_edges_b[2]
            elif shoe=='r':
                self.param_edges_r[0] = np.float32(self.ent_param_edges_X1_r.get())
                self.param_edges_r[2] = np.float32(self.ent_param_edges_dX_r.get())
                self.param_edges_r[1] = self.param_edges_r[0] + self.param_edges_r[2]
            elif shoe=='both':
                self.param_edges_b[0] = np.float32(self.ent_param_edges_X1_r.get())-np.float32(self.ent_param_edges_offset.get())
                self.param_edges_b[1] = self.param_edges_b[0] + self.param_edges_b[2]

            self.param_edges_offset = self.param_edges_r[0]-self.param_edges_b[0]
        else:
            # offset is locked
            self.param_edges_offset = np.float32(self.ent_param_edges_offset.get())

            if self.state_edge_lock_r.get() == 1 and self.state_edge_lock_b.get() == 0:
                # can only change r-side
                self.param_edges_r[0] = np.float32(self.ent_param_edges_X1_r.get())
                self.param_edges_r[2] = np.float32(self.ent_param_edges_dX_r.get())
                self.param_edges_r[1] = self.param_edges_r[0] + self.param_edges_r[2]

                self.param_edges_b[0] = self.param_edges_r[0]-self.param_edges_offset
                self.param_edges_b[2] = self.param_edges_r[2]
                self.param_edges_b[1] = self.param_edges_b[0] + self.param_edges_b[2]

            elif self.state_edge_lock_r.get() == 0 and self.state_edge_lock_b.get() == 1:
                # can only change b-side
                self.param_edges_b[0] = np.float32(self.ent_param_edges_X1_b.get())
                self.param_edges_b[2] = np.float32(self.ent_param_edges_dX_b.get())
                self.param_edges_b[1] = self.param_edges_b[0] + self.param_edges_b[2]

                self.param_edges_r[0] = self.param_edges_b[0]+self.param_edges_offset
                self.param_edges_r[2] = self.param_edges_b[2]
                self.param_edges_r[1] = self.param_edges_r[0] + self.param_edges_r[2]
        
        self.renew_param_edges()

    def renew_param_edges(self):
        """ copy the edge parameters from backend values to GUI """
        self.txt_param_edges_X1_r.set("%.0f"%(self.param_edges_r[0]))
        self.txt_param_edges_X2_r.set("%.0f"%(self.param_edges_r[1]))
        self.txt_param_edges_dX_r.set("%.0f"%(self.param_edges_r[2]))
        self.txt_param_edges_X1_b.set("%.0f"%(self.param_edges_b[0]))
        self.txt_param_edges_X2_b.set("%.0f"%(self.param_edges_b[1]))
        self.txt_param_edges_dX_b.set("%.0f"%(self.param_edges_b[2]))
        self.txt_param_edges_offset.set("%.0f"%(self.param_edges_offset))

    def renew_param_curve(self):
        """ copy the curve parameters from backend values to GUI """
        self.txt_param_curve_A_r.set("%.3e"%(self.param_curve_r[0]))
        self.txt_param_curve_B_r.set("%.1f"%(self.param_curve_r[1]))
        self.txt_param_curve_C_r.set("%.1f"%(self.param_curve_r[2]))
        self.txt_param_curve_A_b.set("%.3e"%(self.param_curve_b[0]))
        self.txt_param_curve_B_b.set("%.1f"%(self.param_curve_b[1]))
        self.txt_param_curve_C_b.set("%.1f"%(self.param_curve_b[2]))

    def lock_edge(self, side):
        """ lock one side of the edges """
        if side == 'r':
            if self.state_edge_lock_r.get() == 1:
                self.ent_param_edges_X1_b.configure(state='disabled', text_color=self.DEFAULT_FG_DISABLED)
                self.ent_param_edges_dX_b.configure(state='disabled', text_color=self.DEFAULT_FG_DISABLED)
                self.cbtn_edge_lock_b.configure(state='disabled')
            else:
                self.ent_param_edges_X1_b.configure(state='normal', text_color=self.DEFAULT_FG)
                self.ent_param_edges_dX_b.configure(state='normal', text_color=self.DEFAULT_FG)
                self.cbtn_edge_lock_b.configure(state='normal')

        elif side == 'b':
            if self.state_edge_lock_b.get() == 1:
                self.ent_param_edges_X1_r.configure(state='disabled', text_color=self.DEFAULT_FG_DISABLED)
                self.ent_param_edges_dX_r.configure(state='disabled', text_color=self.DEFAULT_FG_DISABLED)
                self.cbtn_edge_lock_r.configure(state='disabled')
            else:
                self.ent_param_edges_X1_r.configure(state='normal', text_color=self.DEFAULT_FG)
                self.ent_param_edges_dX_r.configure(state='normal', text_color=self.DEFAULT_FG)
                self.cbtn_edge_lock_r.configure(state='normal')

    def get_curve_params(self, shoe):
        if shoe=='b':
            return np.array([self.param_curve_b[0], self.param_curve_b[1], self.param_curve_b[2], self.param_edges_b[0], self.param_edges_b[2]])
        elif shoe=='r':
            return np.array([self.param_curve_r[0], self.param_curve_r[1], self.param_curve_r[2], self.param_edges_r[0], self.param_edges_r[2]])

    def run_trace(self):
        shoe = self.shoe.get()
        dirname = self.ent_folder_trace.get()
        filename = shoe + self.lbl_file_pypeit.cget("text")
        coef_temp = self.get_curve_params(shoe)

        # load the trace file and reshape according to the curvature
        path_traceFile = os.path.join(dirname, filename+'.fits')
        data_trace, ifu_type_trace, bin_y_trace = load_trace(path_traceFile)
        data_reshaped = reshape_trace_by_curvature(data_trace, coef_temp)

        # trace the resahped data and create an apermap
        trace_array, trace_coefs, N_sl, aper_half_width = do_trace_v3(
            data_reshaped, coef_temp,                          
            shoe, ifu_type_trace, bin_y_trace, verbose=True)
        map_ap, y_middle = create_apermap(data_trace, coef_temp, trace_coefs, aper_half_width)
        #print(len(y_middle), y_middle)
        #print(np.diff(y_middle))

        #### check the number of slits
        self.ifu_type = self.get_ifu_type(N_sl)
        N_ap = np.int32(self.ifu_type.Ntotal/2)
        if N_ap>N_sl:
            print('!!! Warning: Missing %d fiber(s). Expected to find %d fibers !!!'%(N_ap-N_sl, N_ap))
        elif N_ap<N_sl:
            print('!!! Warning: Found %d more fiber(s). Expected to find %d fibers. !!!'%(N_sl-N_ap, N_ap))
        else:
            print('Found all %d fibers.'%N_ap)

        # find the maximum number of pixels in all slits
        num_ap = np.zeros(N_ap, dtype=np.int32)
        for i_ap in range(N_ap):
            num_ap[i_ap] = np.sum(map_ap==i_ap+1)
        num_max = np.max(num_ap)

        #### save AperMap
        #### the following header params may require modifying
        hdu_map = fits.PrimaryHDU(map_ap)
        hdr_map = hdu_map.header
        hdr_map['IFUTYPE'] = (self.ifu_type.label, 'type of IFU')
        #hdr_map.set('IFUTYPE', IFU_type, 'type of IFU')
        hdr_map['NIFU1'] = (self.ifu_type.Nx, 'number of IFU columns')
        hdr_map['NIFU2'] = (self.ifu_type.Ny, 'number of IFU rows')
        hdr_map['NSLITS'] = (N_sl, 'number of slits')
        hdr_map['NMAX'] = (num_max, 'maximum number of pixels among all apertures')
        hdr_map['BINNING'] = ('1x1', 'binning')
        #hdu_map = fits.PrimaryHDU(map_ap, header=hdr_map)

        ## record curve params in header
        hdr_map['CURVE_A'] = (coef_temp[0], 'curve parameter A')
        hdr_map['CURVE_B'] = (coef_temp[1], 'curve parameter B')
        hdr_map['CURVE_C'] = (coef_temp[2], 'curve parameter C')
        hdr_map['CURVE_X1'] = (coef_temp[3], 'starting X position of edges')
        hdr_map['CURVE_DX'] = (coef_temp[4], 'length of edges along X axis')

        dir_aperMap = os.path.join(self.ent_folder_trace.get(),'aperMap')
        if not os.path.exists(dir_aperMap):
            os.mkdir(dir_aperMap)

        today_temp = datetime.today().strftime("%y%m%d")
        file_aperMap = 'ap%s_%s_%s_%s_%s_%s_%s_%s.fits'%(
            shoe, 
            self.ifu_type.label, 
            self.HDR_CONFIG, 
            self.lbl_file_pypeit.cget("text")[0:4],
            self.HDR_BINNING,
            self.HDR_SLIDE,
            self.HDR_SLITNAME, 
            today_temp)

        path_aperMap = os.path.join(dir_aperMap, file_aperMap)
        hdu_map.writeto(path_aperMap,overwrite=True)

        #### save slits file
        dir_slits = os.path.join(dir_aperMap, 'slits')
        if not os.path.exists(dir_slits):
            os.mkdir(dir_slits)
        file_slits = filename.split('_')[0]+'_slits.txt'
        path_slits = os.path.join(dir_slits, file_slits)
        np.savetxt(path_slits, y_middle, fmt='%d', delimiter=' ', header='# y_pos (x=middle)', comments='')

        #### save the trace coefs
        dir_coefs = os.path.join(dir_aperMap, 'trace_coefs')
        if not os.path.exists(dir_coefs):
            os.mkdir(dir_coefs)
        file_coefs = filename.split('_')[0]+'_coefs.txt'
        path_coefs = os.path.join(dir_coefs, file_coefs)
        np.savetxt(path_coefs, trace_coefs, fmt='%.6e', delimiter=',', header='# a b c', comments='# aper_half_width = %d\n'%aper_half_width)

        #### show info
        info_temp = '%s-side AperMap file made!\n\n Saved to %s'%(shoe, path_aperMap)
        self.popup_showinfo('AperMap', info_temp)

        #### show apermap
        fname = filename.split('_')[0]+'_apermap'
        title = '%s (N_sl=%d)'%(fname,N_sl)
        self.clear_image(shoe=shoe)
        self.update_image_single(map_ap, title, shoe=shoe, uniform=True)
        print('++++\n++++ %s-side AperMap file made! \n++++ Saved to %s\n++++\n'%(shoe, path_aperMap))

        self.window.focus_force()

    def make_file_pypeit(self):
        dirname = self.ent_folder_trace.get()
        filename = self.lbl_file_pypeit.cget("text")
        smash_range = self.ent_smash_range.get()
        write_pypeit_file(dirname, 'b'+filename, self.pca.get(), smash_range)
        write_pypeit_file(dirname, 'r'+filename, self.pca.get(), smash_range)

        #### show the smash range
        self.clear_image()
        
        # b side
        xx = np.float32(smash_range.split(','))*len(self.data_full[0])
        self.ax.axvline(xx[0], c='g', ls='--')
        self.ax.axvline(xx[1], c='g', ls='--')

        # r side
        xx = np.float32(smash_range.split(','))*len(self.data_full2[0])
        self.ax2.axvline(xx[0], c='g', ls='--')
        self.ax2.axvline(xx[1], c='g', ls='--')

        self.update_image() 

        self.btn_run_pypeit.configure(state='normal')
        #self.popup_showinfo('', 'Success to make a PypeIt file:\n %s'%os.path.join(dirname, 'pypeit_file', filename+'.pypeit'))

    def run_pypeit(self):
        shoe = self.shoe.get()
        dirname = self.ent_folder_trace.get()
        filename = shoe + self.lbl_file_pypeit.cget("text")

        #### run PypeIt
        dir_pypeitFile = os.path.join(dirname, 'pypeit_file')
        path_pypeitFile = os.path.join(dir_pypeitFile, filename+'.pypeit')
        os.system('pypeit_trace_edges -f %s -s magellan_m2fs_blue'%path_pypeitFile)

        #### handle the PypeIt outputs
        path_MasterEdges_default = os.path.join(dir_pypeitFile, 'Masters/MasterEdges_A_1_DET01.fits.gz')
        path_MasterSlits_default = os.path.join(dir_pypeitFile, 'Masters/MasterSlits_A_1_DET01.fits.gz')
        path_MasterEdges = os.path.join(dir_pypeitFile, 'Masters/MasterEdges_%s.fits.gz'%filename)
        self.path_MasterSlits = os.path.join(dir_pypeitFile, 'Masters/MasterSlits_%s.fits.gz'%filename)

        os.system('mv '+path_MasterEdges_default+' '+path_MasterEdges)
        os.system('mv '+path_MasterSlits_default+' '+self.path_MasterSlits)
        os.system('rm %s'%os.path.join(dir_pypeitFile,filename+'.calib'))


        N_slits = self.check_file_MasterSlits(message=False)       
        #self.lbl_slitnum['text'] = 'N_slits = %d'%N_slits
        self.make_file_apermap()

        self.window.focus_force()

    def check_file_MasterSlits(self, message=True):
        #hdul = cached_fits_open(self.path_MasterSlits)
        hdul = fits.open(self.path_MasterSlits)
        hdr = hdul[1].header
        N_slits = np.int32(hdr['NSLITS'])
        self.ifu_type = self.get_ifu_type(N_slits)

        #### show messages
        if message:
            if N_slits==self.ifu_type.Ntotal/2:
                info_temp = 'PypeIt found all %d slits for %s'%(N_slits,self.ifu_type.label)
            else:
                info_temp = 'PypeIt found %d slits, close to %d (%s)'%(N_slits,self.ifu_type.Ntotal/2,self.ifu_type.label)
            self.popup_showinfo('PypeIt', info_temp)
            print('\n++++\n++++ %s\n++++\n'%(info_temp))

        return N_slits

    def get_ifu_type(self, Nslits):
        Nslits_IFU = np.array([self.LSB.Ntotal/2, self.STD.Ntotal/2, self.HR.Ntotal/2])
        diff_Nslits = np.abs(Nslits_IFU-Nslits)
        idx_IFU = np.where(diff_Nslits==np.min(diff_Nslits))[0][0]
        
        if idx_IFU==0:
            return self.LSB
        elif idx_IFU==1:
            return self.STD
        elif idx_IFU==2:
            return self.HR
        else:
            return self.UNKNOWN

    def make_file_apermap_slits_v2(self):
        #### get the correct number of slits
        N_ap = np.int32(self.ifu_type.Ntotal/2)
        shoe = self.lbl_file_apermap.cget("text")[2]

        #### load the slits coefs
        dirname_coefs = os.path.join(self.folder_trace, 'trace_coefs')
        filename_coefs = self.lbl_file_apermap.cget("text").split('_')[0][2:]+'_coefs.txt'
        path_coefs = os.path.join(dirname_coefs, filename_coefs)
        trace_coefs = np.loadtxt(path_coefs, delimiter=',')
        N_sl = len(trace_coefs)

        # load path_coefs the first line and get aper_half_width
        with open(path_coefs, 'r') as f:
            line = f.readline()
            aper_half_width = int(line.split('# aper_half_width = ')[1])

        x_middle = np.int32(len(self.data_full[0])/2)
        y_middle = np.zeros(N_sl, dtype=np.int32)
        for i_sl in range(N_sl):
            y_middle[i_sl] = np.round( poly.polyval(x_middle, trace_coefs[i_sl]) ).astype(np.int32)

        #### load missing slits file
        dirname_slits = os.path.join(self.folder_trace, 'slits_file')
        filename_slits = self.lbl_file_apermap.cget("text").split('_')[0][3:]+'_slits_'+shoe+'.txt'
        path_slits = os.path.join(dirname_slits, filename_slits)
        if os.path.isfile(path_slits):
            y_missing = np.int32(readFloat_space(path_slits, 0))
        else:
            y_missing = np.array([])

        #### add missing slits and make new AperMap
        y_middle_new = np.append(y_middle, y_missing)
        y_middle_new = np.sort(y_middle_new)
        N_new = len(y_middle_new)

        print('To add %d fibers'%N_new)
        if N_new>N_ap:
            print('!!! Warning: More bad fibers are added. !!!')
        elif N_new<N_ap:
            print('!!! Warning: Less bad fibers are added. !!!')
        else:
            print('All %d fibers are found.'%N_ap)

        #### make a new AperMap
        map_ap = np.zeros((len(self.data_full),len(self.data_full[0])), dtype=np.int32)
        x_trace = np.arange(len(self.data_full[0]))
        for i_ap in range(N_new):
            ap_num = i_ap+1
            y_middle_temp = y_middle_new[i_ap]

            temp_index = np.where(y_middle==y_middle_temp)[0]
            if len(temp_index)==1:
                i_temp = temp_index[0]
                y_temp = np.round( poly.polyval(x_trace, trace_coefs[i_temp]) ).astype(np.int32)
                for j in range(len(x_trace)):
                    map_ap[y_temp[j]-aper_half_width:y_temp[j]+aper_half_width, x_trace[j]] = np.int32(ap_num)
            else:
                print('!!! Now adding the %d-th fiber. !!!'%ap_num)

                #### insert a missing slit using the nearest slit
                dist_temp = np.abs(y_middle-y_middle_temp)
                i_temp = np.where(dist_temp==np.min(dist_temp))[0][0]
                shift_temp = y_middle_temp-y_middle[i_temp]
                print('Fake', ap_num, 'acording to', i_temp+1)

                y_temp = np.round( poly.polyval(x_trace, trace_coefs[i_temp]) ).astype(np.int32)
                for j in range(len(x_trace)):
                    map_ap[y_temp[j]+shift_temp-aper_half_width:y_temp[j]+shift_temp+aper_half_width, x_trace[j]] = np.int32(ap_num)

        #### cut data
        map_ap = self.cut_data_by_edges(map_ap, shoe)

        #### find the maximum number of pixels in all slits
        num_ap = np.zeros(N_ap, dtype=np.int32)
        for i_ap in range(N_ap):
            num_ap[i_ap] = np.sum(map_ap==i_ap+1)
        num_max = np.max(num_ap)

        #### save AperMap
        #### the following header params may require modifying
        hdu_map = fits.PrimaryHDU(map_ap)
        hdr_map = hdu_map.header
        hdr_map['IFUTYPE'] = (self.ifu_type.label, 'type of IFU')
        #hdr_map.set('IFUTYPE', IFU_type, 'type of IFU')
        hdr_map['NIFU1'] = (self.ifu_type.Nx, 'number of IFU columns')
        hdr_map['NIFU2'] = (self.ifu_type.Ny, 'number of IFU rows')
        hdr_map['NSLITS'] = (N_new, 'number of slits')
        hdr_map['NMAX'] = (num_max, 'maximum number of pixels among all apertures')
        hdr_map['BINNING'] = ('1x1', 'binning')
        #hdu_map = fits.PrimaryHDU(map_ap, header=hdr_map)

        dir_aperMap = self.ent_folder_trace.get()
        if not os.path.exists(dir_aperMap):
            os.mkdir(dir_aperMap)
        
        today_temp = datetime.today().strftime('%y%m%d')
        file_aperMap = 'ap%s_%s_%s_%s_%s_%s_%s_%s.fits'%(
            self.lbl_file_apermap.cget("text")[2], 
            self.ifu_type.label, 
            self.HDR_CONFIG, 
            self.lbl_file_apermap.cget("text")[3:7],
            self.HDR_BINNING,
            self.HDR_SLIDE,
            self.HDR_SLITNAME, 
            today_temp)

        path_aperMap = os.path.join(dir_aperMap, file_aperMap)
        hdu_map.writeto(path_aperMap,overwrite=True)

        #self.btn_select_slits['state'] = 'disabled'
        #self.btn_select_slits['state'] = 'disabled'

        ####
        self.clear_image()
        self.file_current = '%s (Nslits=%d)'%(self.lbl_file_apermap.cget("text"), N_new)
        self.update_image_single(map_ap, self.file_current, shoe='b', uniform=True)

        info_temp = 'Saved as %s'%path_aperMap
        self.popup_showinfo('aperMap', info_temp)
        print('\n++++\n++++ %s\n++++\n'%(info_temp))

        self.window.focus_force()
    
    # def make_file_apermap_slits(self):
    #     #### load MasterSlits file
    #     N_ap = np.int32(self.ifu_type.Ntotal/2)
    
    #     basename = os.path.basename(self.path_MasterSlits)
    #     shoe = basename.split('_')[1][0]
    
    #     #hdul = cached_fits_open(self.path_MasterSlits)
    #     hdul = fits.open(self.path_MasterSlits)
    #     hdr = hdul[1].header
    #     data = hdul[1].data
    
    #     N_sl  = np.int32(hdr['NSLITS'])
    #     nspec = np.int32(hdr['NSPEC']) ### binning?
    #     nspat = np.int32(hdr['NSPAT'])
    #     map_ap = np.zeros((nspat,nspec), dtype=np.int32)
    
    #     print('nspat, nspec=',nspat,nspec)
    #     print('Note: %d out of %d fibers are found by pypeit_trace_edges.'%(N_sl,N_ap))
    
    #     #### load missing slits file
    #     dirname_slits = os.path.join(self.folder_trace, 'slits_file')
    #     filename_slits = self.filename_trace.split('_')[2]+'_slits.txt'
    #     path_slits = os.path.join(dirname_slits, filename_slits)
    #     if os.path.isfile(path_slits):
    #         spat_id_missing = np.int32(readFloat_space(path_slits, 0))
    #     else:
    #         spat_id_missing = np.array([])
    
    #     #### add missing slits and make new AperMap
    #     print('Note: %d fiber(s) are added manually.'%(len(spat_id_missing)))
    #     spat_id_raw = data['spat_id']
    #     spat_id_new = np.append(spat_id_raw, spat_id_missing)
    #     spat_id_new = np.sort(spat_id_new)
    #     N_new = len(spat_id_new)
    
    #     if N_new>N_ap:
    #         print('!!! Warning: More bad fibers are added. !!!')
    #     elif N_new<N_ap:
    #         print('!!! Warning: Less bad fibers are added. !!!')
    
    #     for i_ap in range(N_new):
    #         ap_num = i_ap+1
    #         spat_id_temp = spat_id_new[i_ap]
    
    #         temp_index = np.where(spat_id_raw==spat_id_temp)[0]
    #         if len(temp_index)==1:
    #             i_temp = temp_index[0]
    #             for x_temp in range(nspec):
    #                 ap_y1 = np.int32(np.round(data[i_temp]['left_init'][x_temp]-1))
    #                 ap_y2 = np.int32(np.round(data[i_temp]['right_init'][x_temp]))
    #                 map_ap[ap_y1:ap_y2, x_temp] = np.int32(ap_num)
    #         else:
    #             #map_ap[spat_id_temp-2:spat_id_temp+1, int(nspec/2)-2:int(nspec/2)+1] = np.int32(ap_num)
    #             #print("!!!!!!", spat_id_temp-2, spat_id_temp+1, int(nspec/2)-2, int(nspec/2)+1)
    
    #             #### insert a missing slit using the nearest slit
    #             dist_temp = np.abs(spat_id_raw-spat_id_temp)
    #             i_temp = np.where(dist_temp==np.min(dist_temp))[0][0]
    #             shift_temp = spat_id_temp-spat_id_raw[i_temp]
    #             print(ap_num, i_temp)
    
    #             for x_temp in range(nspec):
    #                 ap_y1 = np.int32(np.round(data[i_temp]['left_init'][x_temp]-1))
    #                 ap_y2 = np.int32(np.round(data[i_temp]['right_init'][x_temp]))
    #                 map_ap[ap_y1+shift_temp:ap_y2+shift_temp, x_temp] = np.int32(ap_num)
    
    #             ##### insert a missing slit using either one slit before or after
    #             #d1_spat_id = spat_id_temp - spat_id_new[i_ap-1]
    #             #d2_spat_id = spat_id_new[i_ap+1] - spat_id_temp
    #             #print(d1_spat_id, d2_spat_id)
    #             #if d1_spat_id>d2_spat_id:
    #             #    temp_index = np.where(spat_id_raw==spat_id_new[i_ap+1])[0]
    #             #    if len(temp_index)==1:
    #             #        i_temp = temp_index[0]
    #             #        for x_temp in range(nspec):
    #             #            ap_y1 = np.int32(np.round(data[i_temp]['left_init'][x_temp]-1))
    #             #            ap_y2 = np.int32(np.round(data[i_temp]['right_init'][x_temp]))
    #             #            map_ap[ap_y1-d2_spat_id:ap_y2-d2_spat_id, x_temp] = np.int32(ap_num)
    #             #else:
    #             #    temp_index = np.where(spat_id_raw==spat_id_new[i_ap-1])[0]
    #             #    if len(temp_index)==1:
    #             #        i_temp = temp_index[0]
    #             #        for x_temp in range(nspec):
    #             #            ap_y1 = np.int32(np.round(data[i_temp]['left_init'][x_temp]-1))
    #             #            ap_y2 = np.int32(np.round(data[i_temp]['right_init'][x_temp]))
    #             #            map_ap[ap_y1+d1_spat_id:ap_y2+d1_spat_id, x_temp] = np.int32(ap_num)
    #             #            map_ap[ap_y1:ap_y2, x_temp] = np.int32(ap_num-1)
    
    #     #### cut data
    #     map_ap = self.cut_data_by_edges(map_ap, shoe)
    
    #     #### find the maximum number of pixels in all slits
    #     num_ap = np.zeros(N_ap, dtype=np.int32)
    #     for i_ap in range(N_ap):
    #         num_ap[i_ap] = np.sum(map_ap==i_ap+1)
    #     num_max = np.max(num_ap)
    
    #     #plt.imshow(map_ap, origin='lower')
    
    #     #### save AperMap
    #     #### the following header params may require modifying
    #     hdu_map = fits.PrimaryHDU(map_ap)
    #     hdr_map = hdu_map.header
    #     hdr_map['IFUTYPE'] = (self.ifu_type.label, 'type of IFU')
    #     #hdr_map.set('IFUTYPE', IFU_type, 'type of IFU')
    #     hdr_map['NIFU1'] = (self.ifu_type.Nx, 'number of IFU columns')
    #     hdr_map['NIFU2'] = (self.ifu_type.Ny, 'number of IFU rows')
    #     hdr_map['NSLITS'] = (N_new, 'number of slits')
    #     hdr_map['NMAX'] = (num_max, 'maximum number of pixels among all apertures')
    #     hdr_map['BINNING'] = ('1x1', 'binning')
    #     #hdu_map = fits.PrimaryHDU(map_ap, header=hdr_map)
    
    #     dir_aperMap = self.ent_folder_trace.get()
    #     if not os.path.exists(dir_aperMap):
    #         os.mkdir(dir_aperMap)
    
    #     today_temp = datetime.today().strftime('%y%m%d')
    #     file_aperMap = 'ap%s_%s_%s_%s.fits'%(
    #         self.lbl_file_apermap['text'][2], 
    #         self.ifu_type.label, 
    #         self.lbl_file_apermap['text'][3:7],
    #         today_temp)
    
    #     path_aperMap = os.path.join(dir_aperMap, file_aperMap)
    #     hdu_map.writeto(path_aperMap,overwrite=True)
    
    #     #self.btn_select_slits['state'] = 'disabled'
    #     #self.btn_select_slits['state'] = 'disabled'
    
    #     ####
    #     self.clear_image()
    #     self.file_current = '%s (Nslits=%d)'%(self.lbl_file_apermap['text'], N_new)
    #     self.update_image_single(map_ap, self.file_current, shoe='b', uniform=True)
    
    #     info_temp = 'Saved as %s'%path_aperMap
    #     self.popup_showinfo('aperMap', info_temp)
    #     print('\n++++\n++++ %s\n++++\n'%(info_temp))
    
    #     self.window.focus_force()
    
    def make_file_apermap_fix2_v2(self):
        '''
        For LSB, STD and HR, pick 5, 6 and 8 bundle centers, respectively
        '''
        #### get file names and shoe
        N_ap = np.int32(self.ifu_type.Ntotal/2)
        shoe = self.lbl_file_apermap.cget("text")[2]

        #### load the slits coefs
        dirname_coefs = os.path.join(self.folder_trace, 'trace_coefs')
        filename_coefs = self.lbl_file_apermap.cget("text").split('_')[0][2:]+'_coefs.txt'
        path_coefs = os.path.join(dirname_coefs, filename_coefs)
        trace_coefs = np.loadtxt(path_coefs, delimiter=',')
        N_sl = len(trace_coefs)

        x_middle = np.int32(len(self.data_full[0])/2)
        y_middle = np.zeros(N_sl, dtype=np.int32)
        for i_sl in range(N_sl):
            y_middle[i_sl] = np.round( poly.polyval(x_middle, trace_coefs[i_sl]) ).astype(np.int32)

        #### load bundle centers
        dirname_slits = os.path.join(self.folder_trace, 'slits_file')
        filename_bundles = self.filename_trace.split('_')[2]+'_bundles_'+shoe+'.txt'
        path_bundles = os.path.join(dirname_slits, filename_bundles)
        if os.path.isfile(path_bundles):
            pts_pick = np.int32(readFloat_space(path_bundles, 0))
        else:
            pts_pick = np.array([])
        n_pick = len(pts_pick)

        ####
        if N_ap>N_sl:
            print('!!! Warning: Missing %d fiber(s). !!!'%(N_ap-N_sl))
        elif N_ap<N_sl:
            print('!!! Warning: Found %d more fiber(s) than expected. !!!'%(N_sl-N_ap))

        print("Loaded in %d bundle centers, %s has %d bundle centers"%(n_pick, self.ifu_type.label, self.ifu_type.Ny/4))
        if len(pts_pick)!=self.ifu_type.Ny/4:
            return 0

        #### add missing slits
        y_middle_add = np.array([], dtype=np.int32)

        pts_diff = np.diff(pts_pick)
        pts_gap = pts_pick[0:-1] + pts_diff/2.
        pts_all = np.append(pts_pick, pts_gap)
        pts_all = np.append(pts_all, np.array([0, len(self.data_full[0])]))
        pts_all = np.sort(pts_all)
        for i in range(n_pick*2):
            mask_id_temp = np.logical_and(y_middle>pts_all[i], y_middle<=pts_all[i+1])
            y_middle_temp = y_middle[mask_id_temp]
            y_diff_med_temp = np.median( np.diff(y_middle_temp) )

            y_middle_temp = np.append(pts_all[i], y_middle_temp)
            y_middle_temp = np.append(y_middle_temp, pts_all[i+1])
            y_diff_temp = np.diff(y_middle_temp)

            print('Working on %d/%d to add %d fiber(s)'%(i+1, n_pick*2, self.ifu_type.Nx-len(y_diff_temp)+1))
            print('==== diff_med', y_diff_med_temp)
            while len(y_diff_temp)<=self.ifu_type.Nx:
                mask_diff_temp = y_diff_temp>y_diff_med_temp+np.sqrt(y_diff_med_temp)
                if np.sum(mask_diff_temp)>0:
                    if i%2==0:
                        idx_temp = np.where(mask_diff_temp)[0][-1]
                        y_middle_add_temp = y_middle_temp[idx_temp] + y_diff_med_temp
                    else:
                        idx_temp = np.where(mask_diff_temp)[0][0]
                        y_middle_add_temp = y_middle_temp[idx_temp+1] - y_diff_med_temp
                    y_middle_add = np.sort( np.append(y_middle_add, y_middle_add_temp) )
                    y_middle_temp = np.sort( np.append(y_middle_temp, y_middle_add_temp) )
                    y_diff_temp = np.diff(y_middle_temp)
                    print(len(y_middle_add), y_middle_add_temp)

            y_middle = np.sort( np.append(y_middle[~mask_id_temp], y_middle_temp) )

        ####
        print("Auto-fix found %d fiber(s) to add"%len(y_middle_add))

        #### save new y_middle positions into a file
        dirname = os.path.join(self.folder_trace, 'slits_file')
        filename = self.filename_trace.split('_')[2]+'_slits_'+shoe+'.txt'

        if not os.path.exists(dirname):
            os.mkdir(dirname)

        file = open(os.path.join(dirname, filename), 'w')
        for temp in y_middle_add:
            file.write("%d\n"%temp)
        file.close()

        self.make_file_apermap_slits_v2()

    def make_file_apermap_fix2(self):
        '''
        For LSB, STD and HR, pick 5, 6 and 8 bundle centers, respectively
        '''
        #### get file names and shoe
        N_ap = np.int32(self.ifu_type.Ntotal/2)
        basename = os.path.basename(self.path_MasterSlits)
        fname = basename.split('_')[1]+'_apermap'
        shoe = fname[0]
        print(shoe, fname)

        #### load bundle centers
        dirname_slits = os.path.join(self.folder_trace, 'slits_file')
        filename_bundles = self.filename_trace.split('_')[2]+'_bundles.txt'
        path_bundles = os.path.join(dirname_slits, filename_bundles)
        if os.path.isfile(path_bundles):
            pts_pick = np.int32(readFloat_space(path_bundles, 0))
        else:
            pts_pick = np.array([])       
        n_pick = len(pts_pick)

        print("Loaded in %d bundle centers, %s has %d bundle centers"%(n_pick, self.ifu_type.label, self.ifu_type.Ny/4))
        if len(pts_pick)!=self.ifu_type.Ny/4:
            return 0

        #### read MasterSlits
        #hdul = cached_fits_open(self.path_MasterSlits)
        hdul = fits.open(self.path_MasterSlits)
        hdr = hdul[1].header
        data = hdul[1].data

        N_sl  = np.int32(hdr['NSLITS'])
        nspec = np.int32(hdr['NSPEC']) ### binning?
        nspat = np.int32(hdr['NSPAT'])
        map_ap = np.zeros((nspat,nspec), dtype=np.int32)
        print('nspat, nspec=',nspat,nspec)
        print('Note: %d out of %d fibers are found by pypeit_trace_edges.'%(N_sl,N_ap))

        ####
        if N_ap>N_sl:
            print('!!! Warning: Missing %d fiber(s). !!!'%(N_ap-N_sl))
        elif N_ap<N_sl:
            print('!!! Warning: Found %d more fiber(s) than expected. !!!'%(N_sl-N_ap))
        
        ap_id = data['spat_id']
        ap_diff = np.diff(ap_id)
        Nx = self.ifu_type.Nx
        Ny = self.ifu_type.Ny
        ap_id_add = np.array([], dtype=np.int32)

        #### add missing fibers
        thresh_diff = 1.5
        #ap_diff_med = np.median(ap_diff)

        pts_diff = np.diff(pts_pick)
        pts_gap = pts_pick[0:-1] + pts_diff/2.
        pts_all = np.append(pts_pick, pts_gap)
        pts_all = np.append(pts_all, np.array([0, nspat]))
        pts_all = np.sort(pts_all)
        for i in range(n_pick*2):
            mask_id_temp = np.logical_and(ap_id>pts_all[i], ap_id<=pts_all[i+1])
            ap_id_temp = ap_id[mask_id_temp]
            ap_diff_med_temp = np.median(np.diff(ap_id_temp))

            ap_id_temp = np.append(pts_all[i], ap_id_temp)
            ap_id_temp = np.append(ap_id_temp, pts_all[i+1])
            ap_diff_temp = np.diff(ap_id_temp)

            print('Working on %d/%d to add %d fiber(s)'%(i+1, n_pick*2, self.ifu_type.Nx-len(ap_diff_temp)+1))
            while len(ap_diff_temp)<=self.ifu_type.Nx:
                mask_diff_temp = ap_diff_temp>thresh_diff*ap_diff_med_temp
                if np.sum(mask_diff_temp)>0:
                    if i%2==0:
                        idx_temp = np.where(mask_diff_temp)[0][-1]
                        ap_id_add_temp = ap_id_temp[idx_temp+1] - ap_diff_med_temp
                    else:
                        idx_temp = np.where(mask_diff_temp)[0][0]
                        ap_id_add_temp = ap_id_temp[idx_temp] + ap_diff_med_temp
                    ap_id_add = np.sort( np.append(ap_id_add, ap_id_add_temp) )
                    ap_id_temp = np.sort( np.append(ap_id_temp, ap_id_add_temp) )
                    ap_diff_temp = np.diff(ap_id_temp)
                    print(len(ap_id_add), ap_id_add_temp)

            ap_id = np.sort( np.append(ap_id[~mask_id_temp], ap_id_temp) )

        #### 
        print("Auto-fix found %d fiber(s) to add"%len(ap_id_add))

        #### save new spat_id positions into a file
        dirname = os.path.join(self.folder_trace, 'slits_file')
        filename = self.filename_trace.split('_')[2]+'_slits.txt'

        if not os.path.exists(dirname):
            os.mkdir(dirname)

        file = open(os.path.join(dirname, filename), 'w')
        for temp in ap_id_add:
            file.write("%d\n"%temp)
        file.close()

        self.make_file_apermap_slits()

    def make_file_apermap_fix(self):
        N_ap = np.int32(self.ifu_type.Ntotal/2)
        basename = os.path.basename(self.path_MasterSlits)
        fname = basename.split('_')[1]+'_apermap'
        shoe = fname[0]
        print(shoe, fname)

        #hdul = cached_fits_open(self.path_MasterSlits)
        hdul = fits.open(self.path_MasterSlits)
        hdr = hdul[1].header
        data = hdul[1].data

        N_sl  = np.int32(hdr['NSLITS'])
        nspec = np.int32(hdr['NSPEC']) ### binning?
        nspat = np.int32(hdr['NSPAT'])
        map_ap = np.zeros((nspat,nspec), dtype=np.int32)
        print('nspat, nspec=',nspat,nspec)
        print('Note: %d out of %d fibers are found by pypeit_trace_edges.'%(N_sl,N_ap))

        ####
        if N_ap>N_sl:
            print('!!! Warning: Missing %d fiber(s). !!!'%(N_ap-N_sl))
        elif N_ap<N_sl:
            print('!!! Warning: Found %d more fiber(s) than expected. !!!'%(N_sl-N_ap))
        
        ap_id = data['spat_id']
        ap_diff = np.diff(ap_id)
        Nx = self.ifu_type.Nx
        Ny = self.ifu_type.Ny
        ap_id_add = np.array([], dtype=np.int32)

        ##### manually add starting fibers
        #N_start = 3
        #diff_med = np.median(ap_diff)
        #for i in range(N_start):
        #    ap_id_add = np.append(ap_id_add, ap_id[0]-i*diff_med)
        #ap_id_add = np.sort(ap_id_add)
        #ap_id = np.append(ap_id_add, ap_id)
        #ap_diff = np.diff(ap_id)
        #print(ap_id_add)
        #diff_med = np.median(ap_diff)
        #ap_id_add = np.append(ap_id_add, ap_id[46]-diff_med)
        #ap_id = np.insert(ap_id, 46, ap_id_add[0])
        #ap_diff = np.diff(ap_id)

        thresh_diff = 1.7
        iy = 0
        while iy <= Ny/2:
            ibundle = iy*Nx
            diff_temp = np.median(ap_diff[ibundle:ibundle+Nx-1])
            mask_temp = ap_diff[ibundle:ibundle+Nx-1]>thresh_diff*diff_temp
            if np.sum(mask_temp)>0:
                idx_temp = np.where(mask_temp)[0]
                ap_id_temp =  ap_id[ibundle+idx_temp]+diff_temp
                ap_id_add = np.append(ap_id_add, ap_id_temp)
                ap_id = np.insert(ap_id, ibundle+idx_temp+1, ap_id_temp)
                ap_diff = np.diff(ap_id)
            else:
                iy += 1

        #### 
        print("Auto-fix found %d fiber(s) to add"%len(ap_id_add))

        #### save new spat_id positions into a file
        dirname = os.path.join(self.folder_trace, 'slits_file')
        filename = self.filename_trace.split('_')[2]+'_slits.txt'

        if not os.path.exists(dirname):
            os.mkdir(dirname)

        file = open(os.path.join(dirname, filename), 'w')
        for temp in ap_id_add:
            file.write("%d\n"%temp)
        file.close()

        self.make_file_apermap_slits()

    # def make_file_apermap(self):
    #     N_ap = np.int32(self.ifu_type.Ntotal/2)
    #     basename = os.path.basename(self.path_MasterSlits)
    #     fname = basename.split('_')[1]+'_apermap'
    #     shoe = fname[0]
    #     print(shoe, fname)

    #     #hdul = cached_fits_open(self.path_MasterSlits)
    #     hdul = fits.open(self.path_MasterSlits)
    #     hdr = hdul[1].header
    #     data = hdul[1].data

    #     N_sl  = np.int32(hdr['NSLITS'])
    #     nspec = np.int32(hdr['NSPEC']) ### binning?
    #     nspat = np.int32(hdr['NSPAT'])
    #     map_ap = np.zeros((nspat,nspec), dtype=np.int32)
    #     print('nspat, nspec=',nspat,nspec)
    #     print('Note: %d out of %d fibers are found by pypeit_trace_edges.'%(N_sl,N_ap))

    #     ####
    #     if N_ap>N_sl:
    #         print('!!! Warning: Missing %d fiber(s). !!!'%(N_ap-N_sl))
    #     elif N_ap<N_sl:
    #         print('!!! Warning: Found %d more fiber(s) than expected. !!!'%(N_sl-N_ap))

    #     #### make AperMap (note by YYS: need to improve speed)
    #     for i_ap in range(N_sl):
    #         ap_num = i_ap+1
    #         for x_temp in range(nspec):
    #             ap_y1 = int(np.round(data[i_ap]['left_init'][x_temp]-1))
    #             ap_y2 = int(np.round(data[i_ap]['right_init'][x_temp]))
    #             map_ap[ap_y1:ap_y2, x_temp] = np.int32(ap_num)

    #     #### cut data
    #     map_ap = self.cut_data_by_edges(map_ap, shoe)

    #     #### find the maximum number of pixels in all slits
    #     num_ap = np.zeros(N_ap, dtype=np.int32)
    #     for i_ap in range(N_ap):
    #         num_ap[i_ap] = np.sum(map_ap==i_ap+1)
    #     num_max = np.max(num_ap)

    #     #plt.imshow(map_ap, origin='lower')

    #     #### save AperMap
    #     #### the following header params may require modifying
    #     hdu_map = fits.PrimaryHDU(map_ap)
    #     hdr_map = hdu_map.header
    #     hdr_map['IFUTYPE'] = (self.ifu_type.label, 'type of IFU')
    #     #hdr_map.set('IFUTYPE', IFU_type, 'type of IFU')
    #     hdr_map['NIFU1'] = (self.ifu_type.Nx, 'number of IFU columns')
    #     hdr_map['NIFU2'] = (self.ifu_type.Ny, 'number of IFU rows')
    #     hdr_map['NSLITS'] = (N_sl, 'number of slits')
    #     hdr_map['NMAX'] = (num_max, 'maximum number of pixels among all apertures')
    #     hdr_map['BINNING'] = ('1x1', 'binning')
    #     #hdu_map = fits.PrimaryHDU(map_ap, header=hdr_map)

    #     dir_aperMap = os.path.join(self.ent_folder_trace.get(),'aperMap')
    #     if not os.path.exists(dir_aperMap):
    #         os.mkdir(dir_aperMap)

    #     today_temp = datetime.today().strftime("%y%m%d")
    #     file_aperMap = 'ap%s_%s_%s_%s.fits'%(
    #         shoe, 
    #         self.ifu_type.label, 
    #         self.lbl_file_pypeit['text'][0:4],
    #         today_temp)

    #     path_aperMap = os.path.join(dir_aperMap, file_aperMap)
    #     hdu_map.writeto(path_aperMap,overwrite=True)

    #     #### show apermap
    #     self.clear_image(shoe=shoe)

    #     title = '%s (N_sl=%d)'%(fname,N_sl)
    #     self.update_image_single(map_ap, title, shoe=shoe, uniform=True)

    #     #### show message
    #     #info_temp = 'Saved as %s'%path_aperMap
    #     #self.popup_showinfo('aperMap', info_temp)
    #     #print('\n++++\n++++ %s\n++++\n'%(info_temp))

    #     #### check the MasterSlits file
    #     N_slits = self.check_file_MasterSlits()       
    #     if N_slits!=self.ifu_type.Ntotal/2:
    #         self.btn_make_apermap_bundles['state'] = 'normal'

    #     self.window.focus_force()

    def pick_edges_mono(self):
        return 0

    def open_folder(self):
        """Open a folder using the Browse button."""
        dirname = filedialog.askdirectory(initialdir=self.folder_rawdata)
        if not dirname:
            return
        self.ent_folder.delete(0, tk.END)
        self.ent_folder.insert(tk.END, dirname)
        self.folder_rawdata = dirname
        self.refresh_folder()

    def refresh_folder(self, *args):
        """Refresh the file list in the folder."""
        dirname = self.ent_folder.get()
        if not dirname:
            return
        self.list_fits_file(dirname)
        self.window.focus_force()

    def list_fits_file(self, dirname):
        file_list = os.listdir(dirname)
        fnames = [
            f[1:5]
            for f in file_list
            if os.path.isfile(os.path.join(dirname, f))
            and f.lower().endswith(("c1.fits"))
            and f.lower().startswith(("b"))
        ]
        fnames.sort(reverse=False)
        self.fit_files.set(fnames)

        #txt_edit.delete(1.0, tk.END)
        #for f in fnames:
        #    text = f+"\n"
        #    txt_edit.insert(tk.END, text)

    def disable_make_apermap(self):
        #self.btn_make_pypeit['state'] = 'disabled'
        self.btn_run_pypeit.configure(state='disabled')
        self.btn_select_bundles.configure(state='disabled')
        self.btn_make_apermap_bundles.configure(state='disabled')
        self.btn_select_slits.configure(state='disabled')
        self.btn_make_apermap_slits.configure(state='disabled')

    def open_folder_trace(self):
        """Open a folder using the Browse button."""
        dirname = filedialog.askdirectory(initialdir=self.folder_trace)
        if not dirname:
            return
        self.ent_folder_trace.delete(0, tk.END)
        self.ent_folder_trace.insert(tk.END, dirname)
        self.folder_trace = dirname
        self.disable_make_apermap()
        self.window.focus_force()

    def refresh_folder_trace(self, *args):
        self.folder_trace = self.ent_folder_trace.get()
        self.disable_make_apermap()

    def load_4fits(self):
        """Load the selected fits file."""
        #shoe_i = self.shoe.get()
        dirname = self.ent_folder.get()
        idxs = self.box_files.curselection()

        if len(idxs)==1:
            idx = int(idxs[0])
            fnum = self.box_files.get(idx)
            fname = os.path.join(dirname, "b%sc1.fits"%(fnum))
            if os.path.isfile(fname):
                # get header info
                tmp = self.get_header_info(fname)

                self.data_full = None
                self.hdr_c1_b = None
                self.data_full, self.hdr_c1_b = pack_4fits_simple(fnum, dirname, 'b')
                self.file_current = fnum

                self.data_full2 = None
                self.hdr_c1_r = None
                self.data_full2, self.hdr_c1_r = pack_4fits_simple(fnum, dirname, 'r')

                #### show the fits image
                self.clear_image()
                self.update_image()
                return fnum #shoe_i+fnum
        else:
            return '0000'

    def get_header_info(self, pathname):
        """Get header info from the fits file."""
        if os.path.isfile(pathname):
            hdr_tmp = fits.open(pathname)[0].header
            self.ifu_type = IFUM_UNIT(hdr_tmp['IFU'])
            self.HDR_BINNING = hdr_tmp['BINNING']
            self.HDR_CONFIG = hdr_tmp['CONFIGFL']
            self.HDR_SLIDE = hdr_tmp['SLIDE']
            self.HDR_SLITNAME = hdr_tmp['SLITNAME']

            self.HDR_CONFIG = self.HDR_CONFIG.replace('Config', 'c')
            self.HDR_CONFIG = self.HDR_CONFIG.replace('unknown', 'c?')

            return 1
        else:
            return 0

    def disable_dependent_btns(self):
        self.btn_select_curve_b.configure(state='disabled')
        self.btn_select_curve_r.configure(state='disabled')
        self.btn_select_edges_b.configure(state='disabled')
        self.btn_select_edges_r.configure(state='disabled')
        self.btn_make_trace.configure(state='disabled')
        #self.btn_make_pypeit['state'] = 'disabled'
        self.btn_run_pypeit.configure(state='disabled')
        self.btn_select_bundles.configure(state='disabled')
        self.btn_make_apermap_bundles.configure(state='disabled')
        self.btn_select_slits.configure(state='disabled')
        self.btn_make_apermap_slits.configure(state='disabled')
        self.btn_make_apermap_mono.configure(state='disabled')
        #self.lbl_slitnum['text'] = 'N_slits = 000'

    def gray_all_lbl_file(self):
        bg_color = ("gray80", "gray20")
        self.lbl_file_curve.configure(fg_color=bg_color, text_color=("black", "white"))
        self.lbl_file_edges.configure(fg_color=bg_color, text_color=("black", "white"))
        self.lbl_file_trace.configure(fg_color=bg_color, text_color=("black", "white"))
        self.lbl_file_pypeit.configure(fg_color=bg_color, text_color=("black", "white"))
        self.lbl_file_apermap.configure(fg_color=bg_color, text_color=("black", "white"))
        self.lbl_file_mono.configure(fg_color=bg_color, text_color=("black", "white"))

    def load_4fits_curve(self):
        label = self.load_4fits()
        if label!='0000':
            self.lbl_file_curve.configure(text=label)
            self.gray_all_lbl_file()
            self.lbl_file_curve.configure(fg_color='yellow', text_color='black')
            self.disable_dependent_btns()
            self.btn_select_curve_b.configure(state='normal')
            self.btn_select_curve_r.configure(state='normal')

    def load_4fits_edges(self):
        label = self.load_4fits()
        if label!='0000':
            self.lbl_file_edges.configure(text=label)
            self.gray_all_lbl_file()
            self.lbl_file_edges.configure(fg_color='yellow', text_color='black')
            self.disable_dependent_btns()
            self.btn_select_edges_b.configure(state='normal')
            self.btn_select_edges_r.configure(state='normal')

    def load_4fits_trace(self):
        label = self.load_4fits()
        if label!='0000':
            self.filename_trace = label+'_trace'
            self.lbl_file_trace.configure(text=label)
            self.gray_all_lbl_file()
            self.lbl_file_trace.configure(fg_color='yellow', text_color='black')
            self.disable_dependent_btns()
            self.btn_make_trace.configure(state='normal')

    def open_fits_trace(self):
        self.folder_trace = self.ent_folder_trace.get()
        path_tmp = filedialog.askopenfilename(initialdir=self.folder_trace)
        self.load_fits_trace(path_tmp)

    def load_fits_trace(self, pathname):
        if os.path.isfile(pathname) and pathname.endswith("_trace.fits"):
            #hdul_temp = cached_fits_open(filename)
            dirname, fname = os.path.split(pathname)
            fname = fname[1:]
            hdul_temp = fits.open(os.path.join(dirname, 'b'+fname))
            self.data_full = np.float32(hdul_temp[0].data)
            hdul_temp = fits.open(os.path.join(dirname, 'r'+fname))
            self.data_full2 = np.float32(hdul_temp[0].data)

            #### get config info from header
            tmp = self.get_header_info(pathname)

            #### update trace folder
            self.folder_trace = os.path.dirname(pathname)
            self.ent_folder_trace.delete(0, tk.END)
            self.ent_folder_trace.insert(tk.END, self.folder_trace)

            #### update trace file
            self.filename_trace = os.path.basename(pathname)
            file_temp = self.filename_trace.split('.')[0]
            self.lbl_file_pypeit.configure(text=file_temp[1:])
            self.file_current = file_temp[1:]
            self.shoe.set(file_temp[0])

            self.gray_all_lbl_file()
            self.lbl_file_pypeit.configure(fg_color='yellow', text_color='black')
            self.disable_dependent_btns()
            #self.btn_make_pypeit['state'] = 'normal'
            self.btn_run_pypeit.configure(state='normal')

            #### show the fits image
            self.clear_image()
            self.update_image()
        else:
            self.data_full = np.ones((4048, 4048), dtype=np.int32)
            self.filename_trace = "0000_trace.fits"
            self.file_current = "0000"
            self.lbl_file_pypeit.configure(text=self.filename_trace.split('.')[0])
            self.disable_make_apermap()
            self.gray_all_lbl_file()
            self.remove_image()
        self.window.focus_force()

    def open_fits_apermap2(self):
        self.folder_trace = self.ent_folder_trace.get()
        pathname = filedialog.askopenfilename(initialdir=os.path.join(self.folder_trace,'aperMap'))
        dirname, filename = os.path.split(pathname)
        if os.path.isfile(pathname) and filename.endswith(".fits") and filename.startswith('ap'):
            #hdul_temp = cached_fits_open(filename)
            fname = filename[4:]
            hdul_temp = fits.open(os.path.join(dirname, 'apb_'+fname))
            self.hdr_b = hdul_temp[0].header
            self.data_full = np.float32(hdul_temp[0].data)
            hdul_temp = fits.open(os.path.join(dirname, 'apr_'+fname))
            self.hdr_r = hdul_temp[0].header
            self.data_full2 = np.float32(hdul_temp[0].data)

            #### update apermap folder
            self.folder_apermap = dirname

            #### update apermap file
            self.filename_apermap = os.path.basename(pathname)
            file_temp = self.filename_apermap.split('.')[0].split('_')
            self.file_current = file_temp[1]
            for i in range(2, len(file_temp)-1):
                self.file_current += '_'+file_temp[i] 
            self.lbl_file_mono.configure(text=self.file_current)
            #self.shoe.set(file_temp[0])

            self.gray_all_lbl_file()
            self.lbl_file_mono.configure(fg_color='yellow', text_color='black')
            self.disable_dependent_btns()
            self.btn_make_apermap_mono.configure(state='normal')

            #### show the fits image
            self.clear_image()
            self.update_image(uniform=True)
        else:
            self.data_full = np.ones((4048, 4048), dtype=np.int32)
            self.filename_apermap = "apb_XXX_0000.fits"
            self.file_current = "0000"
            self.lbl_file_mono.configure(text='XXX_'+self.file_current)
            self.btn_make_apermap_mono.configure(state='disabled')
            self.gray_all_lbl_file()
            self.remove_image()
        self.window.focus_force()

    def open_fits_apermap(self):
        self.folder_trace = self.ent_folder_trace.get()
        pathname = filedialog.askopenfilename(initialdir=os.path.join(self.folder_trace, 'aperMap'))
        dirname, filename = os.path.split(pathname)

        if os.path.isfile(pathname) and filename.startswith("ap") and filename.endswith(".fits"):
            str_temp = filename.split('_')
            fnum_temp = str_temp[0][2]+str_temp[2]

            #### load Apermap
            #hdul_temp = cached_fits_open(pathname)
            hdul_temp = fits.open(pathname)
            N_slits_file = hdul_temp[0].header['NSLITS']
            self.ifu_type = self.get_ifu_type(N_slits_file)
            self.data_full = np.float32(hdul_temp[0].data)

            #### update paths and file names
            self.folder_trace = dirname
            self.ent_folder_trace.delete(0, tk.END)
            self.ent_folder_trace.insert(tk.END, self.folder_trace)

            self.filename_trace = filename
            file_temp = "ap%s_%s"%(fnum_temp, str_temp[4].split('.')[0])
            self.lbl_file_apermap.configure(text=file_temp)
            self.file_current = file_temp+" (Nslits=%s)"%N_slits_file
            self.shoe.set(file_temp[2])

            #### handle other widegts
            self.gray_all_lbl_file()
            self.lbl_file_apermap.configure(fg_color='yellow', text_color='black')
            self.disable_dependent_btns()
            self.btn_select_bundles.configure(state='normal')
            self.btn_make_apermap_bundles.configure(state='normal')
            self.btn_select_slits.configure(state='normal')
            self.btn_make_apermap_slits.configure(state='normal')

            #### show the fits image
            self.clear_image()
            self.remove_image(shoe='r')
            #self.update_image(uniform=True)
            self.update_image_single(self.data_full, self.file_current, shoe='b', uniform=True)

            #### first check if the corresponding MasterSlits file exists
            path_MasterSlits_temp = os.path.join(os.path.dirname(dirname), 'pypeit_file/Masters/MasterSlits_%s_trace.fits.gz'%fnum_temp)
            if False: #os.path.isfile(path_MasterSlits_temp):
                #### update paths and file names
                self.path_MasterSlits = path_MasterSlits_temp
                N_slits = self.check_file_MasterSlits()
            else:
                #### trace file was made using new method not using pypeit
                self.path_MasterSlits = None
        else:
            self.data_full = np.ones((4048, 4048), dtype=np.int32)
            self.filename_trace = "apx0000_0000.fits"
            self.file_current = "0000"
            self.lbl_file_apermap.configure(text=self.filename_trace.split('.')[0])
            self.btn_select_bundles.configure(state='disabled')
            self.btn_make_apermap_bundles.configure(state='disabled')
            self.btn_select_slits.configure(state='disabled')
            self.btn_make_apermap_slits.configure(state='disabled')
            self.gray_all_lbl_file()
            self.remove_image()
        self.window.focus_force()

    def init_image1(self):
        # the figure that will contain the plot
        self.fig = Figure(figsize = img_figsize)
        self.fig.clf()

        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame2)  # A tk.DrawingArea.
        self.canvas.draw()

        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack(fill='both', expand=True) #grid(row=0, column=0)

        # creating the Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame2)
        self.toolbar.update()

        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().pack(fill='both', expand=True) #grid(row=1, column=0)

    def init_image2(self):
        # the figure that will contain the plot
        self.fig2 = Figure(figsize = img_figsize)
        self.fig2.clf()

        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.frame3)  # A tk.DrawingArea.
        self.canvas2.draw()

        # placing the canvas on the Tkinter window
        self.canvas2.get_tk_widget().pack(fill='both', expand=True) #grid(row=0, column=0)

        # creating the Matplotlib toolbar
        self.toolbar2 = NavigationToolbar2Tk(self.canvas2, self.frame3)
        self.toolbar2.update()

        # placing the toolbar on the Tkinter window
        self.canvas2.get_tk_widget().pack(fill='both', expand=True) #grid(row=1, column=0)

    def remove_image(self, shoe='both'):
        if shoe=='b' or shoe=='both':
            self.fig.clf()
            self.canvas.draw_idle()
        if shoe=='r' or shoe=='both':
            self.fig2.clf()
            self.canvas2.draw_idle()

    def clear_image(self, shoe='both'):
        if shoe=='b' or shoe=='both':
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
        if shoe=='r' or shoe=='both':
            self.fig2.clf()
            self.ax2 = self.fig2.add_subplot(111)

    def update_image(self, shoe='both', percent=85.9, uniform=False):
        if shoe=='b' or shoe=='both':
            if uniform:
                self.ax.imshow(self.data_full, origin='lower', cmap='gray', vmin=np.min(self.data_full), vmax=np.min(self.data_full)+1)
            else:
                self.ax.imshow(self.data_full, origin='lower', cmap='gray', vmin=0.0, vmax=np.percentile(self.data_full, percent))
            self.ax.set_title("b%s"%self.file_current)
            self.fig.set_tight_layout(True)
            self.canvas.draw_idle()
        if shoe=='r' or shoe=='both':
            if uniform:
                self.ax2.imshow(self.data_full2, origin='lower', cmap='gray', vmin=np.min(self.data_full2), vmax=np.min(self.data_full2)+1)
            else:
                self.ax2.imshow(self.data_full2, origin='lower', cmap='gray', vmin=0.0, vmax=np.percentile(self.data_full2, percent))
            self.ax2.set_title("r%s"%self.file_current)
            self.fig2.set_tight_layout(True)
            self.canvas2.draw_idle()

    def update_image_single(self, data, title, shoe='b', percent=85.9, uniform=False):
        if shoe=='b':
            if uniform:
                self.ax.imshow(data, origin='lower', cmap='gray', vmin=np.min(data), vmax=np.min(data)+1)
            else:
                self.ax.imshow(data, origin='lower', cmap='gray', vmin=0.0, vmax=np.percentile(data, percent))
            self.ax.set_title(title)
            self.fig.set_tight_layout(True)
            self.canvas.draw_idle()
        if shoe=='r':
            if uniform:
                self.ax2.imshow(data, origin='lower', cmap='gray', vmin=np.min(data), vmax=np.min(data)+1)
            else:
                self.ax2.imshow(data, origin='lower', cmap='gray', vmin=0.0, vmax=np.percentile(data, percent))
            self.ax2.set_title(title)
            self.fig2.set_tight_layout(True)
            self.canvas2.draw_idle()

    def plot_curve(self, shoe='both'):
        if shoe=='b' or shoe=='both':
            yy = np.arange(len(self.data_full))
            xx = func_parabola(yy, self.param_curve_b[0], self.param_curve_b[1], self.param_curve_b[2])
            self.ax.plot(xx, yy, 'r--')
            self.update_image(shoe='b')
        if shoe=='r' or shoe=='both':
            yy = np.arange(len(self.data_full2))
            xx = func_parabola(yy, self.param_curve_r[0], self.param_curve_r[1], self.param_curve_r[2])
            self.ax2.plot(xx, yy, 'r--')
            self.update_image(shoe='r')

    def plot_edges(self, shoe='both'):
        if shoe=='b' or shoe=='both':
            yy = np.arange(len(self.data_full))
            x1 = func_parabola(yy, self.param_curve_b[0], self.param_curve_b[1], self.param_edges_b[0])
            self.ax.plot(x1, yy, 'r--')
            x2 = func_parabola(yy, self.param_curve_b[0], self.param_curve_b[1], self.param_edges_b[1])
            self.ax.plot(x2, yy, 'r--')
            self.update_image(shoe='b')

            # check if any values in x1 and x2 are out of bounds, set the number in Step 2 to red
            if np.any(x1 < 0) or np.any(x1 > len(self.data_full[0])):
                self.ent_param_edges_X1_b.configure(text_color='red')
            else:
                if self.ent_param_edges_X1_b.cget("state") == 'normal':
                    self.ent_param_edges_X1_b.configure(text_color=self.DEFAULT_FG)
                else: 
                    self.ent_param_edges_X1_b.configure(text_color=self.DEFAULT_FG_DISABLED)

            if np.any(x2 < 0) or np.any(x2 > len(self.data_full[0])):
                self.ent_param_edges_dX_b.configure(text_color='red')
            else:
                if self.ent_param_edges_dX_b.cget("state") == 'normal':
                    self.ent_param_edges_dX_b.configure(text_color=self.DEFAULT_FG)
                else:
                    self.ent_param_edges_dX_b.configure(text_color=self.DEFAULT_FG_DISABLED)

        if shoe=='r' or shoe=='both':
            yy = np.arange(len(self.data_full2))
            x1 = func_parabola(yy, self.param_curve_r[0], self.param_curve_r[1], self.param_edges_r[0])
            self.ax2.plot(x1, yy, 'r--')
            x2 = func_parabola(yy, self.param_curve_r[0], self.param_curve_r[1], self.param_edges_r[1])
            self.ax2.plot(x2, yy, 'r--')
            self.update_image(shoe='r')

            # check if any values in x1 and x2 are out of bounds, set the number in Step 2 to red
            # if np.any(x1 < 0) or np.any(x1 > len(self.data_full2[0])):
            #     self.ent_param_edges_X1_r.config(fg='red', disabledforeground='red')
            # else:
            #     self.ent_param_edges_X1_r.config(fg=self.DEFAULT_FG, disabledforeground=self.DEFAULT_FG_DISABLED)

            # if np.any(x2 < 0) or np.any(x2 > len(self.data_full2[0])):
            #     self.ent_param_edges_dX_r.config(fg='red', disabledforeground='red')
            # else:
            #     self.ent_param_edges_dX_r.config(fg=self.DEFAULT_FG, disabledforeground=self.DEFAULT_FG_DISABLED)

            if np.any(x1 < 0) or np.any(x1 > len(self.data_full2[0])):
                self.ent_param_edges_X1_r.configure(text_color='red')
            else:
                if self.ent_param_edges_X1_r.cget("state") == 'normal':
                    self.ent_param_edges_X1_r.configure(text_color=self.DEFAULT_FG)
                else:
                    self.ent_param_edges_X1_r.configure(text_color=self.DEFAULT_FG_DISABLED)

            if np.any(x2 < 0) or np.any(x2 > len(self.data_full2[0])):
                self.ent_param_edges_dX_r.configure(text_color='red')
            else:
                if self.ent_param_edges_dX_r.cget("state") == 'normal': 
                    self.ent_param_edges_dX_r.configure(text_color=self.DEFAULT_FG)
                else:
                    self.ent_param_edges_dX_r.configure(text_color=self.DEFAULT_FG_DISABLED)

    def add_instructions_on_image(self, shoe='both'):
        if shoe=='b' or shoe=='both':
            self.ax.text(0.04, 0.02, 'Right Click to select a point;\nUse Toolbar\'s Zoom to zoom in', transform=self.ax.transAxes,
                     fontsize=10, color='white', ha='left', va='bottom',
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
            self.ax.text(0.04, 0.98, 'Press ESC to quit without saving', transform=self.ax.transAxes,
                     fontsize=10, color='white', ha='left', va='top',
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        if shoe=='r' or shoe=='both':
            self.ax2.text(0.04, 0.02, 'Right Click to select a point;\nUse Toolbar\'s Zoom to zoom in', transform=self.ax2.transAxes,
                     fontsize=10, color='white', ha='left', va='bottom',
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
            self.ax2.text(0.04, 0.98, 'Press ESC to quit without saving', transform=self.ax2.transAxes,
                     fontsize=10, color='white', ha='left', va='top',
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    def pick_points(self, shoe):
        """Pick points on the image to select the curve points."""
        if shoe=='b':
            btn_tmp = self.btn_select_curve_b
            fig_tmp = self.fig
        elif shoe=='r':
            btn_tmp = self.btn_select_curve_r
            fig_tmp = self.fig2

        # initialize the figure
        self.disable_others()
        btn_tmp.configure(state='normal')
        self.clear_image(shoe=shoe)
        self.add_instructions_on_image(shoe=shoe)
        self.update_image(shoe=shoe)

        # connect the button press event to the on_click_curve function
        self.cidpick = fig_tmp.canvas.mpl_connect(
            'button_press_event', 
            lambda event: self.on_click_curve(event, shoe=shoe))
        self.cidexit = fig_tmp.canvas.mpl_connect(
            'key_press_event', 
            lambda event: self.key_press(event, step='curve', shoe=shoe))

        # show info
        info_temp = \
            'Select 7 points on the %s-side image:\n\n'%(shoe) \
            + '  -- Right Click to select a point\n' \
            + '  -- Use Toolbar\'s Zoom to zoom in\n' \
            + '  -- Press ESC to quit without saving'
        self.popup_left_aligned('Pick points', info_temp)

        self.window.focus_force()

    def pick_edges(self, shoe):
        """Pick points on the image to select the edges."""
        if shoe=='b':
            btn_tmp = self.btn_select_edges_b
            fig_tmp = self.fig
        elif shoe=='r':
            btn_tmp = self.btn_select_edges_r
            fig_tmp = self.fig2

        # initialize the figure
        self.disable_others()
        btn_tmp.configure(state='normal')
        self.clear_image(shoe=shoe)
        self.add_instructions_on_image(shoe=shoe)

        if shoe=='b':
            self.ax.axhline(len(self.data_full)/2, c='g', ls='-')
        elif shoe=='r':
            self.ax2.axhline(len(self.data_full2)/2, c='g', ls='-')

        self.update_image(shoe=shoe)

        # connect the button press event to the on_click_curve function
        self.cidpick = fig_tmp.canvas.mpl_connect(
            'button_press_event', 
            lambda event: self.on_click_edges(event, shoe=shoe))
        self.cidexit = fig_tmp.canvas.mpl_connect(
            'key_press_event', 
            lambda event: self.key_press(event, step='edges', shoe=shoe))
        
        # show info
        info_temp = \
            'Select 2 points on the green line on the %s-side image:\n\n'%(shoe) \
            + '  -- Right Click to select a point\n' \
            + '  -- Use Toolbar\'s Zoom to zoom in\n' \
            + '  -- Press ESC to quit without saving'
        self.popup_left_aligned('Pick points', info_temp)

        self.window.focus_force()

    def pick_slits(self):
        self.disable_others()
        self.btn_select_slits.configure(state='normal')

        self.ax.axvline(len(self.data_full[0])/2, c='g', ls='-')
        #self.update_image(shoe='b', uniform=True)
        self.update_image_single(self.data_full, self.file_current, shoe='b', uniform=True)
        self.cidpick = self.fig.canvas.mpl_connect('button_press_event', self.on_click_slits)
        self.cidexit = self.fig.canvas.mpl_connect('key_press_event', self.key_press_slits)

    def pick_bundles(self):
        self.disable_others()
        self.btn_select_bundles.configure(state='normal')

        self.ax.axvline(len(self.data_full[0])/2, c='g', ls='-')
        #self.update_image(shoe='b', uniform=True)
        self.update_image_single(self.data_full, self.file_current, shoe='b', uniform=True)
        self.cidpick = self.fig.canvas.mpl_connect('button_press_event', self.on_click_slits)
        self.cidexit = self.fig.canvas.mpl_connect('key_press_event', self.key_press_bundles)

    def key_press_slits(self, event):
        if event.key == 'escape':
            #### enable other functions
            self.enable_others()
            self.btn_select_slits.configure(state='normal')
            self.btn_make_apermap_slits.configure(state='normal')

            #### save the y positions into a file
            dirname = os.path.join(self.folder_trace, 'slits_file')
            filename = self.filename_trace.split('_')[2]+'_slits.txt'

            if not os.path.exists(dirname):
                os.mkdir(dirname)

            file = open(os.path.join(dirname, filename), 'w')
            for point in self.points:
                file.write("%d\n"%point[1])
            file.close()

            #### break the mpl connection
            self.break_mpl_connect(shoe='b')

    def key_press_bundles(self, event):
        if event.key == 'escape':
            #### enable other functions
            self.enable_others()
            self.btn_select_bundles.configure(state='normal')
            self.btn_make_apermap_bundles.configure(state='normal')

            #### save the y positions into a file
            dirname = os.path.join(self.folder_trace, 'slits_file')
            filename = self.filename_trace.split('_')[2]+'_bundles.txt'

            if not os.path.exists(dirname):
                os.mkdir(dirname)

            file = open(os.path.join(dirname, filename), 'w')
            for point in self.points:
                file.write("%d\n"%point[1])
            file.close()

            #### break the mpl connection
            self.break_mpl_connect(shoe='b')

    def key_press(self, event, step, shoe):
        if event.key == 'escape':
            #### enable other functions
            self.enable_others()

            if step=='curve':
                self.btn_select_curve_b.configure(state='normal')
                self.btn_select_curve_r.configure(state='normal')
            elif step=='edges':
                self.btn_select_edges_b.configure(state='normal')
                self.btn_select_edges_r.configure(state='normal')

            #### break the mpl connection
            self.break_mpl_connect(shoe=shoe)

    def on_click_slits(self, event):
        if event.button is MouseButton.RIGHT:
            if np.abs(self.y_last-event.ydata)>2:
                self.points.append([event.xdata, event.ydata])
                print(len(self.points), event.xdata, event.ydata)
                self.x_last, self.y_last = event.xdata, event.ydata
                self.ax.scatter(len(self.data_full[0])/2, event.ydata, c='r', marker='x', zorder=10)
                #self.update_image(uniform=True)
                self.update_image_single(self.data_full, self.file_current, shoe='b', uniform=True)

    def on_click_curve(self, event, shoe):
        if event.button is MouseButton.RIGHT:
            if len(self.points)<7 and (np.abs(self.x_last-event.xdata)>1 or np.abs(self.y_last-event.ydata)>10):
                self.points.append([event.xdata, event.ydata])
                print(len(self.points), event.xdata, event.ydata)
                self.x_last, self.y_last = event.xdata, event.ydata

                if shoe=='b':
                    self.ax.scatter(event.xdata, event.ydata, c='r', marker='x', zorder=10)
                elif shoe=='r':
                    self.ax2.scatter(event.xdata, event.ydata, c='r', marker='x', zorder=10)
                self.update_image(shoe=shoe)

            if len(self.points)==7:
                pts = np.array(self.points)

                for pt in pts:
                    print("%.1f %.1f"%(pt[0], pt[1]))

                popt, pcov = curve_fit(func_parabola, pts[:, 1], pts[:, 0])
                print(popt)

                if shoe=='b':
                    self.txt_param_curve_A_b.set("%.3e"%(popt[0]))
                    self.txt_param_curve_B_b.set("%.1f"%(popt[1]))
                    self.txt_param_curve_C_b.set("%.1f"%(popt[2]))
                    self.param_curve_b = popt
                elif shoe=='r':
                    self.txt_param_curve_A_r.set("%.3e"%(popt[0]))
                    self.txt_param_curve_B_r.set("%.1f"%(popt[1]))
                    self.txt_param_curve_C_r.set("%.1f"%(popt[2]))
                    self.param_curve_r = popt

                #### plot the fitted curve
                self.plot_curve(shoe=shoe)

                #### enable other functions
                self.enable_others()
                
                self.btn_select_curve_b.configure(state='normal') 
                self.btn_select_curve_r.configure(state='normal') 

                #### break the mpl connection
                self.break_mpl_connect(shoe=shoe)

    def on_click_edges(self, event, shoe):
        # reset lock conditions
        self.state_edge_lock_b.set(0)
        self.state_edge_lock_r.set(0)

        if event.button is MouseButton.RIGHT:
            if len(self.points)<2 and (np.abs(self.x_last-event.xdata)>1 or np.abs(self.y_last-event.ydata)>10):
                self.points.append([event.xdata, event.ydata])
                print(len(self.points), event.xdata, event.ydata)
                self.x_last, self.y_last = event.xdata, event.ydata

                if shoe=='b':
                    self.ax.scatter(event.xdata, len(self.data_full)/2, c='r', marker='x', zorder=10)
                elif shoe=='r':
                    self.ax2.scatter(event.xdata, len(self.data_full2)/2, c='r', marker='x', zorder=10)
                self.update_image(shoe=shoe)

            if len(self.points) == 2:

                if shoe=='b':
                    self.param_edges_b = np.array([self.points[0][0], self.points[1][0]])
                    self.param_edges_b = np.sort(self.param_edges_b)
                    self.param_edges_b = np.append(self.param_edges_b, self.param_edges_b[1]-self.param_edges_b[0])
                elif shoe=='r':
                    self.param_edges_r = np.array([self.points[0][0], self.points[1][0]])
                    self.param_edges_r = np.sort(self.param_edges_r)
                    self.param_edges_r = np.append(self.param_edges_r, self.param_edges_r[1]-self.param_edges_r[0])

                self.param_edges_offset = self.param_edges_r[0]-self.param_edges_b[0]
                self.renew_param_edges()

                #### plot the edges
                self.plot_edges(shoe=shoe)

                #### enable other functions
                self.enable_others()
                self.btn_select_edges_b.configure(state='normal')
                self.btn_select_edges_r.configure(state='normal')

                #### break the mpl connection
                self.break_mpl_connect(shoe=shoe)

    def enable_others(self):
        self.btn_folder.configure(state='normal')
        self.btn_refresh.configure(state='normal')
        self.ent_folder.configure(state='normal')
        self.box_files.configure(state='normal')
        self.shoe1.configure(state='normal')
        self.shoe2.configure(state='normal')
        #self.pca1['state'] = 'normal'
        #self.pca2['state'] = 'normal'

        self.btn_load_curve.configure(state='normal')
        self.btn_load_edges.configure(state='normal')
        self.btn_load_trace.configure(state='normal')
        self.btn_load_pypeit.configure(state='normal')
        self.btn_load_apermap.configure(state='normal')
        self.btn_load_mono.configure(state='normal')

        self.btn_plot_curve_b.configure(state='normal')
        self.btn_plot_edges_b.configure(state='normal')
        self.ent_param_curve_A_b.configure(state='normal')
        self.ent_param_curve_B_b.configure(state='normal')
        self.ent_param_curve_C_b.configure(state='normal')
        self.ent_param_edges_X1_b.configure(state='normal')
        self.ent_param_edges_dX_b.configure(state='normal')
        self.ent_param_edges_offset.configure(state='normal')
        self.cbtn_edge_lock_r.configure(state='normal')
        self.cbtn_edge_lock_b.configure(state='normal')
        self.btn_load_all_param.configure(state='normal')

        self.btn_plot_curve_r.configure(state='normal')
        self.btn_plot_edges_r.configure(state='normal')
        self.ent_param_curve_A_r.configure(state='normal')
        self.ent_param_curve_B_r.configure(state='normal')
        self.ent_param_curve_C_r.configure(state='normal')
        self.ent_param_edges_X1_r.configure(state='normal')
        self.ent_param_edges_dX_r.configure(state='normal')

        #self.ent_smash_range['state'] = 'normal'
        self.ent_labelname_mono.configure(state='normal')

        self.ent_folder_trace.configure(state='normal')
        self.btn_folder_trace.configure(state='normal')

    def disable_others(self):
        self.btn_folder.configure(state='disabled')
        self.btn_refresh.configure(state='disabled')
        self.ent_folder.configure(state='disabled')
        self.box_files.configure(state='disabled')
        self.shoe1.configure(state='disabled')
        self.shoe2.configure(state='disabled')
        #self.pca1['state'] = 'disabled'
        #self.pca2['state'] = 'disabled'

        self.btn_load_curve.configure(state='disabled')
        self.btn_load_edges.configure(state='disabled')
        self.btn_load_trace.configure(state='disabled')
        self.btn_load_pypeit.configure(state='disabled')
        self.btn_load_apermap.configure(state='disabled')
        self.btn_load_mono.configure(state='disabled')

        self.btn_plot_curve_b.configure(state='disabled')
        self.btn_plot_edges_b.configure(state='disabled')
        self.ent_param_curve_A_b.configure(state='disabled')
        self.ent_param_curve_B_b.configure(state='disabled')
        self.ent_param_curve_C_b.configure(state='disabled')
        self.ent_param_edges_X1_b.configure(state='disabled')
        self.ent_param_edges_dX_b.configure(state='disabled')
        self.ent_param_edges_offset.configure(state='disabled')
        self.cbtn_edge_lock_r.configure(state='disabled')
        self.cbtn_edge_lock_b.configure(state='disabled')

        self.btn_plot_curve_r.configure(state='disabled')
        self.btn_plot_edges_r.configure(state='disabled')
        self.ent_param_curve_A_r.configure(state='disabled')
        self.ent_param_curve_B_r.configure(state='disabled')
        self.ent_param_curve_C_r.configure(state='disabled')
        self.ent_param_edges_X1_r.configure(state='disabled')
        self.ent_param_edges_dX_r.configure(state='disabled')
        self.btn_load_all_param.configure(state='disabled')

        #self.ent_smash_range['state'] = 'disabled'
        self.ent_labelname_mono.configure(state='disabled')

        self.ent_folder_trace.configure(state='disabled')
        self.btn_folder_trace.configure(state='disabled')

        self.btn_select_curve_b.configure(state='disabled')
        self.btn_select_edges_b.configure(state='disabled')
        self.btn_select_curve_r.configure(state='disabled')
        self.btn_select_edges_r.configure(state='disabled')
        self.btn_make_trace.configure(state='disabled')
        self.btn_select_bundles.configure(state='disabled')
        self.btn_make_apermap_bundles.configure(state='disabled')
        self.btn_select_slits.configure(state='disabled')
        self.btn_make_apermap_slits.configure(state='disabled')
        self.btn_make_apermap_mono.configure(state='disabled')

    def break_mpl_connect(self, shoe='b'):
        #### break the mpl connection
        self.curve_points = np.array(self.points)
        self.points = []
        self.x_last, self.y_last = -1., -1.

        if shoe=='b':
            self.fig.canvas.mpl_disconnect(self.cidpick)
            self.fig.canvas.mpl_disconnect(self.cidexit)
        elif shoe=='r':
            self.fig2.canvas.mpl_disconnect(self.cidpick)
            self.fig2.canvas.mpl_disconnect(self.cidexit)

    def cut_data_by_edges(self, data_raw, shoe):
        data_mask = np.zeros_like(data_raw)
        yy = np.arange(len(data_raw))
        if shoe == 'b':
            x1 = np.int32(np.round(func_parabola(yy, self.param_curve_b[0], self.param_curve_b[1], self.param_edges_b[0])))
            x2 = np.int32(np.round(func_parabola(yy, self.param_curve_b[0], self.param_curve_b[1], self.param_edges_b[1])))
        elif shoe == 'r':
            x1 = np.int32(np.round(func_parabola(yy, self.param_curve_r[0], self.param_curve_r[1], self.param_edges_r[0])))
            x2 = np.int32(np.round(func_parabola(yy, self.param_curve_r[0], self.param_curve_r[1], self.param_edges_r[1])))

        for iy in range(len(yy)):
            data_mask[iy, x1[iy-1]:x2[iy]] = data_raw[iy, x1[iy-1]:x2[iy]]

        return data_mask

    def make_file_trace(self):
        self.data_full = self.cut_data_by_edges(self.data_full, 'b')
        self.data_full2 = self.cut_data_by_edges(self.data_full2, 'r')
        self.file_current = self.file_current+"_trace"

        #### show the fits image
        self.clear_image(shoe='both')
        self.plot_edges(shoe='both')

        #### write the fits file
        self.folder_trace = self.ent_folder_trace.get()
        path_trace_b = write_trace_file(self.data_full, self.hdr_c1_b, self.folder_trace, 'b'+self.file_current)
        path_trace_r = write_trace_file(self.data_full2, self.hdr_c1_r, self.folder_trace, 'r'+self.file_current)

        #### control widgets
        self.btn_make_trace.configure(state='disabled')

        #### save the curve profile
        self.save_curve_file()

        #### show info
        info_temp = "Trace files made!\n\n Go straightly to Step 4"
        self.popup_showinfo("Trace file saved", info_temp)

        #### load the newly saved trace file
        self.load_fits_trace(path_trace_r)

    def make_file_apermap_mono(self):
        self.data_full = self.cut_data_by_edges(self.data_full, 'b')
        self.data_full2 = self.cut_data_by_edges(self.data_full2, 'r')

        #### show the fits image
        self.clear_image(shoe='both')
        #self.plot_edges(shoe='both')
        self.update_image(shoe='both', uniform=True)

        #### write the fits file
        temp_name = self.lbl_file_mono.cget("text").split('_')
        fname = temp_name[0]+'_'+self.ent_labelname_mono.get()
        for i in range(1, len(temp_name)):
            fname += '_'+temp_name[i]
        cut_apermap(self.data_full, self.hdr_b, self.folder_apermap, 'apb_'+fname)
        cut_apermap(self.data_full2, self.hdr_r, self.folder_apermap, 'apr_'+fname)

        #### control widgets
        self.btn_make_apermap_mono.configure(state='disabled')

        #### show info
        info_temp = "Monochromatic Apermap files made!\n\n" \
            +"Saved to %s"%(self.folder_apermap)
        self.popup_showinfo("Apermap file saved", info_temp)

        self.window.focus_force()

    def popup_showinfo(self, title, message):
        """Show a popup message box."""
        showinfo(title=title, message=message)

    def popup_left_aligned(self, title, message):
        """Creates and displays a custom info dialog with left-aligned text."""
        win = ctk.CTkToplevel(self.window)
        win.title(title)

        # set the size of the window
        win_width = 400
        win_height = 200

        # get the screen width and height
        screen_width = win.winfo_screenwidth()
        screen_height = win.winfo_screenheight()

        # center the window on the screen
        x = (screen_width // 2) - (win_width // 2)
        y = (screen_height // 2) - (win_height // 2)
        win.geometry(f"{win_width}x{win_height}+{x}+{y}")
        win.resizable(False, False)
        win.grab_set()
        win.focus_set()
        win.transient(self.window)
        win.attributes("-topmost", True)
        win.protocol("WM_DELETE_WINDOW", win.destroy)
        win.bind("<Return>", lambda event: win.destroy())
        win.bind("<Escape>", lambda event: win.destroy())

        # Create a Label with justify="left" to align text to the left
        label = ctk.CTkLabel(win, text=message, justify="left")  
        label.pack(fill="both", expand=True, padx=5, pady=5)

        button = ctk.CTkButton(win, text="OK", command=win.destroy)
        button.pack(pady=1, padx=5)


if __name__ == "__main__":
    main()
