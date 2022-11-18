#!/usr/bin/env python
from genericpath import exists
import os
import tkinter as tk
from tkinter import filedialog
from tkinter.messagebox import showinfo

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from datetime import datetime

from astropy.io import fits
from scipy.optimize import curve_fit

from utils_ifum import IFUM_UNIT, pack_4fits_simple, func_parabola, readFloat_space, write_pypeit_file, write_trace_file, cached_fits_open

import subprocess
from multiprocessing import Process


def main():
    #### Create the entire GUI program
    program = IFUM_AperMap_Maker()

    #### Start the GUI event loop
    program.window.mainloop()


def check_appearance():
    """Checks DARK/LIGHT mode of macos."""
    """True=DARK, False=LIGHT"""
    cmd = 'defaults read -g AppleInterfaceStyle'
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    return bool(p.communicate()[0])

#print(check_appearance())
if check_appearance():
    LABEL_COLOR = 'limegreen'
    BG_COLOR = 'black'
else:
    LABEL_COLOR = 'black'
    BG_COLOR='white'

class IFUM_AperMap_Maker:

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("IFUM AperMap Maker")
        self.window.rowconfigure(0, minsize=800, weight=1)
        self.window.columnconfigure(1, minsize=600, weight=1)
        self.window.columnconfigure(2, minsize=600, weight=1)

        #### IFUM units
        self.LSB = IFUM_UNIT('LSB')
        self.STD = IFUM_UNIT('STD')
        self.HR = IFUM_UNIT('HR')
        self.M2FS = IFUM_UNIT('M2FS')
        self.UNKNOWN = IFUM_UNIT('unknown')
        self.ifu_type = self.UNKNOWN

        #### frames
        self.frame1 = tk.Frame(self.window, relief=tk.RAISED, bd=2, bg=BG_COLOR)
        self.frame1.grid(row=0, column=0, sticky="ns")

        self.frame2 = tk.Frame(self.window)
        self.frame2.grid(row=0, column=1, sticky="nsew")

        self.frame3 = tk.Frame(self.window)
        self.frame3.grid(row=0, column=2, sticky="nsew")

        #### global widgets
        self.ent_folder = None
        self.box_files = None
        self.lbl_file_curve = None

        #### global values
        self.data_full = np.ones((4048, 4048), dtype=np.int32)
        self.data_full2 = np.ones((4048, 4048), dtype=np.int32)
        self.file_current = "0000"

        self.folder_default = "./data_raw/"
        self.folder_trace   = "./data_trace/"
        self.path_MasterSlits = ' '

        self.points = []
        self.x_last, self.y_last = -1., -1.
        self.curve_points = np.array([])
        self.param_curve_b = np.array([1.22771242e-05, 2.28414233e+03, 7.87506089e+02]) #np.zeros(3)
        self.param_curve_r = np.array([1.53314740e-05, 2.12797487e+03, 6.75423701e+02]) #np.zeros(3)
        self.param_edges_b = np.array([418., 1250., 1250.-418.])#np.zeros(2)
        self.param_edges_r = np.array([426., 1258., 1258.-426.])#np.zeros(2)
        self.param_smash_range = '0.4,0.6'

        #### initialize string variables
        self.fit_files = tk.StringVar()
        self.shoe = tk.StringVar()
        self.pca = tk.StringVar()
        self.txt_param_curve_A_b = tk.StringVar(value=['%.3e'%self.param_curve_b[0]])
        self.txt_param_curve_B_b = tk.StringVar(value=['%.1f'%self.param_curve_b[1]])
        self.txt_param_curve_C_b = tk.StringVar(value=['%.1f'%self.param_curve_b[2]])
        self.txt_param_curve_A_r = tk.StringVar(value=['%.3e'%self.param_curve_r[0]])
        self.txt_param_curve_B_r = tk.StringVar(value=['%.1f'%self.param_curve_r[1]])
        self.txt_param_curve_C_r = tk.StringVar(value=['%.1f'%self.param_curve_r[2]])
        self.txt_param_edges_X1_b = tk.StringVar(value=['%.0f'%self.param_edges_b[0]])
        self.txt_param_edges_X2_b = tk.StringVar(value=['%.0f'%self.param_edges_b[1]])
        self.txt_param_edges_dX_b = tk.StringVar(value=['%.0f'%self.param_edges_b[2]])
        self.txt_param_edges_X1_r = tk.StringVar(value=['%.0f'%self.param_edges_r[0]])
        self.txt_param_edges_X2_r = tk.StringVar(value=['%.0f'%self.param_edges_r[1]])
        self.txt_param_edges_dX_r = tk.StringVar(value=['%.0f'%self.param_edges_r[2]])
        self.txt_folder_trace = tk.StringVar(value=[self.folder_trace])
        self.txt_smash_range = tk.StringVar(value=[self.param_smash_range])

        #### create all widgets
        #self.my_counter = None  # All attributes should be initialize in init
        self.create_widgets_files()
        self.create_widgets_curve()  # step 1
        self.create_widgets_edges()  # step 2
        self.create_widgets_trace()  # step 3
        self.create_widgets_pypeit() # step 4
        self.create_widgets_add_slits()   # step 5
        #self.create_widgets_mono()   # step 6
        self.bind_widgets()

        #### initialize widgets
        self.shoe.set('b')
        self.pca.set('off')
        self.refresh_folder()
        self.init_image1()
        self.init_image2()

    def create_widgets_files(self):
        start, lines = 0, 4
        rows = np.arange(start, start+lines)
        #### folder
        lbl_folder = tk.Label(self.frame1, text="Folder", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_folder.grid(row=rows[0], column=0, sticky="w")

        self.btn_folder = tk.Button(self.frame1, width=6, text="Browse...", command=self.open_folder, highlightbackground=BG_COLOR)
        self.btn_folder.grid(row=rows[0], column=7, sticky="e", padx=5, pady=5)

        self.ent_folder = tk.Entry(self.frame1, textvariable=tk.StringVar(value=[self.folder_default]))
        self.ent_folder.grid(row=rows[1], column=0, columnspan=8, sticky="ew")

        #### file list
        lbl_files = tk.Label(self.frame1, text="Files:", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_files.grid(row=rows[2], column=0, sticky="w")

        self.btn_refresh = tk.Button(self.frame1, width=6, text="Refresh", command=self.refresh_folder, highlightbackground=BG_COLOR)
        self.btn_refresh.grid(row=rows[2], column=7, sticky="e", padx=5, pady=5)

        self.box_files = tk.Listbox(self.frame1, listvariable=self.fit_files)
        self.box_files.grid(row=rows[3], column=0, columnspan=8, sticky="nsew")

    def create_widgets_curve(self):
        """ step 1 fit curvature using an arc or twilight file """
        start, lines = 4, 4
        rows = np.arange(start, start+lines)

        lbl_step1 = tk.Label(self.frame1, text="Step 1:", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_step1.grid(row=rows[0], column=0, sticky="w")
        lbl_step1 = tk.Label(self.frame1, text="fit curvature using ARC/TWI files", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_step1.grid(row=rows[0], column=1, columnspan=5, sticky="w")

        self.lbl_file_curve = tk.Label(self.frame1, relief=tk.SUNKEN, text="0000", fg=LABEL_COLOR)
        self.lbl_file_curve.grid(row=rows[0], column=6, sticky="e")

        self.btn_load_curve = tk.Button(self.frame1, width=6, text="Load", command=self.load_4fits_curve, highlightbackground=BG_COLOR)
        self.btn_load_curve.grid(row=rows[0], column=7, sticky="e", padx=5, pady=5)

        #### pick points
        lbl_note_curve = tk.Label(self.frame1, text="Note: curve model is x-C = A*(y-B)^2; select 6 points", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_note_curve.grid(row=rows[1], column=1, columnspan=5, sticky="w")

        self.btn_select_curve_b = tk.Button(self.frame1, width=6, text="Select (b)", command=self.pick_points_b, state='disabled', highlightbackground=BG_COLOR)
        self.btn_select_curve_b.grid(row=rows[1], column=6, sticky="e", padx=5, pady=5)

        self.btn_select_curve_r = tk.Button(self.frame1, width=6, text="Select (r)", command=self.pick_points_r, state='disabled', highlightbackground=BG_COLOR)
        self.btn_select_curve_r.grid(row=rows[1], column=7, sticky="e", padx=5, pady=5)

        #### curve parameters (b-side)
        lbl_param_curve_A_b = tk.Label(self.frame1, text="b:  A =", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_param_curve_A_b.grid(row=rows[2], column=1, sticky="e")
        self.ent_param_curve_A_b = tk.Entry(self.frame1, width=6, textvariable=self.txt_param_curve_A_b)
        self.ent_param_curve_A_b.grid(row=rows[2], column=2, sticky="ew")

        lbl_param_curve_B_b = tk.Label(self.frame1, text="B =", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_param_curve_B_b.grid(row=rows[2], column=3, sticky="e")
        self.ent_param_curve_B_b = tk.Entry(self.frame1, width=6, textvariable=self.txt_param_curve_B_b)
        self.ent_param_curve_B_b.grid(row=rows[2], column=4, sticky="ew")

        lbl_param_curve_C_b = tk.Label(self.frame1, text="C =", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_param_curve_C_b.grid(row=rows[2], column=5, sticky="e")
        self.ent_param_curve_C_b = tk.Entry(self.frame1, width=6, textvariable=self.txt_param_curve_C_b)
        self.ent_param_curve_C_b.grid(row=rows[2], column=6, sticky="ew")

        self.btn_update_curve_b = tk.Button(self.frame1, width=6, text='Plot (b)', command=self.update_curve_b, highlightbackground=BG_COLOR)
        self.btn_update_curve_b.grid(row=rows[2], column=7, sticky="e", padx=5, pady=5)

        #### curve parameters (r-side)
        lbl_param_curve_A_r = tk.Label(self.frame1, text="r:  A =", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_param_curve_A_r.grid(row=rows[3], column=1, sticky="e")
        self.ent_param_curve_A_r = tk.Entry(self.frame1, width=6, textvariable=self.txt_param_curve_A_r)
        self.ent_param_curve_A_r.grid(row=rows[3], column=2, sticky="ew")

        lbl_param_curve_B_r = tk.Label(self.frame1, text="B =", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_param_curve_B_r.grid(row=rows[3], column=3, sticky="e")
        self.ent_param_curve_B_r = tk.Entry(self.frame1, width=6, textvariable=self.txt_param_curve_B_r)
        self.ent_param_curve_B_r.grid(row=rows[3], column=4, sticky="ew")

        lbl_param_curve_C_r = tk.Label(self.frame1, text="C =", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_param_curve_C_r.grid(row=rows[3], column=5, sticky="e")
        self.ent_param_curve_C_r = tk.Entry(self.frame1, width=6, textvariable=self.txt_param_curve_C_r)
        self.ent_param_curve_C_r.grid(row=rows[3], column=6, sticky="ew")

        self.btn_update_curve_r = tk.Button(self.frame1, width=6, text='Plot (r)', command=self.update_curve_r, highlightbackground=BG_COLOR)
        self.btn_update_curve_r.grid(row=rows[3], column=7, sticky="e", padx=5, pady=5)

    def create_widgets_edges(self):
        """ step 2 select edges using a science or twilight file """
        start, lines = 8, 4
        rows = np.arange(start, start+lines)

        lbl_step2 = tk.Label(self.frame1, text="Step 2:", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_step2.grid(row=rows[0], column=0, sticky="w")
        lbl_step2 = tk.Label(self.frame1, text="select full spectral span using SCI/TWI files", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_step2.grid(row=rows[0], column=1, columnspan=5, sticky="w")

        self.lbl_file_edges = tk.Label(self.frame1, relief=tk.SUNKEN, text="0000", fg=LABEL_COLOR)
        self.lbl_file_edges.grid(row=rows[0], column=6, sticky="e")

        self.btn_load_edges = tk.Button(self.frame1, width=6, text="Load", command=self.load_4fits_edges, state='normal', highlightbackground=BG_COLOR)
        self.btn_load_edges.grid(row=rows[0], column=7, sticky="e", padx=5, pady=5)

        #### pick edges
        lbl_note_edges = tk.Label(self.frame1, text="Note: one left and one right along y-axis middle line", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_note_edges.grid(row=rows[1], column=1, columnspan=5, sticky="w")

        self.btn_select_edges_b = tk.Button(self.frame1, width=6, text="Select (b)", command=self.pick_edges_b, state='disabled', highlightbackground=BG_COLOR)
        self.btn_select_edges_b.grid(row=rows[1], column=6, sticky="e", padx=5, pady=5)

        self.btn_select_edges_r = tk.Button(self.frame1, width=6, text="Select (r)", command=self.pick_edges_r, state='disabled', highlightbackground=BG_COLOR)
        self.btn_select_edges_r.grid(row=rows[1], column=7, sticky="e", padx=5, pady=5)

        #### edge parameters (b-side)
        lbl_param_edges_X1_b = tk.Label(self.frame1, text="b: X1 =", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_param_edges_X1_b.grid(row=rows[2], column=1, sticky="e")
        self.ent_param_edges_X1_b = tk.Entry(self.frame1, width=6, textvariable=self.txt_param_edges_X1_b)
        self.ent_param_edges_X1_b.grid(row=rows[2], column=2, sticky="ew")

        lbl_param_edges_X2_b = tk.Label(self.frame1, text="X2 =", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_param_edges_X2_b.grid(row=rows[2], column=3, sticky="e")
        self.ent_param_edges_X2_b = tk.Entry(self.frame1, width=6, textvariable=self.txt_param_edges_X2_b, state='disable')
        self.ent_param_edges_X2_b.grid(row=rows[2], column=4, sticky="ew")

        lbl_param_edges_dX_b = tk.Label(self.frame1, text="dX =", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_param_edges_dX_b.grid(row=rows[2], column=5, sticky="e")
        self.ent_param_edges_dX_b = tk.Entry(self.frame1, width=6, textvariable=self.txt_param_edges_dX_b)
        self.ent_param_edges_dX_b.grid(row=rows[2], column=6, sticky="ew")

        self.btn_update_edges_b = tk.Button(self.frame1, width=6, text='Plot (b)', command=self.update_edges_b, highlightbackground=BG_COLOR)
        self.btn_update_edges_b.grid(row=rows[2], column=7, sticky="e", padx=5, pady=5)

        #### edge parameters (r-side)
        lbl_param_edges_X1_r = tk.Label(self.frame1, text="r: X1 =", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_param_edges_X1_r.grid(row=rows[3], column=1, sticky="e")
        self.ent_param_edges_X1_r = tk.Entry(self.frame1, width=6, textvariable=self.txt_param_edges_X1_r)
        self.ent_param_edges_X1_r.grid(row=rows[3], column=2, sticky="ew")

        lbl_param_edges_X2_r = tk.Label(self.frame1, text="X2 =", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_param_edges_X2_r.grid(row=rows[3], column=3, sticky="e")
        self.ent_param_edges_X2_r = tk.Entry(self.frame1, width=6, textvariable=self.txt_param_edges_X2_r, state='disable')
        self.ent_param_edges_X2_r.grid(row=rows[3], column=4, sticky="ew")

        lbl_param_edges_dX_r = tk.Label(self.frame1, text="dX =", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_param_edges_dX_r.grid(row=rows[3], column=5, sticky="e")
        self.ent_param_edges_dX_r = tk.Entry(self.frame1, width=6, textvariable=self.txt_param_edges_dX_r)
        self.ent_param_edges_dX_r.grid(row=rows[3], column=6, sticky="ew")

        self.btn_update_edges_r = tk.Button(self.frame1, width=6, text='Plot (r)', command=self.update_edges_r, highlightbackground=BG_COLOR)
        self.btn_update_edges_r.grid(row=rows[3], column=7, sticky="e", padx=5, pady=5)

    def create_widgets_trace(self):
        """ step 3 check and make a masked LED fits file for tracing """
        start, lines = 12, 2
        rows = np.arange(start, start+lines)

        lbl_step3 = tk.Label(self.frame1, text="Step 3:", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_step3.grid(row=rows[0], column=0, sticky="w")
        lbl_step3 = tk.Label(self.frame1, text="make TRACE files using LED files", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_step3.grid(row=rows[0], column=1, columnspan=5, sticky="w")

        self.lbl_file_trace = tk.Label(self.frame1, relief=tk.SUNKEN, text="0000", fg=LABEL_COLOR)
        self.lbl_file_trace.grid(row=rows[0], column=6, sticky="e")

        self.btn_load_trace = tk.Button(self.frame1, width=6, text="Load", command=self.load_4fits_trace, state='normal', highlightbackground=BG_COLOR)
        self.btn_load_trace.grid(row=rows[0], column=7, sticky="e", padx=5, pady=5)

        ####
        self.ent_folder_trace = tk.Entry(self.frame1, textvariable=self.txt_folder_trace, state='normal')
        self.ent_folder_trace.grid(row=rows[1], column=1, columnspan=5, sticky="ew")

        self.btn_folder_trace = tk.Button(self.frame1, width=6, text="Browse...", command=self.open_folder_trace, state='normal', highlightbackground=BG_COLOR)
        self.btn_folder_trace.grid(row=rows[1], column=6, sticky="ew", pady=5)

        self.btn_make_trace = tk.Button(self.frame1, width=6, text="Make", command=self.make_file_trace, state='disabled', highlightbackground=BG_COLOR)
        self.btn_make_trace.grid(row=rows[1], column=7, sticky="e", padx=5, pady=5)

    def create_widgets_pypeit(self):
        """ step 4 run pypeit for tracing and making the AperMap """
        start, lines = 14, 4
        rows = np.arange(start, start+lines)

        lbl_step4 = tk.Label(self.frame1, text="Step 4:", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_step4.grid(row=rows[0], column=0, sticky="w")
        lbl_step4 = tk.Label(self.frame1, text="make AperMap files using TRACE files", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_step4.grid(row=rows[0], column=1, columnspan=4, sticky="w")

        self.lbl_file_pypeit = tk.Label(self.frame1, relief=tk.SUNKEN, text="0000_trace", fg=LABEL_COLOR)
        self.lbl_file_pypeit.grid(row=rows[0], column=5, columnspan=2, sticky="e")

        self.btn_load_pypeit = tk.Button(self.frame1, width=6, text="Open", command=self.open_fits_trace, state='normal', highlightbackground=BG_COLOR)
        self.btn_load_pypeit.grid(row=rows[0], column=7, sticky="e", padx=5, pady=5)

        #### step 4a make a PypeIt file
        lbl_step4a = tk.Label(self.frame1, text="4a. Make PypeIt files", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_step4a.grid(row=rows[1], column=1, columnspan=2, sticky="w")
        #lbl_smash = tk.Label(self.frame1, text="smash range =", fg=LABEL_COLOR, bg=BG_COLOR)
        #lbl_smash.grid(row=rows[1], column=4, columnspan=2, sticky="e")

        lbl_pca = tk.Label(self.frame1, text="PCA:", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_pca.grid(row=rows[1], column=4, sticky="w")

        self.pca1 = tk.Radiobutton(self.frame1, text='off', variable=self.pca, value='off', fg=LABEL_COLOR, bg=BG_COLOR)
        self.pca1.grid(row=rows[1], column=4, sticky='e')
        self.pca2 = tk.Radiobutton(self.frame1, text='on', variable=self.pca, value='on', fg=LABEL_COLOR, bg=BG_COLOR)
        self.pca2.grid(row=rows[1], column=5, sticky='w')

        self.ent_smash_range = tk.Entry(self.frame1, width=6, textvariable=self.txt_smash_range, state='normal')
        self.ent_smash_range.grid(row=rows[1], column=6, sticky="ew")

        self.btn_make_pypeit = tk.Button(self.frame1, width=6, text='Make', command=self.make_file_pypeit, state='disabled', highlightbackground=BG_COLOR)
        self.btn_make_pypeit.grid(row=rows[1], column=7, sticky='e', padx=5, pady=5)

        #### step 4b run PypeIt
        lbl_step4b = tk.Label(self.frame1, text="4b. Run PypeIt to trace slits", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_step4b.grid(row=rows[2], column=1, columnspan=3, sticky="w")


        #### select shoe side
        lbl_shoe = tk.Label(self.frame1, text="Shoe:", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_shoe.grid(row=rows[2], column=4, sticky="w")

        self.shoe1 = tk.Radiobutton(self.frame1, text='b', variable=self.shoe, value='b', fg="cyan", bg=BG_COLOR)
        self.shoe1.grid(row=rows[2], column=4, sticky='e')
        self.shoe2 = tk.Radiobutton(self.frame1, text='r', variable=self.shoe, value='r', fg="red", bg=BG_COLOR)
        self.shoe2.grid(row=rows[2], column=5, sticky='w')

        self.btn_run_pypeit = tk.Button(self.frame1, width=6, text='Run', command=self.run_pypeit, state='disabled', highlightbackground=BG_COLOR)
        self.btn_run_pypeit.grid(row=rows[2], column=6, sticky='e', padx=5, pady=5)

        #### step 4c save the AperMap
        #lbl_save = tk.Label(self.frame1, text="4c. Save the AperMap file")
        #lbl_save.grid(row=rows[1], column=1, columnspan=4, sticky="w")
        #self.lbl_slitnum = tk.Label(self.frame1, relief=tk.SUNKEN, text="N_slits = 000", fg=LABEL_COLOR)
        #self.lbl_slitnum.grid(row=rows[3], column=5, columnspan=2, sticky="e")

        self.btn_make_apermap = tk.Button(self.frame1, width=6, text='Save', command=self.make_file_apermap, state='disabled', highlightbackground=BG_COLOR)
        self.btn_make_apermap.grid(row=rows[2], column=7, sticky='e', padx=5, pady=5)

    def create_widgets_add_slits(self):
        """ step 5 add bad/missing slits """
        start, lines = 18, 2
        rows = np.arange(start, start+lines)

        lbl_step5 = tk.Label(self.frame1, text="Step 5:", fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_step5.grid(row=rows[0], column=0, sticky="w")
        lbl_step5 = tk.Label(self.frame1, text="(optional) add missing slits to an AperMap",fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_step5.grid(row=rows[0], column=1, columnspan=4, sticky="w")

        self.lbl_file_apermap = tk.Label(self.frame1, relief=tk.SUNKEN, text="apx0000_0000",fg=LABEL_COLOR)
        self.lbl_file_apermap.grid(row=rows[0], column=5, columnspan=2, sticky="e")

        self.btn_load_apermap = tk.Button(self.frame1, width=6, text="Open", command=self.open_fits_apermap, state='normal', highlightbackground=BG_COLOR)
        self.btn_load_apermap.grid(row=rows[0], column=7, sticky="e", padx=5, pady=5)

        #### select y positions of all missing slits
        lbl_slits_note = tk.Label(self.frame1, text='Note: Select along x-axis middle line; ESC to finish', fg=LABEL_COLOR, bg=BG_COLOR)
        lbl_slits_note.grid(row=rows[1], column=1, columnspan=5, sticky='w')

        self.btn_select_slits = tk.Button(self.frame1, width=6, text="Select", command=self.pick_slits, state='disabled', highlightbackground=BG_COLOR)
        self.btn_select_slits.grid(row=rows[1], column=6, sticky="e", padx=5, pady=5)

        self.btn_make_apermap_slits = tk.Button(self.frame1, width=6, text='Save', command=self.make_file_apermap_slits, state='disabled', highlightbackground=BG_COLOR)
        self.btn_make_apermap_slits.grid(row=rows[1], column=7, sticky='e', padx=5, pady=5)

    #def create_widgets_mono(self):
    #    """ step 6 make monochromatic apermap """
    #    lbl_step6 = tk.Label(self.frame1, text="Step 6:")
    #    lbl_step6.grid(row=rows[2], column=0, sticky="w")
    #    lbl_step6 = tk.Label(self.frame1, text="(optional) make a monochromatic AperMap file")
    #    lbl_step6.grid(row=rows[2], column=1, columnspan=5, sticky="w")

    #    self.lbl_file_apermap = tk.Label(self.frame1, relief=tk.SUNKEN, text="x0000_full")
    #    self.lbl_file_apermap.grid(row=rows[2], column=6, sticky="e")

    #    self.btn_load_apermap = tk.Button(self.frame1, width=6, text="Open", command=self.open_fits, state='normal')
    #    self.btn_load_apermap.grid(row=rows[2], column=7, sticky="e", padx=5, pady=5)

    #    #### pick edges for the monochromatic spec span
    #    #lbl_edge = tk.Label(self.frame1, text="Note: one left and one right along y-axis middle line")
    #    #lbl_edge.grid(row=rows[1], column=1, sticky="w", columnspan=2)

    #    #self.btn_edge_mc_select = tk.Button(self.frame1, width=6, text="Select", command=self.pick_edges_mono, state='disabled')
    #    #self.btn_edge_mc_select.grid(row=rows[1], column=2, sticky="e", pady=5)

    #    #self.btn_edge_mc_make = tk.Button(self.frame1, width=6, text="Save", command=self.make_file_apermap, state='disabled')
    #    #self.btn_edge_mc_make.grid(row=rows[1], column=3, sticky="e", padx=5, pady=5)

    #    #self.lbl_edge_mc_param = tk.Label(self.frame1, relief=tk.SUNKEN, text="(x1, x2, dx) =")
    #    #self.lbl_edge_mc_param.grid(row=rows[1], column=1, columnspan=3, sticky="w")

    def bind_widgets(self):
        """ event bindings """
        self.ent_folder.bind('<Return>', self.refresh_folder)
        self.ent_folder.bind('<KP_Enter>', self.refresh_folder) # unique for macOS

        self.ent_param_curve_A_b.bind('<Return>', self.update_curve_b)
        self.ent_param_curve_A_b.bind('<KP_Enter>', self.update_curve_b) # unique for macOS
        self.ent_param_curve_B_b.bind('<Return>', self.update_curve_b)
        self.ent_param_curve_B_b.bind('<KP_Enter>', self.update_curve_b) # unique for macOS
        self.ent_param_curve_C_b.bind('<Return>', self.update_curve_b)
        self.ent_param_curve_C_b.bind('<KP_Enter>', self.update_curve_b) # unique for macOS
        self.ent_param_curve_A_r.bind('<Return>', self.update_curve_r)
        self.ent_param_curve_A_r.bind('<KP_Enter>', self.update_curve_r) # unique for macOS
        self.ent_param_curve_B_r.bind('<Return>', self.update_curve_r)
        self.ent_param_curve_B_r.bind('<KP_Enter>', self.update_curve_r) # unique for macOS
        self.ent_param_curve_C_r.bind('<Return>', self.update_curve_r)
        self.ent_param_curve_C_r.bind('<KP_Enter>', self.update_curve_r) # unique for macOS

        self.ent_param_edges_X1_b.bind('<Return>', self.update_edges_b)
        self.ent_param_edges_X1_b.bind('<KP_Enter>', self.update_edges_b) # unique for macOS
        self.ent_param_edges_dX_b.bind('<Return>', self.update_edges_b)
        self.ent_param_edges_dX_b.bind('<KP_Enter>', self.update_edges_b) # unique for macOS
        self.ent_param_edges_X1_r.bind('<Return>', self.update_edges_r)
        self.ent_param_edges_X1_r.bind('<KP_Enter>', self.update_edges_r) # unique for macOS
        self.ent_param_edges_dX_r.bind('<Return>', self.update_edges_r)
        self.ent_param_edges_dX_r.bind('<KP_Enter>', self.update_edges_r) # unique for macOS

        self.ent_folder_trace.bind('<Return>', self.refresh_folder_trace)
        self.ent_folder_trace.bind('<KP_Enter>', self.refresh_folder_trace) # unique for macOS

        self.ent_smash_range.bind('<Return>', self.refresh_smash_range)
        self.ent_smash_range.bind('<KP_Enter>', self.refresh_smash_range) # unique for macOS

        self.window.bind_all('<1>', lambda event: event.widget.focus_set())

    def refresh_smash_range(self, *args):
        self.window.focus_set()

    def update_curve_b(self, *agrs):
        shoe = 'b'
        self.refresh_param_curve(shoe)
        self.clear_image(shoe=shoe)
        self.plot_curve(shoe=shoe)
        self.window.focus_set()

    def update_curve_r(self, *agrs):
        shoe = 'r'
        self.refresh_param_curve(shoe)
        self.clear_image(shoe=shoe)
        self.plot_curve(shoe=shoe)
        self.window.focus_set()

    def update_edges_b(self, *args):
        shoe = 'b'
        self.refresh_param_edges(shoe)
        self.clear_image(shoe=shoe)
        self.plot_edges(shoe=shoe)
        self.window.focus_set()

    def update_edges_r(self, *args):
        shoe = 'r'
        self.refresh_param_edges(shoe)
        self.clear_image(shoe=shoe)
        self.plot_edges(shoe=shoe)
        self.window.focus_set()

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
        if shoe=='b':
            self.param_edges_b[0] = np.float32(self.ent_param_edges_X1_b.get())
            self.param_edges_b[1] = np.float32(self.ent_param_edges_X1_b.get())+np.float32(self.ent_param_edges_dX_b.get())
            self.param_edges_b[2] = np.float32(self.ent_param_edges_dX_b.get())
            self.ent_param_edges_X2_b['textvariable'] = tk.StringVar(value='%.0f'%self.param_edges_b[1])
        elif shoe=='r':
            self.param_edges_r[0] = np.float32(self.ent_param_edges_X1_r.get())
            self.param_edges_r[1] = np.float32(self.ent_param_edges_X1_r.get())+np.float32(self.ent_param_edges_dX_r.get())
            self.param_edges_r[2] = np.float32(self.ent_param_edges_dX_r.get())
            self.ent_param_edges_X2_r['textvariable'] = tk.StringVar(value='%.0f'%self.param_edges_r[1])

    def make_file_pypeit(self):
        dirname = self.ent_folder_trace.get()
        filename = self.lbl_file_pypeit['text']
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

        self.btn_run_pypeit['state'] = 'normal'
        #self.popup_showinfo('', 'Success to make a PypeIt file:\n %s'%os.path.join(dirname, 'pypeit_file', filename+'.pypeit'))

    def run_pypeit(self):
        dirname = self.ent_folder_trace.get()
        filename = self.lbl_file_pypeit['text']

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

        #### check the MasterSlits file
        N_slits = self.check_file_MasterSlits()
        #self.lbl_slitnum['text'] = 'N_slits = %d'%N_slits
        self.btn_make_apermap['state'] = 'normal'

    def check_file_MasterSlits(self):
        #hdul = cached_fits_open(self.path_MasterSlits)
        hdul = fits.open(self.path_MasterSlits)
        hdr = hdul[1].header
        N_slits = np.int32(hdr['NSLITS'])
        self.ifu_type = self.get_ifu_type(N_slits, 20)

        #### show messages
        info_temp = 'PypeIt found %s slits, close to %d (%s)'%(N_slits,self.ifu_type.Ntotal/2,self.ifu_type.label)
        self.popup_showinfo('PypeIt', info_temp)
        print('\n++++\n++++ %s\n++++\n'%(info_temp))

        return N_slits

    def get_ifu_type(self, Nslits, Nerr):
        if np.abs(Nslits-self.LSB.Ntotal/2)<Nerr:
            return self.LSB
        elif np.abs(Nslits-self.STD.Ntotal/2)<Nerr:
            return self.STD
        elif np.abs(Nslits-self.HR.Ntotal/2)<Nerr:
            return self.HR
        elif np.abs(Nslits-self.M2FS.Ntotal/2)<Nerr:
            return self.M2FS
        else:
            return self.UNKNOWN

    def make_file_apermap_slits(self):
        #### load MasterSlits file
        N_ap = np.int32(self.ifu_type.Ntotal/2)

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

        #### load missing slits file
        dirname_slits = os.path.join(self.folder_trace, 'slits_file')
        filename_slits = self.filename_trace.split('_')[2]+'_slits.txt'
        path_slits = os.path.join(dirname_slits, filename_slits)
        if os.path.isfile(path_slits):
            spat_id_missing = np.int32(readFloat_space(path_slits, 0))
        else:
            spat_id_missing = np.array([])

        #### add missing slits and make new AperMap
        print('Note: %d fiber(s) are added manually.'%(len(spat_id_missing)))
        spat_id_raw = data['spat_id']
        spat_id_new = np.append(spat_id_raw, spat_id_missing)
        spat_id_new = np.sort(spat_id_new)
        N_new = len(spat_id_new)

        if N_new>N_ap:
            print('!!! Warning: More bad fibers are added. !!!')
        elif N_new<N_ap:
            print('!!! Warning: Less bad fibers are added. !!!')

        for i_ap in range(N_new):
            ap_num = i_ap+1
            spat_id_temp = spat_id_new[i_ap]

            temp_index = np.where(spat_id_raw==spat_id_temp)[0]
            if len(temp_index)==1:
                i_temp = temp_index[0]
                for x_temp in range(nspec):
                    ap_y1 = np.int32(np.round(data[i_temp]['left_init'][x_temp]-1))
                    ap_y2 = np.int32(np.round(data[i_temp]['right_init'][x_temp]))
                    map_ap[ap_y1:ap_y2, x_temp] = np.int32(ap_num)
            else:
                #map_ap[spat_id_temp-2:spat_id_temp+1, int(nspec/2)-2:int(nspec/2)+1] = np.int32(ap_num)
                #print("!!!!!!", spat_id_temp-2, spat_id_temp+1, int(nspec/2)-2, int(nspec/2)+1)

                d1_spat_id = spat_id_temp - spat_id_new[i_ap-1]
                d2_spat_id = spat_id_new[i_ap+1] - spat_id_temp
                print(d1_spat_id, d2_spat_id)
                if d1_spat_id>d2_spat_id:
                    temp_index = np.where(spat_id_raw==spat_id_new[i_ap+1])[0]
                    if len(temp_index)==1:
                        i_temp = temp_index[0]
                        for x_temp in range(nspec):
                            ap_y1 = np.int32(np.round(data[i_temp]['left_init'][x_temp]-1))
                            ap_y2 = np.int32(np.round(data[i_temp]['right_init'][x_temp]))
                            map_ap[ap_y1-d2_spat_id:ap_y2-d2_spat_id, x_temp] = np.int32(ap_num)
                else:
                    temp_index = np.where(spat_id_raw==spat_id_new[i_ap-1])[0]
                    if len(temp_index)==1:
                        i_temp = temp_index[0]
                        for x_temp in range(nspec):
                            ap_y1 = np.int32(np.round(data[i_temp]['left_init'][x_temp]-1))
                            ap_y2 = np.int32(np.round(data[i_temp]['right_init'][x_temp]))
                            map_ap[ap_y1+d1_spat_id:ap_y2+d1_spat_id, x_temp] = np.int32(ap_num)
                            map_ap[ap_y1:ap_y2, x_temp] = np.int32(ap_num-1)

        #### cut data
        map_ap = self.cut_data_by_edges(map_ap)

        #### find the maximum number of pixels in all slits
        num_ap = np.zeros(N_ap, dtype=np.int32)
        for i_ap in range(N_ap):
            num_ap[i_ap] = np.sum(map_ap==i_ap+1)
        num_max = np.max(num_ap)

        #plt.imshow(map_ap, origin='lower')

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

        today_temp = datetime.today().strftime('%y%m%d')
        today_backup = datetime.today().strftime('%y%m%d_%H%M')
        dir_aperMap = self.ent_folder_trace.get()
        dir_backup = os.path.join(dir_aperMap, 'backup_aperMap')
        if not os.path.exists(dir_aperMap):
            os.mkdir(dir_aperMap)
        if not os.path.exists(dir_backup):
            os.mkdir(dir_backup)
        path_aperMap = os.path.join(dir_aperMap, 'ap%s_%s_%s_%s_3000.fits'%(self.lbl_file_apermap['text'][2], self.ifu_type.label, self.lbl_file_apermap['text'][3:7],today_temp))
        path_backup = os.path.join(dir_backup, 'ap%s_%s_%s_%s.fits'%(self.lbl_file_apermap['text'][2], self.ifu_type.label, self.lbl_file_apermap['text'][3:7],today_backup))
        hdu_map.writeto(path_aperMap,overwrite=True)
        os.system('cp %s %s'%(path_aperMap, path_backup))

        #self.btn_select_slits['state'] = 'disabled'
        #self.btn_select_slits['state'] = 'disabled'

        info_temp = 'Saved as %s'%path_aperMap
        self.popup_showinfo('aperMap', info_temp)
        print('\n++++\n++++ %s\n++++\n'%(info_temp))

    def make_file_apermap(self):
        N_ap = np.int32(self.ifu_type.Ntotal/2)

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

        #### make AperMap (note by YYS: need to improve speed)
        for i_ap in range(N_sl):
            ap_num = i_ap+1
            for x_temp in range(nspec):
                ap_y1 = int(np.round(data[i_ap]['left_init'][x_temp]-1))
                ap_y2 = int(np.round(data[i_ap]['right_init'][x_temp]))
                map_ap[ap_y1:ap_y2, x_temp] = np.int32(ap_num)

        #### cut data
        map_ap = self.cut_data_by_edges(map_ap)

        #### find the maximum number of pixels in all slits
        num_ap = np.zeros(N_ap, dtype=np.int32)
        for i_ap in range(N_ap):
            num_ap[i_ap] = np.sum(map_ap==i_ap+1)
        num_max = np.max(num_ap)

        #plt.imshow(map_ap, origin='lower')

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

        today_temp = datetime.today().strftime("%y%m%d")
        today_backup = datetime.today().strftime("%y%m%d_%H%M")
        dir_aperMap = os.path.join(self.ent_folder_trace.get(),'aperMap')
        dir_backup = os.path.join(dir_aperMap, 'backup_aperMap')
        if not os.path.exists(dir_aperMap):
            os.mkdir(dir_aperMap)
        if not os.path.exists(dir_backup):
            os.mkdir(dir_backup)
        file_aperMap = 'ap%s_%s_%s_%s_3000.fits'%(self.lbl_file_pypeit['text'][0], self.ifu_type.label, self.lbl_file_pypeit['text'][1:5],today_temp)
        file_backup = 'ap%s_%s_%s_%s.fits'%(self.lbl_file_pypeit['text'][0], self.ifu_type.label, self.lbl_file_pypeit['text'][1:5],today_backup)
        path_aperMap = os.path.join(dir_aperMap, file_aperMap)
        path_backup = os.path.join(dir_backup, file_backup)
        hdu_map.writeto(path_aperMap,overwrite=True)
        os.system('cp %s %s'%(path_aperMap, path_backup))

        info_temp = 'Saved as %s'%path_aperMap
        self.popup_showinfo('aperMap', info_temp)
        print('\n++++\n++++ %s\n++++\n'%(info_temp))


    def pick_edges_mono(self):
        return 0

    def open_folder(self):
        """Open a folder using the Browse button."""
        dirname = filedialog.askdirectory(initialdir=self.folder_default)
        if not dirname:
            return
        self.ent_folder.delete(0, tk.END)
        self.ent_folder.insert(tk.END, dirname)
        self.folder_default = dirname

    def refresh_folder(self, *args):
        """Refresh the file list in the folder."""
        dirname = self.ent_folder.get()
        if not dirname:
            return
        self.list_fits_file(dirname)
        self.window.focus_set()

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
        self.btn_make_pypeit['state'] = 'disabled'
        self.btn_run_pypeit['state'] = 'disabled'
        self.btn_make_apermap['state'] = 'disabled'
        self.btn_select_slits['state'] = 'disabled'
        self.btn_make_apermap_slits['state'] = 'disabled'

    def open_folder_trace(self):
        """Open a folder using the Browse button."""
        dirname = filedialog.askdirectory(initialdir=self.folder_trace)
        if not dirname:
            return
        self.ent_folder_trace.delete(0, tk.END)
        self.ent_folder_trace.insert(tk.END, dirname)
        self.folder_trace = dirname
        self.disable_make_apermap()
        self.window.focus_set()

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

    def disable_dependent_btns(self):
        self.btn_select_curve_b['state'] = 'disabled'
        self.btn_select_curve_r['state'] = 'disabled'
        self.btn_select_edges_b['state'] = 'disabled'
        self.btn_select_edges_r['state'] = 'disabled'
        self.btn_make_trace['state'] = 'disabled'
        self.btn_make_pypeit['state'] = 'disabled'
        self.btn_run_pypeit['state'] = 'disabled'
        self.btn_make_apermap['state'] = 'disabled'
        self.btn_select_slits['state'] = 'disabled'
        self.btn_make_apermap_slits['state'] = 'disabled'
        #self.lbl_slitnum['text'] = 'N_slits = 000'

    def gray_all_lbl_file(self):
        _dummy_lbl = tk.Label(self.frame1)
        bg_color = _dummy_lbl['bg']
        self.lbl_file_curve.config(bg=bg_color)
        self.lbl_file_edges.config(bg=bg_color)
        self.lbl_file_trace.config(bg=bg_color)
        self.lbl_file_pypeit.config(bg=bg_color)
        self.lbl_file_apermap.config(bg=bg_color)
        _dummy_lbl.destroy()

    def load_4fits_curve(self):
        label = self.load_4fits()
        if label!='0000':
            self.lbl_file_curve["text"] = label
            self.gray_all_lbl_file()
            self.lbl_file_curve.config(bg='yellow')
            self.disable_dependent_btns()
            self.btn_select_curve_b['state'] = 'normal'
            self.btn_select_curve_r['state'] = 'normal'

    def load_4fits_edges(self):
        label = self.load_4fits()
        if label!='0000':
            self.lbl_file_edges["text"] = label
            self.gray_all_lbl_file()
            self.lbl_file_edges.config(bg='yellow')
            self.disable_dependent_btns()
            self.btn_select_edges_b['state'] = 'normal'
            self.btn_select_edges_r['state'] = 'normal'

    def load_4fits_trace(self):
        label = self.load_4fits()
        if label!='0000':
            self.filename_trace = label+'_trace'
            self.lbl_file_trace["text"] = label
            self.gray_all_lbl_file()
            self.lbl_file_trace.config(bg='yellow')
            self.disable_dependent_btns()
            self.btn_make_trace['state'] = 'normal'

    def open_fits_trace(self):
        self.folder_trace = self.ent_folder_trace.get()
        filename = filedialog.askopenfilename(initialdir=self.folder_trace)
        if os.path.isfile(filename) and filename.endswith("_trace.fits"):
            #hdul_temp = cached_fits_open(filename)
            dirname, fname = os.path.split(filename)
            fname = fname[1:]
            hdul_temp = fits.open(os.path.join(dirname, 'b'+fname))
            self.data_full = np.float32(hdul_temp[0].data)
            hdul_temp = fits.open(os.path.join(dirname, 'r'+fname))
            self.data_full2 = np.float32(hdul_temp[0].data)

            #### update trace folder
            self.folder_trace = os.path.dirname(filename)
            self.ent_folder_trace.delete(0, tk.END)
            self.ent_folder_trace.insert(tk.END, self.folder_trace)

            #### update trace file
            self.filename_trace = os.path.basename(filename)
            file_temp = self.filename_trace.split('.')[0]
            self.lbl_file_pypeit['text'] = file_temp[1:]
            self.file_current = file_temp[1:]
            self.shoe.set(file_temp[0])

            self.gray_all_lbl_file()
            self.lbl_file_pypeit.config(bg='yellow')
            self.disable_dependent_btns()
            self.btn_make_pypeit['state'] = 'normal'

            #### show the fits image
            self.clear_image()
            self.update_image()
        else:
            self.data_full = np.ones((4048, 4048), dtype=np.int32)
            self.filename_trace = "0000_trace.fits"
            self.file_current = "0000"
            self.lbl_file_pypeit['text'] = self.filename_trace.split('.')[0]
            self.disable_make_apermap()
            self.gray_all_lbl_file()
            self.fig.clf()
            self.canvas.draw_idle()
        self.window.focus_set()

    def open_fits_apermap(self):
        self.folder_trace = self.ent_folder_trace.get()
        pathname = filedialog.askopenfilename(initialdir=self.folder_trace)
        dirname, filename = os.path.split(pathname)

        if os.path.isfile(pathname) and filename.startswith("ap") and filename.endswith(".fits"):
            #### first check if the corresponding MasterSlits file exists
            str_temp = filename.split('_')
            fnum_temp = str_temp[0][2]+str_temp[2]
            path_MasterSlits_temp = os.path.join(os.path.dirname(dirname), 'pypeit_file/Masters/MasterSlits_%s_trace.fits.gz'%fnum_temp)

            if os.path.isfile(path_MasterSlits_temp):
                #### update paths and file names
                self.path_MasterSlits = path_MasterSlits_temp
                N_slits = self.check_file_MasterSlits()

                ####
                #hdul_temp = cached_fits_open(pathname)
                hdul_temp = fits.open(pathname)
                #N_slits_file = hdul_temp[0].header['NSLITS']

                #if N_slits==N_slits_file:
                self.folder_trace = dirname
                self.ent_folder_trace.delete(0, tk.END)
                self.ent_folder_trace.insert(tk.END, self.folder_trace)

                self.filename_trace = filename
                file_temp = "ap%s_%s"%(fnum_temp, str_temp[4].split('.')[0])
                self.lbl_file_apermap['text'] = file_temp
                self.file_current = file_temp+" (Nslits=%s)"%N_slits ### check if this is correct
                self.shoe.set(file_temp[2])

                #### handle other widegts
                self.gray_all_lbl_file()
                self.lbl_file_apermap.config(bg='yellow')
                self.disable_dependent_btns()
                self.btn_select_slits['state'] = 'normal'
                self.btn_make_apermap_slits['state'] = 'normal'

                #### load Apermap
                #hdul_temp = cached_fits_open(pathname)
                hdul_temp = fits.open(pathname)
                self.data_full = np.float32(hdul_temp[0].data)

                #### show the fits image
                self.clear_image()
                self.update_image(uniform=True)
        else:
            self.data_full = np.ones((4048, 4048), dtype=np.int32)
            self.filename_trace = "apx0000_0000.fits"
            self.file_current = "0000"
            self.lbl_file_apermap['text'] = self.filename_trace.split('.')[0]
            #self.disable_add_slit()
            self.gray_all_lbl_file()
            self.fig.clf()
            self.canvas.draw_idle()

    def init_image1(self):
        # the figure that will contain the plot
        self.fig = Figure(figsize = (6, 8))
        self.fig.clf()

        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame2)  # A tk.DrawingArea.
        self.canvas.draw()

        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack() #grid(row=0, column=0)

        # creating the Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame2)
        self.toolbar.update()

        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().pack() #grid(row=1, column=0)

    def init_image2(self):
        # the figure that will contain the plot
        self.fig2 = Figure(figsize = (6, 8))
        self.fig2.clf()

        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.frame3)  # A tk.DrawingArea.
        self.canvas2.draw()

        # placing the canvas on the Tkinter window
        self.canvas2.get_tk_widget().pack() #grid(row=0, column=0)

        # creating the Matplotlib toolbar
        self.toolbar2 = NavigationToolbar2Tk(self.canvas2, self.frame3)
        self.toolbar2.update()

        # placing the toolbar on the Tkinter window
        self.canvas2.get_tk_widget().pack() #grid(row=1, column=0)

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
                self.ax.imshow(self.data_full, origin='lower', cmap='gray', vmin=np.min(self.data_full), vmax=np.percentile(self.data_full, percent))
            self.ax.set_title("b%s"%self.file_current)
            self.canvas.draw_idle()
        if shoe=='r' or shoe=='both':
            if uniform:
                self.ax2.imshow(self.data_full2, origin='lower', cmap='gray', vmin=np.min(self.data_full2), vmax=np.min(self.data_full2)+1)
            else:
                self.ax2.imshow(self.data_full2, origin='lower', cmap='gray', vmin=np.min(self.data_full2), vmax=np.percentile(self.data_full2, percent))
            self.ax2.set_title("r%s"%self.file_current)
            self.canvas2.draw_idle()

    def plot_curve(self, shoe='both'):
        if shoe=='b' or shoe=='both':
            yy = np.arange(len(self.data_full))
            xx = func_parabola(yy, self.param_curve_b[0], self.param_curve_b[1], self.param_curve_b[2])
            self.ax.plot(xx, yy, 'r--')
            self.update_image(shoe=shoe)
        if shoe=='r' or shoe=='both':
            yy = np.arange(len(self.data_full2))
            xx = func_parabola(yy, self.param_curve_r[0], self.param_curve_r[1], self.param_curve_r[2])
            self.ax2.plot(xx, yy, 'r--')
            self.update_image(shoe=shoe)

    def plot_edges(self, shoe='both'):
        if shoe=='b' or shoe=='both':
            yy = np.arange(len(self.data_full))
            x1 = func_parabola(yy, self.param_curve_b[0], self.param_curve_b[1], self.param_edges_b[0])
            self.ax.plot(x1, yy, 'r--')
            x2 = func_parabola(yy, self.param_curve_b[0], self.param_curve_b[1], self.param_edges_b[1])
            self.ax.plot(x2, yy, 'r--')
            self.update_image()
        if shoe=='r' or shoe=='both':
            yy = np.arange(len(self.data_full2))
            x1 = func_parabola(yy, self.param_curve_r[0], self.param_curve_r[1], self.param_edges_r[0])
            self.ax2.plot(x1, yy, 'r--')
            x2 = func_parabola(yy, self.param_curve_r[0], self.param_curve_r[1], self.param_edges_r[1])
            self.ax2.plot(x2, yy, 'r--')
            self.update_image()

    def pick_points_b(self):
        shoe = 'b'
        self.disable_others()
        self.btn_select_curve_b['state'] = 'active'
        self.cidpick = self.fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click_curve(event, shoe=shoe))
        self.cidexit = self.fig.canvas.mpl_connect('key_press_event', lambda event: self.key_press(event, step='curve'))

    def pick_points_r(self):
        shoe = 'r'
        self.disable_others()
        self.btn_select_curve_r['state'] = 'active'
        self.cidpick = self.fig2.canvas.mpl_connect('button_press_event', lambda event: self.on_click_curve(event, shoe=shoe))
        self.cidexit = self.fig2.canvas.mpl_connect('key_press_event', lambda event: self.key_press(event, step='curve'))

    def pick_edges_b(self):
        shoe = 'b'
        self.disable_others()
        self.btn_select_edges_b['state'] = 'active'

        self.ax.axhline(len(self.data_full)/2, c='g', ls='-')
        self.update_image(shoe=shoe)
        self.cidpick = self.fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click_edges(event, shoe=shoe))
        self.cidexit = self.fig.canvas.mpl_connect('key_press_event', lambda event: self.key_press(event, step='edges'))

    def pick_edges_r(self):
        shoe = 'r'
        self.disable_others()
        self.btn_select_edges_r['state'] = 'active'

        self.ax2.axhline(len(self.data_full2)/2, c='g', ls='-')
        self.update_image(shoe=shoe)
        self.cidpick = self.fig2.canvas.mpl_connect('button_press_event', lambda event: self.on_click_edges(event, shoe=shoe))
        self.cidexit = self.fig2.canvas.mpl_connect('key_press_event', lambda event: self.key_press(event, step='edges'))

    def pick_slits(self):
        self.disable_others()
        self.btn_select_slits['state'] = 'active'

        self.ax.axvline(len(self.data_full[0])/2, c='g', ls='-')
        self.update_image(uniform=True)
        self.cidpick = self.fig.canvas.mpl_connect('button_press_event', self.on_click_slits)
        self.cidexit = self.fig.canvas.mpl_connect('key_press_event', self.key_press_slits)

    def key_press_slits(self, event):
        if event.key == 'escape':
            #### enable other functions
            self.enable_others()
            self.btn_select_slits['state'] = 'normal'
            self.btn_make_apermap_slits['state'] = 'normal'

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
            self.break_mpl_connect()

    def key_press(self, event, step):
        if event.key == 'escape':
            #### enable other functions
            self.enable_others()

            if step=='curve':
                self.btn_select_curve_b['state'] = 'normal'
                self.btn_select_curve_r['state'] = 'normal'
            elif step=='edges':
                self.btn_select_edges_b['state'] = 'normal'
                self.btn_select_edges_r['state'] = 'normal'

            #### break the mpl connection
            self.break_mpl_connect()

    def on_click_slits(self, event):
        if event.button is MouseButton.RIGHT:
            if np.abs(self.y_last-event.ydata)>2:
                self.points.append([event.xdata, event.ydata])
                print(len(self.points), event.xdata, event.ydata)
                self.x_last, self.y_last = event.xdata, event.ydata
                self.ax.scatter(len(self.data_full[0])/2, event.ydata, c='r', marker='x', zorder=10)
                self.update_image(uniform=True)

    def on_click_curve(self, event, shoe):
        if event.button is MouseButton.RIGHT:
            if len(self.points)<6 and (np.abs(self.x_last-event.xdata)>1 or np.abs(self.y_last-event.ydata)>10):
                self.points.append([event.xdata, event.ydata])
                print(len(self.points), event.xdata, event.ydata)
                self.x_last, self.y_last = event.xdata, event.ydata

                if shoe=='b':
                    self.ax.scatter(event.xdata, event.ydata, c='r', marker='x', zorder=10)
                elif shoe=='r':
                    self.ax2.scatter(event.xdata, event.ydata, c='r', marker='x', zorder=10)
                self.update_image(shoe=shoe)

            if len(self.points)==6:
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
                
                self.btn_select_curve_b['state'] = 'normal' 
                self.btn_select_curve_r['state'] = 'normal' 

                #### break the mpl connection
                self.break_mpl_connect()

    def on_click_edges(self, event, shoe):
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
                    self.txt_param_edges_X1_b.set("%.0f"%(self.param_edges_b[0]))
                    self.txt_param_edges_X2_b.set("%.0f"%(self.param_edges_b[1]))
                    self.txt_param_edges_dX_b.set("%.0f"%(self.param_edges_b[2]))
                    self.ent_param_edges_X2_b['textvariable'] = tk.StringVar(value='%.0f'%self.param_edges_b[1])
                elif shoe=='r':
                    self.param_edges_r = np.array([self.points[0][0], self.points[1][0]])
                    self.param_edges_r = np.sort(self.param_edges_r)
                    self.param_edges_r = np.append(self.param_edges_r, self.param_edges_r[1]-self.param_edges_r[0])
                    self.txt_param_edges_X1_r.set("%.0f"%(self.param_edges_r[0]))
                    self.txt_param_edges_X2_r.set("%.0f"%(self.param_edges_r[1]))
                    self.txt_param_edges_dX_r.set("%.0f"%(self.param_edges_r[2]))
                    self.ent_param_edges_X2_r['textvariable'] = tk.StringVar(value='%.0f'%self.param_edges_r[1])

                #### plot the edges
                self.plot_edges(shoe=shoe)

                #### enable other functions
                self.enable_others()
                self.btn_select_edges_b['state'] = 'normal'
                self.btn_select_edges_r['state'] = 'normal'

                #### break the mpl connection
                self.break_mpl_connect()

    def enable_others(self):
        self.btn_folder['state'] = 'normal'
        self.btn_refresh['state'] = 'normal'
        self.ent_folder['state'] = 'normal'
        self.box_files['state'] = 'normal'
        self.shoe1['state'] = 'normal'
        self.shoe2['state'] = 'normal'
        self.pca1['state'] = 'normal'
        self.pca2['state'] = 'normal'

        self.btn_load_curve['state'] = 'normal'
        self.btn_load_edges['state'] = 'normal'
        self.btn_load_trace['state'] = 'normal'
        self.btn_load_pypeit['state'] = 'normal'
        self.btn_load_apermap['state'] = 'normal'
        
        self.btn_update_curve_b['state'] = 'normal'
        self.btn_update_edges_b['state'] = 'normal'
        self.ent_param_curve_A_b['state'] = 'normal'
        self.ent_param_curve_B_b['state'] = 'normal'
        self.ent_param_curve_C_b['state'] = 'normal'
        self.ent_param_edges_X1_b['state'] = 'normal'
        self.ent_param_edges_dX_b['state'] = 'normal'

        self.btn_update_curve_r['state'] = 'normal'
        self.btn_update_edges_r['state'] = 'normal'
        self.ent_param_curve_A_r['state'] = 'normal'
        self.ent_param_curve_B_r['state'] = 'normal'
        self.ent_param_curve_C_r['state'] = 'normal'
        self.ent_param_edges_X1_r['state'] = 'normal'
        self.ent_param_edges_dX_r['state'] = 'normal'

        self.ent_smash_range['state'] = 'normal'

        self.ent_folder_trace['state'] = 'normal'
        self.btn_folder_trace['state'] = 'normal'

    def disable_others(self):
        self.btn_folder['state'] = 'disabled'
        self.btn_refresh['state'] = 'disabled'
        self.ent_folder['state'] = 'disabled'
        self.box_files['state'] = 'disabled'
        self.shoe1['state'] = 'disabled'
        self.shoe2['state'] = 'disabled'
        self.pca1['state'] = 'disabled'
        self.pca2['state'] = 'disabled'

        self.btn_load_curve['state'] = 'disabled'
        self.btn_load_edges['state'] = 'disabled'
        self.btn_load_trace['state'] = 'disabled'
        self.btn_load_pypeit['state'] = 'disabled'
        self.btn_load_apermap['state'] = 'disabled'

        self.btn_update_curve_b['state'] = 'disabled'
        self.btn_update_edges_b['state'] = 'disabled'
        self.ent_param_curve_A_b['state'] = 'disabled'
        self.ent_param_curve_B_b['state'] = 'disabled'
        self.ent_param_curve_C_b['state'] = 'disabled'
        self.ent_param_edges_X1_b['state'] = 'disabled'
        self.ent_param_edges_dX_b['state'] = 'disabled'

        self.btn_update_curve_r['state'] = 'disabled'
        self.btn_update_edges_r['state'] = 'disabled'
        self.ent_param_curve_A_r['state'] = 'disabled'
        self.ent_param_curve_B_r['state'] = 'disabled'
        self.ent_param_curve_C_r['state'] = 'disabled'
        self.ent_param_edges_X1_r['state'] = 'disabled'
        self.ent_param_edges_dX_r['state'] = 'disabled'

        self.ent_smash_range['state'] = 'disabled'

        self.ent_folder_trace['state'] = 'disabled'
        self.btn_folder_trace['state'] = 'disabled'

        self.btn_select_curve_b['state'] = 'disabled'
        self.btn_select_edges_b['state'] = 'disabled'
        self.btn_select_curve_r['state'] = 'disabled'
        self.btn_select_edges_r['state'] = 'disabled'
        self.btn_make_trace['state'] = 'disabled'
        self.btn_make_apermap['state'] = 'disabled'
        self.btn_select_slits['state'] = 'disabled'
        self.btn_make_apermap_slits['state'] = 'disabled'

    def break_mpl_connect(self):
        #### break the mpl connection
        self.curve_points = np.array(self.points)
        self.points = []
        self.x_last, self.y_last = -1., -1.
        self.fig.canvas.mpl_disconnect(self.cidpick)
        self.fig.canvas.mpl_disconnect(self.cidexit)

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
        write_trace_file(self.data_full, self.hdr_c1_b, self.folder_trace, 'b'+self.file_current)
        write_trace_file(self.data_full2, self.hdr_c1_r, self.folder_trace, 'r'+self.file_current)

        #### control widgets
        self.btn_make_trace['state'] = 'disabled'

    def popup_showinfo(self, title, message):
        showinfo(title=title, message=message)


if __name__ == "__main__":
    main()
