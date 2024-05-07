import numpy as np
from astropy.nddata import CCDData

import scipy.signal
from collections import Counter
import numpy.polynomial.polynomial as poly

from utils_io import func_parabola

import sys
sys.path.append('../m2fs')
import m2fs_process as m2fs

def load_trace(file_path):
    """Load trace from a fits file using CCDData. """

    trace = CCDData.read(file_path, unit='electron')
    trace.mask = np.zeros_like(trace.data, dtype=bool)
    trace.uncertainty = np.ones_like(trace.data, dtype=float) 

    return trace


def reshape_trace_by_curvature(trace, curve_params):
    """Reshape trace by curvature. """

    # Get the trace data
    trace_data = trace.data

    # Get the trace shape
    trace_shape = trace_data.shape

    # cut the data into the range between two parabolic curves
    data_new = []
    for y_idx in range(trace_shape[0]):
        x_1 = func_parabola(y_idx, curve_params[0], curve_params[1], curve_params[3])
        x_2 = func_parabola(y_idx, curve_params[0], curve_params[1], curve_params[3]+curve_params[4])
        # round x_1 to the lower integer, and x_2 to the higher integer
        x_1 = int(np.floor(x_1))
        x_2 = int(np.ceil(x_2))
        data_new.append(trace_data[y_idx, x_1:x_2])
    data_new = np.array(data_new)

    trace_new =  CCDData(data_new, unit='electron')
    trace_new.mask = np.zeros_like(trace_new, dtype=bool)
    trace_new.uncertainty = np.ones_like(trace_new, dtype=float)

    return trace_new

def determine_signal_height(columnspec_array, min_height=100.):
    """Determine signal height. """

    height_arrary = []
    for column in range(len(columnspec_array)):
        height_arrary.append(max(columnspec_array[column].spec))
    
    cut_height = np.max(height_arrary)/30.

    signal_height = np.max([min_height, cut_height])

    return signal_height

def do_trace(trace, curve_params, trace_params=None, verbose=False):
    """Do trace. """

    if trace_params is None:
        trace_step=20 
        n_lines=11
        columnspec_continuum_rejection_low=-5.
        columnspec_continuum_rejection_high=1.
        columnspec_continuum_rejection_iterations=10
        columnspec_continuum_rejection_order=1
        window=4
        threshold_factor=25.
    else:
        trace_step = trace_params['trace_step']
        n_lines = trace_params['n_lines']
        columnspec_continuum_rejection_low = trace_params['columnspec_continuum_rejection_low']
        columnspec_continuum_rejection_high = trace_params['columnspec_continuum_rejection_high']
        columnspec_continuum_rejection_iterations = trace_params['columnspec_continuum_rejection_iterations']
        columnspec_continuum_rejection_order = trace_params['columnspec_continuum_rejection_order']
        window = trace_params['window']
        threshold_factor = trace_params['threshold_factor']

    columnspec_array = m2fs.get_columnspec(trace, trace_step,n_lines,columnspec_continuum_rejection_low,columnspec_continuum_rejection_high,columnspec_continuum_rejection_iterations,columnspec_continuum_rejection_order,threshold_factor,window)

    col_centers = np.array([np.median(columnspec_array[i].columns) for i in range(len(columnspec_array))])

    #### Find the number of apertures and possible aperture half width
    # find the number of peaks in each column
    signal_height = determine_signal_height(columnspec_array)

    peaks_array = []
    for column in range(len(columnspec_array)):
        spec = columnspec_array[column].spec
        peaks = scipy.signal.find_peaks(spec, height=signal_height)
        peaks_array.append(peaks[0])
    peaks_num = [len(peaks_array[i]) for i in range(len(peaks_array))]

    # find n_aper by getting the most common number in peaks_num except 0
    peaks_num_counter = Counter(peaks_num)
    peaks_num_counter.pop(0, None)
    n_aper = peaks_num_counter.most_common(1)[0][0]

    # find aper_half_width by getting the median of the difference between the peaks
    peaks_diff_array = []
    for i in range(len(peaks_array)):
        if len(peaks_array[i]) == n_aper:
            peaks_diff = np.diff(peaks_array[i])
            peaks_diff_array.append(peaks_diff)
    peaks_diff_array = np.array(peaks_diff_array)
    aper_half_width = int(np.median(peaks_diff_array)/2)

    if verbose:
        print('The number of apertures is: ', n_aper)
        print('The possible aperture half width is: ', aper_half_width)

    #### Find the traces by fitting a second order polynomial
    traces_array = []
    for i, peaks in enumerate(peaks_array):
        if len(peaks) == n_aper:
            traces_col = col_centers[i] + func_parabola(peaks, curve_params[0], curve_params[1], curve_params[3]) 
            traces_val = peaks
            traces_array.append([traces_col, traces_val])
    traces_array = np.array(traces_array)
    
    if verbose:
        print("traces_array.shape: ", traces_array.shape)

    traces_coefs = []
    for  i in range(len(traces_array[0, 0, :])):
        traces_col = traces_array[:, 0, i]
        traces_val = traces_array[:, 1, i]
        temp_coefs = poly.polyfit(traces_col, traces_val, 2)
        traces_coefs.append(temp_coefs)
    traces_coefs = np.array(traces_coefs)

    if verbose:
        print("traces_coefs.shape: ", traces_coefs.shape)

    return traces_array, traces_coefs, n_aper, aper_half_width


def create_apermap(trace, curve_params, traces_coefs, aper_half_width, verbose=False):
    """Create aperture map. """

    aper_map_full = np.zeros_like(trace.data, dtype=np.int32)
    x_middle = int(trace.data.shape[1]/2)
    x_trace = np.arange(trace.data.shape[1])

    y_middle = np.array([], dtype=np.int32)
    for i in range(len(traces_coefs)):
        y_middle = np.append(y_middle, np.round(poly.polyval(x_middle, traces_coefs[i])).astype(np.int32))
        y_trace = poly.polyval(x_trace, traces_coefs[i])
        y_trace = np.round(y_trace).astype(np.int32)
        for j in range(len(y_trace)):
            aper_map_full[y_trace[j]-aper_half_width:y_trace[j]+aper_half_width, x_trace[j]] = i+1

    # trim the aperture map according to the curvature
    aper_map_trim = np.zeros_like(trace.data, dtype=np.int32)
    for y_idx in range(aper_map_full.shape[0]):
        x_1 = func_parabola(y_idx, curve_params[0], curve_params[1], curve_params[3])
        x_2 = func_parabola(y_idx, curve_params[0], curve_params[1], curve_params[3]+curve_params[4])
        x_1 = int(np.floor(x_1))
        x_2 = int(np.ceil(x_2))
        aper_map_trim[y_idx, x_1:x_2] = aper_map_full[y_idx, x_1:x_2]

    return aper_map_trim, y_middle
