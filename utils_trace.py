import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt

from astropy.nddata import CCDData

from utils_io import func_parabola
from columnspec import get_columnspec


def load_trace(file_path):
    """Load trace from a fits file using CCDData. """

    trace = CCDData.read(file_path, unit='electron')
    trace.mask = np.zeros_like(trace.data, dtype=bool)
    trace.uncertainty = np.ones_like(trace.data, dtype=float) 

    # get the ifu type from the header
    ifu_type = trace.meta['IFU']
    naxis2 = trace.meta['NAXIS2']
    bin_y = 4112./naxis2

    return trace, ifu_type, bin_y


def reshape_trace_by_curvature(trace, curve_params):
    """Reshape trace by curvature. """

    # Get the trace data
    trace_data = trace.data

    # Get the trace shape
    trace_shape = trace_data.shape

    # cut the data into the range between two parabolic curves
    data_new = []
    for y_idx in range(trace_shape[0]):
        x_1 = func_parabola(y_idx, curve_params[0], 
                            curve_params[1], curve_params[3])
        # round x_1 to the lower integer, and x_2 to the higher integer
        x_1 = int(np.rint(x_1))
        x_2 = x_1 + int(curve_params[4])
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


def preanalyze_columnspec_array(columnspec_array, ifu_type):
    """Preanalyze columnspec array. """

    # get the inital peaks and properties from columnspec_array
    peaks_diff_array = np.array([])
    for column in range(len(columnspec_array)):
        spec = columnspec_array[column].spec
        peaks = signal.find_peaks(spec, prominence=0, width=0)
        peaks_diff_array = np.append(peaks_diff_array, np.diff(peaks[0]))

    # get the aperture half width
    aper_half_width = int(np.median(peaks_diff_array)/2)

    # get the width and distance cut
    width_cut = int( aper_half_width - 1 )
    #width_cut = int( aper_half_width * 0.5 )
#    if ifu_type == 'LSB' or ifu_type == 'STD':
#        width_cut = int( aper_half_width - 2 )
#    else:   # if ifu_type == 'HR' or ifu_type == 'STD'
#        width_cut = int( aper_half_width - 1 )
#
    distance_cut = int( aper_half_width * 1.8 )

    # get the prominence cut
    # prominences_array_sub = prominences_array[
    #     prominences_array < np.percentile(prominences_array, 95)
    #     ]

    # kde = stats.gaussian_kde(prominences_array_sub)
    # x = np.linspace(0, np.max(prominences_array_sub), 1000)
    # y = kde(x)
    # dips, _ = signal.find_peaks(-y)
    # prominence_cut = x[dips[0]]
    prominence_cut = 20

    return aper_half_width, width_cut, distance_cut, prominence_cut


def get_peaks_in_one_column(columnspec_array, column, distance, prominence, 
                            width, rel_height=0.5):
    """Get peaks in one column. """

    spec = columnspec_array[column].spec
    _, properties = signal.find_peaks(spec, distance=distance, 
                                      prominence=prominence, width=width, 
                                      rel_height=rel_height)
    
    # use the center of left_ips and right_ips as the peak
    peaks_left = properties['left_ips']
    peaks_right = properties['right_ips']
    peaks = (peaks_left + peaks_right)/2

    return peaks


def get_peaks_array(columnspec_array, distance, prominence, width, 
                    rel_height=0.5, verbose=False):
    """Get peaks array. """

    peaks_array = []
    for column in range(len(columnspec_array)):
        peaks = get_peaks_in_one_column(
            columnspec_array, column, distance, prominence, width, rel_height
            )
        peaks_array.append(peaks)

    if verbose:
        print("---- Initial peaks array")

        n_min, n_max = len(peaks_array[0]), len(peaks_array[0])
        for i in range(1, len(peaks_array)):
            n_min = np.min([n_min, len(peaks_array[i])])
            n_max = np.max([n_max, len(peaks_array[i])])
        print("---- n_min, n_max: ", n_min, n_max)

    return peaks_array


def align_peaks_array(peaks_array_raw, verbose=False):
    """Align peaks into apertures. """

    peaks_col0 = peaks_array_raw[0] 
    peaks_array = np.array([peaks_col0], dtype=float).reshape(1, -1)

    # add right column one-by-one to the peaks_array
    for col in range(1, len(peaks_array_raw)):
        #if col > 2:
        #    break

        # get the peaks in the left column
        peaks_left = peaks_array[-1]
        len_left = len(peaks_left)

        # get the peaks in the right column
        peaks_right = peaks_array_raw[col]
        len_right = len(peaks_right)

        # align the peaks in the left and right columns
        i_left = 0
        i_right = 0
        peaks_array_left = []
        peaks_array_right = []
        while i_left < len_left and i_right < len_right:
            diff_temp = np.abs(peaks_left[i_left]) \
                        - np.abs(peaks_right[i_right])
            # if the absolute difference is less than 3, belong to the
            # same aperture
            if np.abs(diff_temp) < 3:
                peaks_array_left.append(peaks_array[:, i_left])
                peaks_array_right.append(peaks_right[i_right])
                i_left += 1
                i_right += 1
            else:
                # if the absolute difference is larger than 3,
                # and the difference is negative, add one element
                # to the right column; otherwise, add one element
                # to the left column
                # Note that added element is negative
                if diff_temp < 0:
                    peaks_array_left.append(peaks_array[:, i_left])
                    peaks_array_right.append(
                        -np.abs(peaks_array[-1, i_left])
                        )
                    i_left += 1
                else:
                    peaks_array_left.append(
                        np.zeros(len(peaks_array))
                        - np.abs(peaks_right[i_right])
                        )
                    peaks_array_right.append(peaks_right[i_right])
                    i_right += 1

        # add the remaining elements to the peaks_array
        if i_left < len_left:
            for i in range(i_left, len_left):
                peaks_array_left.append(peaks_array[:, i])
                peaks_array_right.append(-np.abs(peaks_array[-1, i]))
                i_left += 1

        # merge the peaks_array_left and peaks_array_right
        # to the peaks_array
        peaks_array_left = np.transpose(np.array(peaks_array_left))
        peaks_array_right = np.array(peaks_array_right)
        n_temp = len(peaks_array_left)+1
        peaks_array = np.append(
            peaks_array_left, peaks_array_right
            ).reshape(n_temp, -1)

    if verbose:
        print("---- After aligning the peaks into apertures.")
        print("---- peaks_array.shape: ", peaks_array.shape)

    return peaks_array


def clean_peaks_array(peaks_array, percentile=0.2, verbose=False):
    """
    Clean peaks array by 
    1) removing short traces, and 
    2) NaN rows in the max column.
    """
    # change all negative values to nan
    peaks_array[peaks_array < 0] = np.nan

    # remove short traces
    row_ids = []
    for i in range(len(peaks_array[0])):
        y = peaks_array[:, i]
        mask_nan = np.isnan(y)
        if np.sum(mask_nan) > percentile*len(y):
            row_ids.append(i)
    peaks_array = np.delete(peaks_array, row_ids, axis=1)

    # find the column with the maximum number of peaks
    n_peaks = np.array( 
        [np.sum(~np.isnan(peaks_array[i])) for i in range(len(peaks_array))] 
        )
    # column_max = np.argmax(n_peaks)
    column_max_all = np.where(n_peaks == np.max(n_peaks))[0]
    if len(column_max_all) > 1:
        # if there are multiple columns with the same number of peaks,
        # take the one with the median value
        column_max = np.median(column_max_all).astype(int)
    else:
        # otherwise, take the column with the maximum number of peaks
        column_max = np.argmax(n_peaks)

    # remove nan rows in column max
    mask_nan = np.isnan(peaks_array[column_max])
    peaks_array = peaks_array[:, ~mask_nan]

    if verbose:
        print("---- After cleaning the peaks array.")
        print("---- peaks_array.shape: ", peaks_array.shape)
        print("---- column_max: ", column_max)

    return peaks_array, column_max


def get_group_gaps_from_template(peaks_template, peak_diff_cut=1.5, 
                                 verbose=False):
    """Get the group gaps from a IFU template file. """

    # get the difference between the peaks
    diff_template = np.diff(peaks_template, axis=0)
    med_diff_template = np.median(diff_template, axis=0)

    # get the group gaps
    mask = diff_template > med_diff_template*peak_diff_cut
    gap_template = diff_template[mask]

    # append the first and last element of peaks_template to peaks_gap_template
    peaks_gap_template = peaks_template[:-1][mask]
    peaks_gap_template = np.append(peaks_template[0], peaks_gap_template)
    peaks_gap_template = np.append(peaks_gap_template, peaks_template[-1])

    min_diff_gap = np.min(np.diff(peaks_gap_template))

    if verbose:
        print("---- peaks_gap_template: ", 
              len(peaks_gap_template), peaks_gap_template)
        print("---- min_diff_gap: ", min_diff_gap)

    return peaks_gap_template, min_diff_gap


def get_group_gaps_from_column_max(peaks_cmax, min_diff_gap, 
                                   peak_diff_cut=1.5, verbose=False):
    """Get the group gaps from the column max. """

    # get the difference between the peaks
    diff_cmax = np.diff(peaks_cmax)
    med_diff_cmax = np.median(diff_cmax)

    # get the group gaps
    mask_cmax = diff_cmax > med_diff_cmax*peak_diff_cut
    peaks_gap_cmax = peaks_cmax[:-1][mask_cmax]
    peaks_gap_cmax = np.append(peaks_cmax[0], peaks_gap_cmax)
    peaks_gap_cmax = np.append(peaks_gap_cmax, peaks_cmax[-1])

    # remove fake gaps by mering the gaps that are too close
    diff_gap_cmax = np.diff(peaks_gap_cmax)
    i = 0
    while i < len(diff_gap_cmax):
        if diff_gap_cmax[i] < min_diff_gap*0.9:
            peaks_gap_cmax = np.delete(peaks_gap_cmax, i+1)
            diff_gap_cmax = np.diff(peaks_gap_cmax)
            if verbose:
                print("---- i: ", i, diff_gap_cmax)
        else:
            i += 1
    
    if verbose:
        print("---- peaks_gap_cmax: ", len(peaks_gap_cmax), peaks_gap_cmax)

    return peaks_gap_cmax


def fit_template_to_column_max(peaks_gap_template, peaks_gap_cmax, 
                               peaks_template, order=4, verbose=False):
    """Fit projection from template to column max. """

    # fit the projection
    coefs = poly.polyfit(peaks_gap_template, peaks_gap_cmax, order)
    ffit = poly.Polynomial(coefs)

    peaks_template_cmax = ffit(peaks_template) 

    if verbose:
        print("coefs: ", coefs)

    return coefs, peaks_template_cmax


def find_missing_fibers(y_data, y_pred, delta_y=3.0, verbose=False):
    """find missing fibers in the column max. """

    if verbose:
        print("!!!! Finding missing fibers for HR or STD")
        print("!!!! # of default fibers: ", len(y_pred))
        n_missing = len(y_pred) - len(y_data)
        print("!!!! # of missing fibers: ", n_missing)
        print("!!!! cut of delta_y: ", delta_y)

    # count_add = 0
    # count_del = 0
    # ids = []
    # y_new = np.zeros_like(y_pred)
    # for i_pred in range(len(y_pred)):
    #     i_data = i_pred-count_add+count_del
    #     if np.abs(y_data[i_data] - y_pred[i_pred]) > delta_y:
    #         y_new[i_pred] = y_pred[i_pred]
    #         count_add += 1
    #         ids.append(i_pred+1)

    #         if verbose:
    #             print('!!!! Missing %d: %.2f - %.2f = %.2f'%
    #                   (i_pred+1, y_data[i_data], y_pred[i_pred],
    #                    y_data[i_data]-y_pred[i_pred])
    #                    )
    #     else:
    #         y_new[i_pred] = y_data[i_data]

    # rewrite by comparing diff array
    diff_y_pred = np.diff(y_pred, axis=0)
    diff_y_data = np.diff(y_data, axis=0)

    ids = []
    y_data_temp = y_data.copy()
    for i_pred in range(len(y_pred)-1):
        if i_pred >= len(diff_y_data):
            break
        elif np.abs(diff_y_data[i_pred] - diff_y_pred[i_pred]) > delta_y:
            if verbose:
                print('!!!! Missing %d: %.2f - %.2f = %.2f'%
                      (i_pred+2, diff_y_data[i_pred], diff_y_pred[i_pred],
                       diff_y_data[i_pred]-diff_y_pred[i_pred])
                       )
            y_data_temp = np.insert(
                y_data_temp, i_pred+1, 
                np.abs(y_data_temp[i_pred]) + diff_y_pred[i_pred])
            diff_y_data = np.diff(y_data_temp, axis=0)
            ids.append(i_pred+2)
            i_pred -= 1  # adjust the index since we added an element

    # if the last element is missing, add it
    if (len(diff_y_pred) - len(diff_y_data) == 1):
        if verbose:
            print('!!!! Missing %d (the last fiber)' % (len(y_pred)))
        ids.append(len(y_pred))
    elif (len(diff_y_pred) - len(diff_y_data) > 1):
        print('!!!! Warning: possible missing fiber at the start!')

        # add the missing fibers to the end
        for i in range(len(diff_y_pred) - len(diff_y_data) ):
            ids.append(len(y_data_temp) + i + 1)

    return ids


def add_missing_fibers(peaks_array, peaks_template, ids,
                       order=4, verbose=False):
    """Add missing fibers to peaks array. """

    # add the missing fibers to the peaks_array
    peaks_array_new = peaks_array.copy()
    for id in ids:
        peaks_array_new = np.insert(peaks_array_new, id-1, np.nan, axis=1)

    # fill the missing fibers by fitting with the template
    for i in range(peaks_array_new.shape[0]):
        y = peaks_array_new[i]
        mask_nan = np.isnan(y)
        y = y[~mask_nan]
        x = peaks_template[~mask_nan]
        coefs = poly.polyfit(x, y, order)
        ffit= poly.Polynomial(coefs)
        yfit = ffit(peaks_template)
        peaks_array_new[i][mask_nan] = yfit[mask_nan]

    if verbose:
        print("---- peaks_array_new.shape: ", peaks_array_new.shape)

    return peaks_array_new


def find_missing_fibers_LSB(y_data, y_template, 
                           rel_delta_y=1.5, verbose=False):
    """Add missing fibers to peaks array for LSB. """

    if verbose:
        print("!!!! Finding missing fibers for LSB")
        print("!!!! # of default fibers: ", len(y_template))
        n_missing = len(y_template) - len(y_data)
        print("!!!! # of missing fibers: ", n_missing)

    # analyze peaks_template
    diff_template = np.diff(y_template, axis=0)
    max_diff_template = np.max(diff_template, axis=0)

    # find the missing fibers
    diff_data = np.diff(y_data)
    med_diff_data = np.median(diff_data)

    if verbose:
        print("!!!! med_diff_data:", max_diff_template)
        print("!!!! cut: ", max_diff_template*rel_delta_y)

    # add negative values to gaps in y_data
    y_data_new = y_data.copy()
    idx, cts = 0, 0
    while idx < len(y_data)-1:
        temp_diff = np.abs(y_data_new[idx+cts+1]) - np.abs(y_data_new[idx+cts])
        if temp_diff > max_diff_template*rel_delta_y:
            if verbose:
                print("!!!! :", idx+1, temp_diff)

            y_data_new = np.insert(
                y_data_new, 
                idx+cts+1, 
                -(np.abs(y_data_new[idx+cts]) + med_diff_data)
                )
            cts += 1
        else:
            idx += 1

    # if the last element is missing, add it
    # Note that the last element is not checked in the while loop
    while len(y_data_new) < len(y_template):
        y_data_new = np.append(
            y_data_new,
            -(np.abs(y_data_new[-1]) + med_diff_data)
            )

    # get the added ids
    ids = np.where(y_data_new < 0)[0] + 1

    return ids 


def plt_gaps(peaks_cmax, peaks_template, ids_add, shoe, ifu_type):
    """Plot a view of gaps of the column max vs. the template. """

    x_gap_template = np.arange(len(peaks_template)-1)+1
    diff_template = np.diff(peaks_template)

    x_gap_cmax = np.arange(len(peaks_cmax)-1)+1
    diff_cmax = np.diff(peaks_cmax)

    # plot the gaps
    fig = plt.figure(figsize=(12, 6))
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(x_gap_template, diff_template, 'ko', label='template')
    ax.plot(x_gap_cmax, diff_cmax, 'rx', label='current data')

    if ids_add is not None:
        for i, id in enumerate(ids_add):
            # add a vertical dashed line to the plot
            if i==0:
                ax.axvline(x=id, color='b', linestyle='--', 
                           label='added fibers')
            else:
                ax.axvline(x=id, color='b', linestyle='--')

    ax.legend(loc='upper right')
    ax.text(0.05, 0.95, '%s_%s'%(shoe.lower(), ifu_type.upper()), fontsize=20,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)
    ax.set_xlabel('Gap #')
    ax.set_ylabel('Gap size (pixels)')

    # tight the figure
    plt.tight_layout()
    plt.show()

    # save the plot
    #plt.savefig('gaps.png')


def fit_aperture_traces(peaks_array, col_centers, curve_params, 
                        order=4, verbose=False):
    """Fit aperture traces. """

    # transfer the peaks_array back up to the original data coordinates
    traces_array = []
    for column in range(len(peaks_array)):
        x_temp = func_parabola(peaks_array[column], curve_params[0], 
                               curve_params[1], curve_params[3])
        traces_temp = col_centers[column] + x_temp
        traces_array.append(traces_temp)
    traces_array = np.array(traces_array)

    # fit the traces
    traces_coefs = []
    for  i in range(len(peaks_array[0])):
        traces_col = traces_array[:, i]
        traces_val = peaks_array[:, i]
        temp_coefs = poly.polyfit(traces_col, traces_val, order)
        traces_coefs.append(temp_coefs)
    traces_coefs = np.array(traces_coefs)    

    return traces_array, traces_coefs


def do_trace_v2(trace, curve_params, 
                shoe, ifu_type, bin_y,
                trace_params=None, verbose=False, plot=True):
    """Do trace. """

    # get the columnspec array from the trace data
    if trace_params is None:
        trace_step=20 
        n_lines=11
    else:
        trace_step = trace_params['trace_step']
        n_lines = trace_params['n_lines']

    columnspec_array = get_columnspec(trace, trace_step, n_lines)

    col_centers = np.array(
        [np.median(columnspec_array[i].columns) 
         for i in range(0, len(columnspec_array))]
        )

    # pre-analyze the trace data
    aper_half_width, width_cut, distance_cut, prominence_cut \
        = preanalyze_columnspec_array(columnspec_array, ifu_type)
    print("++++ aper_half_width: ", aper_half_width)
    print("++++ width_cut: ", width_cut)
    print("++++ distance_cut: ", distance_cut)
    print("++++ prominence_cut: ", prominence_cut)

    rel_height_cut = 0.5
    print("++++ rel_height_cut: ", rel_height_cut)

    # get the inital peaks array
    peaks_array_raw = get_peaks_array(columnspec_array, distance_cut,
                                        prominence_cut, width_cut, 
                                        rel_height=rel_height_cut, 
                                        verbose=True)

    # get the aligned peaks array
    peaks_array = align_peaks_array(peaks_array_raw, verbose=True)

    # clean the peaks array
    peaks_array, column_max = clean_peaks_array(peaks_array, verbose=True)
    peaks_cmax = peaks_array[column_max]

    # get the peaks template
    template_path = './template_files/template_%s_%s.txt'%(ifu_type, shoe)
    if ifu_type == 'HR':
        bin_y_template = 1.0
    else:   # ifu_type == 'STD' or ifu_type == 'LSB'
        bin_y_template = 2.0
    bin_ratio = bin_y_template/bin_y
    peaks_template = np.loadtxt(template_path)*bin_ratio

    # find the missing fibers
    if ifu_type == 'HR' or ifu_type == 'STD':
        # get the group gaps from the template
        peaks_gap_template, min_diff_gap \
            = get_group_gaps_from_template(peaks_template, verbose=True)

        # get the group gaps from the column max
        peaks_gap_cmax \
            = get_group_gaps_from_column_max(peaks_cmax, min_diff_gap, 
                                             verbose=True)

        # transfer the template to the column max
        coefs, peaks_template_cmax \
            = fit_template_to_column_max(peaks_gap_template, peaks_gap_cmax,
                                         peaks_template, order=4)

        # find the missing fibers
        ids_add = find_missing_fibers(peaks_cmax, peaks_template_cmax,
                                      delta_y=aper_half_width*2.0, verbose=True)
    else:   
        # i.e., ifu_type == 'LSB'
        # LSB has no distingushed group gaps, so use the template directly
        ids_add = find_missing_fibers_LSB(peaks_cmax, peaks_template, 
                                          rel_delta_y=1.5, verbose=True)
    print("++++ # of added fibers: ", len(ids_add))
    print("++++ Added IDs: ", ids_add)

    # add missing fibers to the peaks array
    peaks_array = add_missing_fibers(peaks_array, peaks_template, 
                                     ids_add, verbose=True)

    # plot a view of gaps of the column max vs. the template
    if plot:
        plt_gaps(peaks_array[column_max], peaks_template, ids_add, 
                 shoe, ifu_type)

    # fit the aperture traces
    traces_array, traces_coefs \
        = fit_aperture_traces(peaks_array, col_centers, curve_params)
    n_aper = len(traces_array[0])

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