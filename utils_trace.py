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

def _determine_signal_height(columnspec_array, min_height=100.):
    """Determine signal height. """

    height_arrary = []
    for column in range(len(columnspec_array)):
        height_arrary.append(max(columnspec_array[column].spec))
    
    cut_height = np.max(height_arrary)/30.

    signal_height = np.max([min_height, cut_height])

    return signal_height


def _preanalyze_columnspec_array(columnspec_array, ifu_type):
    """Preanalyze columnspec array. """

    # get the inital peaks and properties from columnspec_array
    peaks_diff_array = np.array([])
    for column in range(len(columnspec_array)):
        spec = columnspec_array[column].spec
        _, properties = signal.find_peaks(
            spec, prominence=10, width=1, rel_height=0.5)
        peaks_left = properties['left_ips']
        peaks_right = properties['right_ips']
        peaks = (peaks_left + peaks_right)/2
        peaks_diff_array = np.append(peaks_diff_array, np.diff(peaks))

    # get the aperture half width
    aper_half_width = int(np.median(peaks_diff_array)/2)

    # get the width and distance cut
    width_cut = int( aper_half_width - 2 )
    #width_cut = int( aper_half_width * 0.5 )
#    if ifu_type == 'LSB' or ifu_type == 'STD':
#        width_cut = int( aper_half_width - 2 )
#    else:   # if ifu_type == 'HR' or ifu_type == 'STD'
#        width_cut = int( aper_half_width - 1 )
#
    distance_cut = int( width_cut * 2.0 )
    #distance_cut = int( aper_half_width * 2 )

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


def _get_peaks_in_one_column(columnspec_array, column, distance, prominence, 
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


def _get_peaks_array(columnspec_array, distance, prominence, width, 
                    rel_height=0.5, verbose=False):
    """Get peaks array. """

    peaks_array = []
    for column in range(len(columnspec_array)):
        peaks = _get_peaks_in_one_column(
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


def _align_peaks_array(peaks_array_raw, verbose=False):
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


def _clean_peaks_array(peaks_array, percentile=0.2, verbose=False):
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


def _get_group_gaps_from_template(peaks_template, peak_diff_cut=1.5, 
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


def _get_group_gaps_from_column_max(peaks_cmax, min_diff_gap, 
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


def _fit_template_to_column_max(peaks_gap_template, peaks_gap_cmax, 
                               peaks_template, order=4, verbose=False):
    """Fit projection from template to column max. """

    # fit the projection
    coefs = poly.polyfit(peaks_gap_template, peaks_gap_cmax, order)
    ffit = poly.Polynomial(coefs)

    peaks_template_cmax = ffit(peaks_template) 

    if verbose:
        print("coefs: ", coefs)

    return coefs, peaks_template_cmax


def _find_missing_fibers(y_data, y_pred, delta_y=3.0, verbose=False):
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


def _add_missing_fibers(peaks_array, peaks_template, ids,
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


def _find_missing_fibers_LSB(y_data, y_template, 
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


def _plt_gaps(peaks_cmax, peaks_template, ids_add, shoe, ifu_type):
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


def _fit_aperture_traces(peaks_array, col_centers, curve_params, 
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
        mask_good = (~np.isnan(traces_val)) & (traces_val > 0)
        temp_coefs = poly.polyfit(traces_col[mask_good], traces_val[mask_good], order)
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
        = _preanalyze_columnspec_array(columnspec_array, ifu_type)
    print("++++ aper_half_width: ", aper_half_width)
    print("++++ width_cut: ", width_cut)
    print("++++ distance_cut: ", distance_cut)
    print("++++ prominence_cut: ", prominence_cut)

    rel_height_cut = 0.5
    print("++++ rel_height_cut: ", rel_height_cut)

    # get the inital peaks array
    peaks_array_raw = _get_peaks_array(
        columnspec_array, distance_cut, prominence_cut, width_cut, 
        rel_height=rel_height_cut, verbose=True)

    # get the aligned peaks array
    peaks_array = _align_peaks_array(peaks_array_raw, verbose=True)

    # clean the peaks array
    peaks_array, column_max = _clean_peaks_array(peaks_array, verbose=True)
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
            = _get_group_gaps_from_template(peaks_template, verbose=True)

        # get the group gaps from the column max
        peaks_gap_cmax \
            = _get_group_gaps_from_column_max(peaks_cmax, min_diff_gap, 
                                             verbose=True)

        # transfer the template to the column max
        coefs, peaks_template_cmax \
            = _fit_template_to_column_max(peaks_gap_template, peaks_gap_cmax,
                                         peaks_template, order=4)

        # find the missing fibers
        ids_add = _find_missing_fibers(peaks_cmax, peaks_template_cmax,
                                      delta_y=aper_half_width*2.0, verbose=True)
    else:   
        # i.e., ifu_type == 'LSB'
        # LSB has no distingushed group gaps, so use the template directly
        ids_add = _find_missing_fibers_LSB(peaks_cmax, peaks_template, 
                                          rel_delta_y=1.5, verbose=True)
    print("++++ # of added fibers: ", len(ids_add))
    print("++++ Added IDs: ", ids_add)

    # add missing fibers to the peaks array
    peaks_array = _add_missing_fibers(peaks_array, peaks_template, 
                                     ids_add, verbose=True)

    # plot a view of gaps of the column max vs. the template
    if plot:
        _plt_gaps(peaks_array[column_max], peaks_template, ids_add, 
                 shoe, ifu_type)

    # fit the aperture traces
    traces_array, traces_coefs \
        = _fit_aperture_traces(peaks_array, col_centers, curve_params)
    n_aper = len(traces_array[0])

    return traces_array, traces_coefs, n_aper, aper_half_width


def _plot_columnspec(columnspec_array, col):
    """Plot the columnspec of one column. """

    spec = columnspec_array[col].spec
    pixel = columnspec_array[col].pixel

    fig = plt.figure(figsize=(12, 6))
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(pixel, spec, 'k-')

    ax.set_xlim(0, len(spec)/4)
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Counts')
    ax.set_title('Columnspec of column %d'%col)

    # tight the figure
    plt.tight_layout()
    plt.show()


def _plot_spec_window(spec, rel_height, col_num, flag_success, 
                     idx_min_left=None, idx_min_right=None, idx_max=None):
    fig = plt.figure(0, figsize=(22,6))
    fig.clf()
    ax=fig.add_subplot(111)
    ax.plot(np.arange(len(spec)), spec, 'kx-', lw=1.5, label='Column Spectrum')
    ax.axhline(y=np.max(spec)*rel_height, color='gray', linestyle='--', label='Relative Height')

    if flag_success:
        ax.axvline(x=idx_min_left, color='r', linestyle='--', label='Window Lower')
        ax.axvline(x=idx_min_right, color='g', linestyle='--', label='Window Upper')
        ax.axvline(x=idx_max, color='b', linestyle='--', label='Peak 1')
    ax.legend()

    ax.set_xlabel('Pixel')
    ax.set_ylabel('Counts')
    ax.set_title(f'Columnspectrum of the {col_num} column')
    ax.legend()
    plt.tight_layout()
    plt.show()


def _plot_first_peaks(peaks1, mask_good, col_num):
    """
    plot the first peaks
    """
    excluded_columns = []   # stores indices of selected (right-clicked) points
    selecting_mode = [False]  # toggled with 's' / 'Esc'

    fig = plt.figure(figsize=(12,6))
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_title("Press 'Shift+S' to enter selecting mode, 'Esc' to quit selecting mode")
    ax.set_xlabel('Column Number')
    ax.set_ylabel('Pixel Position')

    xx = np.arange(len(peaks1))
    ax.plot(xx[peaks1>0], peaks1[peaks1>0], 'k+', label="All first peaks")
    ax.plot(xx[peaks1<0], -peaks1[peaks1<0], 'g+', label="Failed peaks")
    ax.plot(xx[mask_good], peaks1[mask_good], 'r+', label="Selected (good) peaks")

    ax.axvline(x=col_num, color='gray', linestyle='--')

    # Scatter for highlighted (selected) points – starts empty
    highlight = ax.scatter([], [], s=60, facecolors="none", edgecolors="red",
                           linewidths=2, zorder=5, label="Excluded peaks")

    # Status text in the top-left corner
    status_text = ax.text(0.01, 0.97, "Mode: NORMAL",
                          transform=ax.transAxes, va="top",
                          fontsize=10, color="grey")
    
    def _refresh_highlight():
        """Redraw the red circles on every currently selected point."""
        if excluded_columns:
            idx = np.array(excluded_columns)
            highlight.set_offsets(np.column_stack([idx, peaks1[idx]]))
        else:
            highlight.set_offsets(np.empty((0, 2)))
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "S":
            selecting_mode[0] = True
            status_text.set_text("Mode: SELECTING  (right-click to pick, Esc to exit)")
            status_text.set_color("red")
            fig.canvas.draw_idle()
        elif event.key == "escape":
            selecting_mode[0] = False
            status_text.set_text("Mode: NORMAL")
            status_text.set_color("grey")
            fig.canvas.draw_idle()

    def on_click(event):
        """Right-click in selecting mode → find nearest point and toggle selection."""
        if not selecting_mode[0]:
            return
        if event.button != 3:          # 3 = right mouse button
            return
        if event.inaxes != ax:
            return
        if event.xdata is None:
            return

        # Find the nearest index in the array
        nearest_idx = int(np.round(np.clip(event.xdata, 0, len(peaks1) - 1)))

        if nearest_idx in excluded_columns:
            excluded_columns.remove(nearest_idx)   # toggle off
        else:
            excluded_columns.append(nearest_idx)   # toggle on

        excluded_columns.sort()  # keep sorted for easier debugging

        print(f"excluded_columns = {excluded_columns}")
        _refresh_highlight()

    cid_key   = fig.canvas.mpl_connect("key_press_event", on_key)
    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    return excluded_columns


def _plot_peaks_array(peaks_array, mask_good, col_num):
    """ 
    plot peaks_array for all columns
    """

    excluded_columns = []   # stores indices of selected (right-clicked) points
    selecting_mode = [False]  # toggled with 's' / 'Esc'
    vlines = {} # to store vertical line artists for easy updating

    fig = plt.figure(6, figsize=(10,10))
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_title("Press 'Shift+S' to enter selecting mode, 'Esc' to quit selecting mode")
    ax.set_xlabel('Column Number')
    ax.set_ylabel('Pixel Position')

    for col in range(len(peaks_array)):
        if mask_good[col]:
            ax.plot(np.zeros(len(peaks_array[col]))+col, peaks_array[col], 
                    marker='o', linestyle='None')
            
    ax.axvline(x=col_num, color='gray', linestyle='--', label='Middle Column')

    # Status text in the top-left corner
    status_text = ax.text(0.01, 0.97, "Mode: NORMAL",
                          transform=ax.transAxes, va="top",
                          fontsize=10, color="grey")
    
    def on_key(event):
        if event.key == "S":
            selecting_mode[0] = True
            status_text.set_text("Mode: SELECTING  (right-click to pick, Esc to exit)")
            status_text.set_color("red")
            fig.canvas.draw_idle()
        elif event.key == "escape":
            selecting_mode[0] = False
            status_text.set_text("Mode: NORMAL")
            status_text.set_color("grey")
            fig.canvas.draw_idle()

    def on_click(event):
        """Right-click in selecting mode → find nearest point and toggle selection."""
        if not selecting_mode[0]:
            return
        if event.button != 3:          # 3 = right mouse button
            return
        if event.inaxes != ax:
            return
        if event.xdata is None:
            return

        # Find the nearest index in the array
        nearest_idx = int(np.round(np.clip(event.xdata, 0, len(mask_good) - 1)))

        if nearest_idx in excluded_columns:
            # Deselect: remove from list and erase the vline
            excluded_columns.remove(nearest_idx)
            vlines[nearest_idx].remove()
            del vlines[nearest_idx]
        else:
            # Select: add to list and draw a vertical red line
            excluded_columns.append(nearest_idx)
            vl = ax.axvline(x=nearest_idx, color="red", linewidth=1.2, alpha=0.7)
            vlines[nearest_idx] = vl

        excluded_columns.sort()  # keep sorted for easier debugging

        print(f"excluded_columns = {excluded_columns}")
        fig.canvas.draw_idle()

    cid_key   = fig.canvas.mpl_connect("key_press_event", on_key)
    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    return excluded_columns


def _get_one_fiber_window(center, width, ratio):
    """
    Calculate the window for one fiber
        center: the center position of the fiber
        width: the total width of the fiber
        ratio: the scaling factor applied to the width.
    """
    lower = int(center - 0.5 * width * ratio)
    upper = int(center + 0.5 * width * ratio)
    return lower, upper


def _find_one_peak(spec, spec_max, peak_init, col_num, rel_height=0.25, 
                   plot=False, verbose=False):
    """find only one peak manually"""

    try:
        if np.max(spec) < spec_max * rel_height:
            print("Spectrum in this window is too low, skipping...")
            return None
    except:   
        return None

    spec_norm = spec / np.max(spec)
    idx_peak0 = int(peak_init)

    idx_min_left = np.argmin(spec_norm[:idx_peak0+1])
    idx_min_right = idx_peak0 + np.argmin(spec_norm[idx_peak0:])

    spec_norm_min = np.max([spec_norm[idx_min_left],spec_norm[idx_min_right]])

    while spec_norm_min > rel_height:
        rel_height += 0.05 
        if rel_height > 0.95:
            break

    if verbose:
        print('    Adopted Relative Height:', rel_height)

    try:
        # get the left and right positions when spec_norm crosses rel_height
        mask_temp = spec_norm[idx_min_left:idx_peak0+1] >= rel_height
        idx_left = idx_min_left + np.where(mask_temp)[0][0]

        mask_temp = spec_norm[idx_peak0:idx_min_right+1] <= rel_height
        idx_right = idx_peak0 + np.where(mask_temp)[0][0]

        if verbose:
            print(f'    idx_left, idx_right: {idx_left}, {idx_right}')

        # from spec_norm[idx_left-1:idx_left] intepolate x position where spec_norm > rel_height
        if spec_norm[idx_left-1] < spec_norm[idx_left]:
            x_left = np.interp(rel_height, spec_norm[idx_left-1:idx_left+1], [idx_left-1, idx_left])

        if spec_norm[idx_right-1] > spec_norm[idx_right]:
            x_right = np.interp(-rel_height, -spec_norm[idx_right-1:idx_right+1], [idx_right-1, idx_right])

        x_center = 0.5 * (x_left + x_right)

        if idx_left and idx_right:
            if plot:
                _plot_spec_window(
                    spec, rel_height, col_num, True, 
                    idx_min_left=x_left, idx_min_right=x_right, idx_max=x_center)
            return x_center
    except Exception as e:
        print("Error:", e)

        if plot:
            _plot_spec_window(spec, rel_height, col_num, False)

        return None


def _find_all_first_peaks(columnspec_array, med_dif_pos_model, 
                          rel_thresh=0.3, rel_width_max=1.5, rel_height=0.25):
    # Find all first peaks in columnspec_array

    peaks1 = []

    for col in range(len(columnspec_array)):
        pixel = columnspec_array[col].pixel.value
        spec = columnspec_array[col].spec
        spec_max = np.max(spec)

        # initial guess of the first peak position
        mask_thresh = spec / np.max(spec) > rel_thresh
        spec_diff = np.append(0, np.diff(spec))
        mask_minus = spec_diff < 0

        try:
            peak1 = np.where(mask_thresh & mask_minus)[0][0] - 1
        except:
            peak1 = 0

        # refine the peak position
        lower, upper = _get_one_fiber_window(
            peak1, med_dif_pos_model, rel_width_max)
        
        peak_find = _find_one_peak(
            spec[lower:upper], spec_max, peak1-lower, col, 
            rel_height=rel_height, plot=False, verbose=False)
        
        if peak_find is not None:
            peak1 = pixel[lower]+peak_find
            peaks1.append(peak1)
        else:
            print("Fail to find peak automatically, roll back to initial value:", col)
            peaks1.append(np.nan)

            # if col>0:
            #     return (peaks1)

    return np.array(peaks1)


def _find_all_peaks_in_one_column(columnspec_array, col_num, peak1, 
                                  pos_model, dif_pos_model, med_dif_pos_model,
                                  rel_width_max):
    # get the spec data of the column
    pixel = columnspec_array[col_num].pixel.value
    spec = columnspec_array[col_num].spec

    spec_max = np.max(spec)

    # find peaks one-by-one
    peaks_init = np.zeros_like(pos_model, dtype=float)
    peaks_init[0] = peak1

    peaks = np.zeros_like(pos_model, dtype=float)
    mask_bad = np.zeros_like(pos_model, dtype=bool)
    peaks[0] = peak1

    count_missing = 0
    fid_missing = []
    for i in range(1, len(dif_pos_model)):
        peak_init = np.abs(peaks[i-1]) + dif_pos_model[i]
        peaks_init[i] = peak_init

        lower, upper = _get_one_fiber_window(
            peak_init, med_dif_pos_model, rel_width_max)
        peak_find = _find_one_peak(
            spec[lower:upper], spec_max, peak_init-lower, col_num, 
            rel_height=0.25, plot=False)

        if peak_find is not None:
            peaks[i] = pixel[lower] + peak_find
        else:
            peaks[i] = peak_init
            count_missing += 1
            fid_missing.append(i+1)
            mask_bad[i] = True
    fid_missing = np.array(fid_missing)
    print("Missing Peaks:", count_missing)
    print("Missing Fiber IDs:", fid_missing)
    print(f"Find {np.sum(peaks > 0)} out of {len(peaks)}")

    return peaks, mask_bad


def _find_peaks_in_next_column(peaks_prev, columnspec_array, col_num, mask_bad,
                               peak1_offset, dif_pos_peaks,
                               med_dif_pos_model, rel_width_max):
    # use the peaks of template column to get peaks to the next column
    # dif_pos_peaks = np.append(0, np.diff(peaks_prev))

    # find peaks one-by-one
    pixel_next = columnspec_array[col_num].pixel.value
    spec_next = columnspec_array[col_num].spec
    spec_next_max = np.max(spec_next)

    peaks_next_init = np.zeros_like(peaks_prev, dtype=float)
    peaks_next = np.zeros_like(peaks_prev, dtype=float)

    count_missing = 0
    fid_missing = []
    for i in range(0, len(peaks_prev)):
        if i == 0:
            peak_next_init = np.abs(peaks_prev[0]) + peak1_offset
        else:
            peak_next_init = np.abs(peaks_next[i-1]) + dif_pos_peaks[i]
        # peak_next_init = np.abs(peaks_prev[i]) + peak1_offset
        peaks_next_init[i] = peak_next_init
        peaks_next[i] = peak_next_init

        if mask_bad[i]:
            continue

        lower, upper = _get_one_fiber_window(
            peak_next_init, med_dif_pos_model, rel_width_max)
        peak_find = _find_one_peak(
            spec_next[lower:upper], spec_next_max, peak_next_init-lower, col_num, 
            rel_height=0.25, plot=False)

        if peak_find is not None:
            peaks_next[i] = pixel_next[lower] + peak_find
        else:
            count_missing += 1
            fid_missing.append(i+1)

    fid_missing = np.array(fid_missing)
    print("Working on column:", col_num)
    print("    Extra Missing Peaks:", count_missing)
    print("    Extra Missing Fiber IDs:", fid_missing)
    print(f"    Find {np.sum(peaks_next > 0)} out of {len(peaks_next)}")

    #_plot_spec(pixel_next, spec_next, peaks_next_init, peaks_next, col_num-1)

    return peaks_next

def do_trace_v3(trace, curve_params, 
                shoe, ifu_type, bin_y,
                trace_params=None, verbose=False, plot=True):
    """Do trace. """

    # fine-tune parameters for finding peaks
    rel_thresh=0.3
    rel_width_max=1.5
    rel_height=0.25
    
    # Step 0: load in fiber model
    path_model = f'./fiber_positions_250804/{ifu_type}_LoRes_Fiber_Position.txt'
    fiber_model = np.loadtxt(path_model, dtype='float', skiprows=4)

    print('fiber_model:', fiber_model.shape)

    # get the median offset
    if shoe == 'b':
        pos_model = fiber_model[:, 2] / bin_y
    else:
        pos_model = fiber_model[:, 5] / bin_y

    dif_pos_model = np.append(0, np.diff(pos_model))
    med_dif_pos_model = np.median(dif_pos_model)
    print('med_dif_pos_model:', med_dif_pos_model)

    # Step 1: get the columnspec array from the trace data
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
  
    ## plot the columnspec of the middle column
    if plot:
        # col_temp = np.argmax([len(peaks_array_raw[i]) for i in range(len(peaks_array_raw))])
        col_temp = len(columnspec_array) // 2
        # _plot_columnspec(columnspec_array, col_temp)

    # Step 1.5: pre-analyze the trace data
    n_col = len(columnspec_array)
    col_num = n_col // 2

    aper_half_width, width_cut, distance_cut, prominence_cut \
        = _preanalyze_columnspec_array(columnspec_array, ifu_type)
    print("++++ aper_half_width: ", aper_half_width)
    print("++++ width_cut: ", width_cut)
    print("++++ distance_cut: ", distance_cut)
    print("++++ prominence_cut: ", prominence_cut)

    # Step 2: find all first peaks in the columnspec_array
    peaks1 = _find_all_first_peaks(columnspec_array, med_dif_pos_model)

    ## mask out bad first peaks based on the difference between the peaks
    d1_peaks1 = np.append(0, np.diff(peaks1))
    d2_peaks1 = np.append(0, np.diff(peaks1[::-1]))[::-1]
    mask_good = (np.abs(d1_peaks1) < 2.0/bin_y) | (np.abs(d2_peaks1) < 2.0/bin_y)
    mask_good[0] = False # first peak always bad
    mask_good[-1] = False # last peak always bad
    # mask_good[:16] = False # first 16 fibers always bad
    print("---- Automatic selection of first peaks:", np.sum(mask_good), "out of", len(peaks1))

    excluded_columns = _plot_first_peaks(peaks1, mask_good, col_num)
    mask_good[excluded_columns] = False
    print("---- Manually excluded columns:", excluded_columns)
    print("---- Final selection of first peaks:", np.sum(mask_good), "out of", len(peaks1))

    # Step 3: find all peaks in the middle column
    ## get the peak of the middle column
    peak1 = peaks1[col_num]
    print(f'---- Working on column {col_num} out of {n_col}')

    peaks, mask_bad = _find_all_peaks_in_one_column(
        columnspec_array, col_num, peak1, pos_model, dif_pos_model, 
        med_dif_pos_model, rel_width_max)
    # print("Peaks in the middle column:", peaks)
    print("---- Bad/missing fibers in the middle column:", np.where(mask_bad)[0]+1)

    dif_pos_peaks_mid = np.append(0, np.diff(peaks))

    # Step 4: find the peaks in the next column based on the peaks in the middle column, and so on
    peaks_array = np.zeros((len(columnspec_array), len(peaks)), dtype=float)
    peaks_array[col_num] = peaks

    ## find peaks for all columns to the left of col_num
    for col in range(col_num-1, 0, -1):
        if mask_good[col]:
            col_prev = col + 1
            while mask_good[col_prev] == False and col_prev < len(columnspec_array)-1:
                col_prev += 1

            if col_prev < len(columnspec_array):
                peaks_temp = _find_peaks_in_next_column(
                    peaks_array[col_prev], columnspec_array, col, mask_bad,
                    peak1_offset=peaks1[col]-peaks1[col_prev], 
                    dif_pos_peaks=dif_pos_peaks_mid,
                    med_dif_pos_model=med_dif_pos_model, 
                    rel_width_max=rel_width_max)
                peaks_array[col] = peaks_temp
            else:
                print("No good previous column found for column:", col)

    ## find peaks for all columns to the right of col_num
    for col in range(col_num+1, len(columnspec_array)-1):
        if mask_good[col]:
            col_prev = col - 1
            while mask_good[col_prev] == False and col_prev > 0:
                col_prev -= 1

            if col_prev > 0:
                peaks_temp = _find_peaks_in_next_column(
                    peaks_array[col_prev], columnspec_array, col, mask_bad,
                    peak1_offset=peaks1[col]-peaks1[col_prev], 
                    dif_pos_peaks=dif_pos_peaks_mid,
                    med_dif_pos_model=med_dif_pos_model, 
                    rel_width_max=rel_width_max)
                peaks_array[col] = peaks_temp
            else:
                print("No good previous column found for column:", col)

    excluded_columns = _plot_peaks_array(peaks_array, mask_good, col_num)
    mask_good[excluded_columns] = False
    peaks_array[~mask_good] = np.nan
    print("---- Manually excluded columns after checking peaks array:", excluded_columns)
    print("---- Final selection of columns after checking peaks array:", np.sum(mask_good), "out of", len(peaks1))

    # Step 5: fit the aperture traces
    traces_array, traces_coefs \
        = _fit_aperture_traces(peaks_array, col_centers, curve_params)
    n_aper = len(traces_array[0])

    return traces_array, traces_coefs, n_aper, aper_half_width
