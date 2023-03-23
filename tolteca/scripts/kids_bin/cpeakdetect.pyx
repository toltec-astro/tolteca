#!/usr/bin/env python

#cython: language_level=3

cimport cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
def peakdetect(
        np.ndarray[np.double_t, ndim=1] y_axis,
        int lookahead=200, double delta=0.0,
        int offset_for_height=0,
        double min_height=0,
        int exclude_edge=0,
        ):

    cdef int length = y_axis.shape[0]
    cdef np.ndarray x_axis = np.array(range(length), dtype=int)

    cdef int index = 0, mxpos = 0, mnpos = 0
    cdef double mn = np.Inf, mx = -np.Inf, y

    cdef list max_peaks = []
    cdef list min_peaks = []
    cdef list dump = []   # Used to pop the first hit which almost always is false


    # perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    # Only detect peak if there is 'lookahead' amount of points after it
    for index in range(length - lookahead):
        y = y_axis[index]
        x = x_axis[index]

        if y > mx:
            # print(f"index={index} update mx mxpos")
            mx = y
            mxpos = x
        if y < mn:
            # print(f"index={index} update mn mnpos")
            mn = y
            mnpos = x

        # look for max
        if y < mx-delta and mx != np.Inf:
            # print(f"index={index} y < mx")
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                # print(f"index={index} {index}:{index+lookahead} max < mx")
                max_peaks.append([mxpos, mx])
                dump.append(True)
                # set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break
                continue
        else:
            pass
            # print(f"index={index} y={y} mx={mx}")

        # look for min
        if y > mn+delta and mn != -np.Inf:
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                # set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break

    # Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        # no peaks were found, should the function return empty lists?
        pass

    # calculate peak props

    # we label the data first
    cdef np.ndarray labx = np.zeros_like(x_axis)
    cdef int count = 1

    cdef list idx_valleys = []
    idx_valleys.append(0)
    for index in range(len(min_peaks)):
        idx_valleys.append(min_peaks[index][0])
    idx_valleys.append(length - 1)

    for index in range(len(idx_valleys) - 1):
        if idx_valleys[index] != idx_valleys[index + 1]:
            labx[idx_valleys[index]: idx_valleys[index +1] + 1] = count
            count = count + 1

    # for each segment, calcualte indices and derive peak height
    # left index, right index, offset left index, offset right index,
    cdef int label_id, il, ip, i0=0, i1=0, i2=length-1, i3=length-1
    cdef double yp, y0, y1, y2, y3,
    cdef h_offset = offset_for_height
    if h_offset <= 0:
        h_offset = lookahead
    cdef np.ndarray heights = np.zeros_like(y_axis, dtype=np.double)
    cdef double height
    cdef int height_good=0, pos_good=0
    cdef np.ndarray peak_is_good = np.zeros(len(max_peaks), dtype=bool)
    cdef list peaks_out = []
    cdef list peaks_good_out = []
    # cdef double ymx =np.Inf, y1=-np.Inf
    for index in range(len(max_peaks)):
        ip = max_peaks[index][0]
        yp = max_peaks[index][1]
        label_id = labx[ip]
        print(f"peak at ip={ip} label_id={label_id} yp={yp}")
        i0, i3 = np.where(labx == label_id)[0][[0, -1]]
        # remove edge point as those can be bad
        if i0 < 1:
            i0 = 1
        if i3 > length - 2:
            i3 = length - 2
        i1 = ip - h_offset
        i2 = ip + h_offset
        if i1 < i0:
            i1 = i0
        if i2 > i3:
            i2 = i3
        print(f"peak info h_offset={h_offset} i0={i0} i1={i1} i2={i2} i3={i3} ip={ip}")
        # calculate height
        y0 = y_axis[i0]
        y1 = y_axis[i1]
        y2 = y_axis[i2]
        y3 = y_axis[i3]
        height = yp - 0.5 * (y1 + y2)
        heights[index] = height
        print(f"y0={y0} y1={y1} y2={y2} y3={y3} yp={yp} min_height={min_height} height={height}")

        if height < min_height:
            height_good = 0
        else:
            height_good = 1

        # check if close to edge
        if ip < exclude_edge or ip >= length - exclude_edge:
            pos_good = 0
        else:
            pos_good = 1
        peaks_out.append((ip, yp, height, i0, i1, i2, i3))
        if pos_good > 0 and height_good > 0:
            peaks_good_out.append((ip, yp, height, i0, i1, i2, i3))
            peak_is_good[index] = True
    return peaks_out, labx, peaks_good_out, peak_is_good

    # for index in range(len(max_peaks)):
    #     # reject edge
    #     pidx = max_peaks[index][0]
    #     if pidx < exclude_edge or pidx >= length - exclude_edge:
    #         # print(f"reject pidx {pidx}")
    #         peak_good[index] = False
    #         continue
    #     # reject by height
    #     label_id = labx[pidx]
    #     # print(f"peak at pidx={pidx} label_id={label_id}")
    #     for index2 in np.where(labx == label_id)[0]:
    #         if index2 + fwhm_offset == pidx:
    #             i0 = index2
    #         if pidx + fwhm_offset == index2:
    #             i1 = index2
    #         # print(f"index2={index2} fwhm_offset={fwhm_offset} i0={i0} i1={i1}")
    #     for index2 in range(i0, i1 + 1):
    #         if y0 > y_axis[index2]:
    #             y0 = y_axis[index2]
    #         if y1 < y_axis[index2]:
    #             y1 = y_axis[index2]
    #     height = max_peaks[index][1] - 0.5 * (y1 + y0)
    #     # print(f"y1={y1} y0={y0} height={height} min_height={min_height}")
    #     if height < min_height:
    #         peak_good[index] = False
    #         continue
    #     peak_good[index] = True



    # check height with respect to min height

    # print(f"detected peaks: {max_peaks}")
    # now filter the height if requested
   # cdef np.ndarray peak_good = np.ones((len(max_peaks, )), dtype=bool)
    # if min_height <= 0:
    #     return labx, max_peaks, peak_good

    # cdef list max_peaks_out = []

    # # for each peak, get the label, and extract the data and do the check
    # # here we also check the edget

    # return labx, max_peaks, peak_good
