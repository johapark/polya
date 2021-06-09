#!/usr/bin/env python3
"""utils.py: Collection of useful utility functions"""

__author__ = "Joha Park"
__all__ = ['smooth','savitzky_golay','FSAAnalyzer', 'binning']

# Built-in
from collections import Counter
from math import factorial
import numbers

# Third-party 
from scipy.signal import find_peaks_cwt
from scipy.interpolate import interp1d
import numpy as np

# Local 
from .io import ABIFReader


def binning(vmin, vmax, binsize=10, cap=True):
    if (vmax - vmin) < binsize:
        raise ValueError("binsize is bigger than the range (vmax - vmin)")
    if binsize <= 0:
        raise ValueError("binsize must be greater than 0")

    edges = [vmin]
    edge = vmin
    while edge < vmax:
        edge += binsize 
        edges.append(edge)

    if cap is True:
        edges.pop()
        edges.append(vmax)
    return edges


# from http://wiki.scipy.org/Cookbook/SignalSmooth
def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='same')

    return y[window_len:-window_len+1]

# from http://wiki.scipy.org/Cookbook/SavitzkyGolay
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))

    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])

    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def extrap1d(interpolator):
    """1-d extrapolation"""

    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        if isinstance(xs, numbers.Number):
            return pointwise(xs)
        elif len(xs) == 1:
            return pointwise(xs)
        else:
            return np.array(list(map(pointwise, np.array(xs ))))

    return ufunclike


class FSAAnalyzer(object):

    DATA_SECTION = 'DATA'

    MARKER_SIZES = {
        'genescan500liz': [35, 50, 75, 100, 139, 150, 160, 200, 250, 300,
                           340, 350, 400, 450, 490, 500],
    }

    def __init__(self, filename, marker_lane=105, marker='genescan500liz',
                 smoothing_method='peakmedian'):
        self.filename = filename
        self.data = dict()
        self.marker_lane = marker_lane
        self.marker_peaks = None

        if isinstance(marker, str):
            self.marker_sizes = self.MARKER_SIZES[marker]
        elif isinstance(marker, list) and all([isinstance(x, numbers.Number) for x in marker]):
            self.marker_sizes = sorted(marker) # for custom size markers

        self.intrapolator = None
        self.smoothing_method = smoothing_method
        self.reader = ABIFReader(filename)
        #self.prepare_control()

    def load(self, lane):
        """Load the data by lane as a numpy array."""
        data = np.array(self.reader.getData(self.DATA_SECTION, lane))

        if self.data is None:
            self.data = {lane: data}
        else: self.data[lane] = data

        return

    def get(self, lane):
        """Get signals intensities along the fragment size estimates."""
        func_ext = self._fit_markers()

        try: data = self.data[lane]
        except KeyError:
            self.load(lane)
            data = self.data[lane]

        return func_ext(np.arange(len(data))), data

    ## Deprecated for manual correction of the peak positions
    #@property 
    #def marker_peaks(self):
    #    return self._find_marker_peaks()

    def get_marker_peaks(self, force=False, **kwargs):
        if (self.marker_peaks is None) or (force is True):
            self.marker_peaks = self._find_marker_peaks(**kwargs)

        return self.marker_peaks

    def add_marker_peaks(self, positions):
        '''Manually add marker positions.'''
        if self.marker_peaks is None:
            raise ValueError("Marker peaks should be automatically detected first before manually corrected.")

        peak_positions = sorted(list(set(self.marker_peaks).union(set(positions))))
        self.marker_peaks = np.array(peak_positions)

        return

    def remove_marker_peaks(self, positions):
        '''Manually add marker positions.'''
        if self.marker_peaks is None:
            raise ValueError("Marker peaks should be automatically detected first before manually corrected.")

        peak_positions = sorted(list(set(self.marker_peaks).difference(set(positions))))
        self.marker_peaks = np.array(peak_positions)

        return


    def _find_marker_peaks(self, widths_for_cwt=np.arange(10,40), min_snr=1.5, noise_perc=1,
                                height_filter=True, height_filter_tolerance_factor=0.3):
        try: marker_data = self.data[self.marker_lane]
        except KeyError:
            self.load(self.marker_lane)
            marker_data = self.data[self.marker_lane]

        peak_positions = find_peaks_cwt(marker_data, widths_for_cwt, min_snr=min_snr, noise_perc=noise_perc)

        if height_filter is True:
            p = height_filter_tolerance_factor # tolerance scaling factor; (1-p) ~ (1+p)
            peak_heights = marker_data[peak_positions]
            median_height = np.median(peak_heights)
            valid_peak_indices = np.where( ( median_height * (1-p) < peak_heights ) & ( median_height * (1+p) > peak_heights ) )
            peak_positions =  peak_positions[valid_peak_indices]

        return peak_positions

    def _fit_markers(self):
        func = self._interpolate_marker_peaks()
        self.intrapolator = func
        func_ext = extrap1d(func)

        return func_ext

    def _interpolate_marker_peaks(self):
        pairs = list(zip(self.marker_peaks[::-1], self.marker_sizes[::-1]))
        x = sorted([p[0] for p in pairs])
        y = sorted([p[1] for p in pairs])

        return interp1d(x, y)


if __name__ == "__main__":
    pass
