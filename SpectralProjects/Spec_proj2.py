# python modules that we will use
import os
import numpy as np
import glob as glob
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d, LSQUnivariateSpline, LSQBivariateSpline
from scipy.optimize import curve_fit, fmin
from scipy.signal import find_peaks

#matplotlib inline
import matplotlib.pylab as plt

from platform import python_version
assert python_version() >= '3.6', "Python version 3.6 or greater is required for f-strings"

import scipy
assert scipy.__version__ >= '1.4', "scipy version 1.4 or higher is required for this module"

# change plotting defaults
plt.rc('axes', labelsize=14)
plt.rc('axes', labelweight='bold')
plt.rc('axes', titlesize=16)
plt.rc('axes', titleweight='bold')
plt.rc('font', family='sans-serif')
plt.rcParams['figure.figsize'] = (10, 5)

# where are the data located?
cwd = os.getcwd() #Current working directory
data_dir = os.path.join(cwd, 'data') #subdirectory with all data

print('--------------------PROCESSING IMAGES-----------------------')

#Load 2D science image
image = fits.getdata(os.path.join(data_dir, 'spec_sci.fits')) #array of the data

def show_image(image, lower=-1, upper=3, extent=None):
    sample = sigma_clip(image)
    vmin = sample.mean() - (lower * sample.std())
    vmax = sample.mean() + (upper * sample.std())
    plt.figure(figsize=(15, 3))
    plt.imshow(image, origin='lower', cmap='gray', aspect='auto', vmin=vmin, vmax=vmax, extent=extent)
    plt.xlabel('Column Number')
    plt.ylabel('Row Number');
    plt.close()

#size of image
ny, nx = image.shape
midy, midx = ny//2, nx//2 #center pixel coordinate
#1D array of all possible x & y values
xs = np.arange(nx)
ys = np.arange(ny)
#pixel coordinates for each pixel
yvals, xvals = np.indices(image.shape)

#show arc lamp image to map known wavelengths
lamp_image = fits.getdata(os.path.join(data_dir, 'spec_lamp.fits'))#image of arc lamps(mostly neon and argon)
show_image(lamp_image)
#load linelist which shows list of known wavelengths from the arc lamps
dtype = [('wav', float), ('id', 'U2')]
linelist = np.genfromtxt(os.path.join(data_dir, 'line_list.dat'), dtype=dtype)
linelist.sort(order='wav')

row1 = midy - 5
row2 = midy + 5
lamp_spec = lamp_image[row1:row2, :].mean(axis=0)#compute mean along the 0 axis(rows), resulting in array size(1, 4026)
plt.plot(lamp_spec, c='g')
plt.xlabel('Column Number')
plt.ylabel('Counts');

peaks, prop = find_peaks(lamp_spec, height=(5000, 60000), prominence = 200)
plt.plot(lamp_spec, c='g')
plt.scatter(peaks, lamp_spec[peaks], marker = 'x', c='k')
plt.xlabel('Column Number')
plt.ylabel('Counts')
plt.close()

#create gaussian function to model spectral lines
def gaussian(x, *params):
    amp, x0, sigma = params
    return amp * np.exp(-(x - x0)**2 / 2 / sigma**2)

#function to return precise column center for each line
def get_lamp_lines(lamp_spec, h_min, h_max, prominence = 200):
    peaks, properties = find_peaks(lamp_spec, height=(h_min, h_max), prominence=prominence)
    dtype = []
    dtype.append(('col', float))
    dtype.append(('counts', float))
    dtype.append(('x', float))
    dtype.append(('y', float))
    dtype.append(('sigma', float))
    dtype.append(('id', 'U2'))
    dtype.append(('wav', float))
    dtype.append(('wavres', float))
    dtype.append(('used', bool))
    lamp_lines = np.zeros(peaks.size, dtype=dtype)
    lamp_lines['col'] = peaks
    lamp_lines['counts'] = lamp_spec[peaks]
    lamp_lines['x'] = np.nan
    lamp_lines['y'] = np.nan
    lamp_lines['sigma'] = np.nan
    lamp_lines['wav'] = np.nan
    lamp_lines['wavres'] = np.nan
    
    #fit peaks to determine more precise center
    cols = np.arange(lamp_spec.size)
    sigma_guess = 2.5
    for line in lamp_lines:
        i0 = max(0, int(line['col'])-5)
        i1 = min([lamp_spec.size -1, int(line['col'] + 5)])
        guess = (line['counts'], line['col'], sigma_guess)
        bounds = ((0, line['col'] - 3, 0), (np.inf, line['col'] + 3, np.inf))
        try:
            popt, pcov = curve_fit(gaussian, cols[i0:i1], lamp_spec[i0:i1], p0=guess, bounds=bounds)
            #popt=optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized
            #pcov=The estimated covariance of popt.
        except RuntimeError:
            #curve fit failed to converge-skip
            continue
        
        line['x'] = popt[1]
        line['y'] = gaussian(popt[1], *popt)
        line['sigma'] = popt[2]
        
    fitted = np.isfinite(lamp_lines['x']) #mark all x values in lamp lines with bool value(True if fitted, False if not fitted)
    lamp_lines = lamp_lines[fitted] #make lamp_lines into an array with only the lines that were fit
    print('Precise fitted %s lines' % (lamp_lines.size))
    return lamp_lines
    
lamp_lines = get_lamp_lines(lamp_spec, 5000, 60000)

    
def mark_peaks(lamp_lines, xtype='x', ytype='y', c='k'):
    plt.scatter(lamp_lines['x'], lamp_lines['y'], marker='x', c='k' )
    
mark_peaks(lamp_lines)
plt.plot(lamp_spec)
plt.xlabel('Column')
plt.ylabel('Counts')
plt.title('Gaussian Fitted Column Centers for Spectral Lines')
plt.close()

#Begin working to identify each line in the spectra
#Load a spectral atlas to use as reference for identifying lines
#this code uses the wavelength calibrated Keck/LIRS spectral atlas
lamp_ref = np.genfromtxt(os.path.join(data_dir, 'lamp_reference.dat'), names='wav, counts')
plt.plot(lamp_ref['wav'], lamp_ref['counts'], c='r')
plt.xlabel('Wavelength ($\AA$)')
plt.ylabel('Counts')
plt.title('Keck/LIRS Spectral Atlas')
plt.close()

print('----------------------BEGINNING WAVELENGTH CALIBRATIONS-------------')

#define function to match a line to a list
#listing=list of kown spectra lines and their wavelengths, value=values of lines in column format to match with known lamp lines, plt=choose what plotting tool to use
def match_list(listing, values, plt=None, tol=None, revcoeff=None, c='k'):
    matched= []
    cols = []
    for value in values:
        absdiff = np.abs(value - listing)
        index = np.argmin(absdiff)
        if tol is None:
            bestmatch = listing[index]
        elif absdiff[index] < tol:
            bestmatch = listing[index]
        else:
            bestmatch = np.nan
        matched.append(bestmatch)
        
        if plt is not None:
            plt.axvline(bestmatch, ls='dotted', c=c)
            
        if revcoeff is not None:
            col = np.polyval(revcoeff, bestmatch)
            cols.append(col)
            print(f"{bestmatch:.1f} is expected near column {col:.0f}")

    if revcoeff is not None:
        return np.array(matched), cols
    
    return np.array(matched)
    
wav1 = 5700
wav2 = 7500

# plot the reference spectrum in red
plt.plot(lamp_ref['wav'], lamp_ref['counts'], label='reference', c='r')
plt.xlim(wav1, wav2)
plt.xlabel('Wavelength ($\AA$)')
plt.ylabel('Counts'); plt.grid()
plt.close()

# mark all wavelengths available in linelist
#for row in linelist:
#plt.axvline(row['wav'], ls='dashed', c='0.8')

# pick a few lines in this plot
rough_waves = [6144, 6320, 6400, 6680] # <---- add a few values here
refwavs = match_list(linelist['wav'], rough_waves, plt=plt)

# find a section of lamp_spec that looks similar to the section of spectra atlas that your plotted
col1 = 300
col2 = 1800
plt.plot(lamp_spec, c='g')
plt.xlim(col1, col2)
plt.xlabel('Column Number'); plt.ylabel('Counts'); plt.grid();
plt.close()

# mark lines with Gaussian-fit centers
mark_peaks(lamp_lines)

# record the rough column numbers of the same lines as above in the same order
rough_cols = [670, 833, 892, 1130] # <---- add rough column values here
refcols = match_list(lamp_lines['x'], rough_cols, plt=plt)

def set_line_identity(lamp_lines, linelist, x, wav):
    # find the closest matching lamp_line
    ilamp = np.argmin(np.abs(lamp_lines['x'] - x))
    # find the closest matching row in the linelist
    ilist = np.argmin(np.abs(linelist['wav'] - wav))
    # reset values in lamp_lines
    lamp_lines[ilamp]['id'] = linelist[ilist]['id']
    lamp_lines[ilamp]['wav'] = linelist[ilist]['wav']
    
# record ids and wavelengths of matched lines
for col, wav in zip(refcols, refwavs):
    set_line_identity(lamp_lines, linelist, col, wav)

# this routine finds the polynomial coefficients needed to transform between column numbers and wavelengths (and vice versa). Outlier rejection is included.
def get_wavelength_solution(lamp_lines, order=4):
    wfit = np.isfinite(lamp_lines['wav']) #boolean value array with true for all lines that have been mapped previouslt manually(with a value in the 'wav' column
    
    # define the reverse mapping (wavelength to column)- revcoeff
    revcoeff = np.polyfit(lamp_lines['wav'][wfit], lamp_lines['x'][wfit], order)
    # define the forward mapping (column to wavelength)- coeff
    coeff = np.polyfit(lamp_lines['x'][wfit], lamp_lines['wav'][wfit], order)
    
    # check the fit for outliers
    fit_wav = np.polyval(coeff, lamp_lines['x'])
    wavres = fit_wav - lamp_lines['wav']
    lamp_lines['wavres'] = wavres
    sample = wavres[wfit]
    sample.sort()
    sample = sample[int(0.1 * sample.size) : int(0.9 * sample.size + 0.5)]
    std = np.std(sample, ddof=1)
    w = wfit
    w[wfit] = (np.abs(lamp_lines['wavres'][wfit]) < (5 * std))
    if w.sum() != lamp_lines.size:
        # re-fit with outliers rejected
        coeff, revcoeff = get_wavelength_solution(lamp_lines[w], order=order)
        
        # reset wavelength residuals using new coefficients
        fit_wav = np.polyval(coeff, lamp_lines['x'])
        wavres = fit_wav - lamp_lines['wav']
        lamp_lines['wavres'] = wavres
        
    lamp_lines['used'] = w
    return coeff, revcoeff
        
def check_wavelength_solution(lamp_spec, lamp_lines, coeff):
    wavs = col_to_wav(coeff, np.arange(lamp_spec.size))
    plt.plot(wavs, lamp_spec, c='g', lw=2)
    mark_peaks(lamp_lines, 'wav')
    plt.colorbar(label='Residual ($\AA$)')
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('Counts')
    plt.grid()
    
def col_to_wav(coeff, cols):
    return np.polyval(coeff, cols)

def wav_to_col(revcoeff, wavs):
    return np.polyval(revcoeff, wavs)

def mark_matched(lamp_lines):
    for line in lamp_lines:
        plt.axvline(line['wav'], ls='dotted', c='k')
        
# estimate a linear relation between column number and wavlength
coeff, revcoeff = get_wavelength_solution(lamp_lines, order=1)

# apply the wavelength solution to the column numbers
wavs = col_to_wav(coeff, np.arange(lamp_spec.size))


# plot the initial wavelength calibrated spectrum in green
#plt.plot(wavs, lamp_spec, c='g', lw=2, label='lamps')
#plt.xlim(wav1, wav2)
# plot the reference spectrum in red
#plt.plot(lamp_ref['wav'], lamp_ref['counts'], label='reference', c='r')
#plt.xlim(wav1, wav2); plt.title('Wavelength Calibrated Lamp Spectrum on top of Spectra atlas')
#plt.xlabel('Wavelength ($\AA$)'); plt.ylabel('Counts'); plt.grid();
#mark_matched(lamp_lines)
#plt.close()


# check for more matches in the range already fit
def match_more(lamp_lines, linelist, order=4, tol=2):
    coeff, revcoeff = get_wavelength_solution(lamp_lines, order=order)
    wfit = np.isfinite(lamp_lines['wav'])
    minwav = lamp_lines['wav'][wfit].min()
    maxwav = lamp_lines['wav'][wfit].max()
    
    xmin = lamp_lines['x'][wfit].min()
    xmax = lamp_lines['x'][wfit].max()
    
    w = (lamp_lines['x'] > xmin) & (lamp_lines['x'] < xmax)
    for line in lamp_lines[w]:
        rough_wav = col_to_wav(coeff, line['x'])
        refwav = match_list(linelist['wav'], [rough_wav], tol=tol)
        if np.isfinite(refwav):
            #print(f'matched column {line["x"]:.1f} to wavelength {refwav[0]}')
            set_line_identity(lamp_lines, linelist, line['x'], refwav)
            
match_more(lamp_lines, linelist, order=1)

# re-fit with a higher order
coeff, revcoeff = get_wavelength_solution(lamp_lines, order=4)
check_wavelength_solution(lamp_spec, lamp_lines, coeff)
plt.show()
