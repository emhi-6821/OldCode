import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
#from mpyfit import fit
import pdb
#from functions import 

#import yo shit
data=fits.getdata('spec-0429-51820-0056.fits')
flux=np.array([i[0] for i in data])
wavelength=np.array([10**i[1] for i in data])

#import it again
restwave=np.genfromtxt("linelist-0429-51820-0056.csv", delimiter=',',skip_header=1)
restwave=restwave[:,1]

#plot the data
plt.plot(wavelength, flux)
for i in range(len(restwave)):
    plt.axvline(restwave[i])
plt.xlabel("Wavelength (angstroms)")
plt.ylabel("Flux (Watts/m^2)")
plt.title("Unshifted rest wavelengths on top of galaxy spectrum")
plt.savefig("unshiftedontop.png")
plt.show()
#plot specific area
plt.plot(wavelength, flux)
for i in range(len(restwave)):
    plt.axvline(restwave[i])
plt.xlabel("Wavelength (angstroms)")
plt.ylabel("Flux (Watts/m^2)")
plt.xlim(6400, 7000)
plt.title("Unshifted rest wavelengths group")
plt.savefig("unshift6400.png")
plt.show()


#make array of zeros and make restwave start at same wavelength
restwaveint=[]

shiftedwave=(restwave*.017+restwave)#guess for shifting using visual guess

#plot the correlated emission lines
plt.plot(wavelength, flux)
for i in range(len(shiftedwave)):
    plt.axvline(shiftedwave[i])
plt.xlabel("Wavelength (angstroms)")
plt.ylabel("Flux (Watts/m^2)")
plt.title("shifted rest wavelengths on top of galaxy spectrum")
plt.savefig("shiftedontop.png")
plt.show()
#section
plt.plot(wavelength, flux)
for i in range(len(shiftedwave)):
    plt.axvline(shiftedwave[i])
plt.xlabel("Wavelength (angstroms)")
plt.ylabel("Flux (Watts/m^2)")
plt.xlim(6500, 7000)
plt.title("shifted rest wavelengths groups")
plt.savefig("shifted6500.png")
plt.show()
def gauss(x,params):
    """
    Function will produce a gaussian curve given a set of parameters

    Input
    -----
    x = independent values
    a = a list of background value, center, width, and the height of
        the bottom/top of the boxcar/tophat

    Output
    ------
    f = a set of corresponding dependent values of the gaussian function
    """
    f=(np.exp(-np.power(x-params[0], 2.) / (2. * np.power(params[1], 2.))))/(params[1]*(np.sqrt(2.*np.pi)))*params[2]+params[3]#function for a gaussian

    return f

def minfunc(params,x,data):
    """
    Finds the absolute difference between the data and fitted gaussian

    Input
    -----
    a = parameters to feed to the boxcar function
    args = a tuple of the dependent and independent values

    Output
    ------
    The absolute difference between the dependent values and the gaussian
    function specified by a.
    """
    return data-gauss(x, params)#find difference between gaussian and data

def fit_gauss(params, x, data):
    """
    Fits a transit with a gaussian function

    Input
    -----
    x,f = time and relative flux or flux values

    Output
    ------
    the transit depth, and the fitted gaussian
    """

    fit=leastsq(minfunc, params, args=(x,data), full_output=1)#find the least square or distance
    return fit


def fitt(shiftedwave,restwave, dist, wavelengtharr, fluxarr):
    params=np.array([shiftedwave, 3.5, 500., 300.])#set values of parameters
    x=wavelengtharr[np.where((wavelengtharr>=(shiftedwave-dist)) * (wavelengtharr<=(shiftedwave+dist)))]#where it is between the width parameters
    data=fluxarr[np.where((wavelengtharr>=(shiftedwave-dist)) * (wavelengtharr<=(shiftedwave+dist)))]#same for y
    fitted=fit_gauss(params, x, data)#fit this to a gaussian

    f=gauss(x, fitted[0])#make into gaussian

    z= (fitted[0][0]-restwave)/restwave#find the redshift

    return f,x,z, fitted

sig=[]
z=[]

for i in range(len(restwave)-2):#loop though all of the emission lines minus the first two which are negative
    f,x,z1,fi1=fitt(shiftedwave[i+2], restwave[i+2], 22., wavelength, flux)#use the functions
    sig.append(fi1[0][1])#list of sigmas
    z.append(z1)

redshift=np.array(z)#array of all of the redshifts

sig=np.array(sig)#make an array of sigmas
good=[7,9,12,15,16,17,18,19]#the spots in the spectrum of the galaxy that correlate with an element
redshiftgood=redshift[good]#the redshifts for only to good emission bumps
sigma=sig[good]#the sigmas of only the ones with good emissions

#find the uncertainty
uncert=[]
for i in range(len(good)):#loop through emission lines used
    uncert.append((sigma[i]/restwave[i])**2)#find individual uncertainty
sumof=np.sum(uncert)
sqrtof=np.sqrt(sumof)
uncertainty=sqrtof/len(good)#total uncertainty

shift=np.mean(redshiftgood)#the redshift

print("The redshift is: ", shift, "+/-", uncertainty)


#plot all of the correlations to the rest wavelengths to see which ones exist and not
fig, axarr=plt.subplots(4,3, figsize=(10,10))
e=0

count=0

#loop though the first half because it is hard to see all at once
for i in range(4):
    for j in range(3):
        f,x,z,fitted=fitt(restwave[j+e+2], shiftedwave[j+e+2], 10., wavelength, flux)
        
        axarr[i,j].plot(wavelength, flux)
        axarr[i,j].plot(x,f)
        axarr[i,j].set_xlim(shiftedwave[j+e+2]-20, shiftedwave[j+e+2]+20)
        axarr[i,j].set_title(count)
        axarr[i,j].set_xlabel("Wavelength(angstroms)")
        axarr[i,j].set_ylabel("Flux(watts/m^2)")
        axarr[i,j].xaxis.set_ticks(np.arange(shiftedwave[j+e+2]-20,shiftedwave[j+e+2]+40, 20))
        count+=1
    e+=3
fig.tight_layout()
plt.savefig("all1.png")
plt.show()



fig, axarr=plt.subplots(3,3, figsize=(10,10))
#loop through second half

for i in range(3):
    for j in range(3):
        f,x,z,fitted=fitt(restwave[j+e+2], shiftedwave[j+e+2], 10., wavelength, flux)
        
        axarr[i,j].plot(wavelength, flux)
        axarr[i,j].plot(x,f)
        axarr[i,j].set_xlim(shiftedwave[j+e+2]-20, shiftedwave[j+e+2]+20)
        axarr[i,j].set_title(count)
        axarr[i,j].set_xlabel("Wavelength(angstroms)")
        axarr[i,j].set_ylabel("Flux(watts/m^2)")
        axarr[i,j].xaxis.set_ticks(np.arange(shiftedwave[j+e+2]-20,shiftedwave[j+e+2]+40, 20))
        count+=1
    e+=3
#for label in axarr[2,2].get_xticklabels()[::2]:
 #   label.set_visible(False)#make less tick marks so looks better 
fig.tight_layout()
plt.savefig("all2.png")
plt.show()


uncertlist = []
ilist=[]
uncert=[]
for i in range(len(good)):
    ilist.append(i)
    uncert.append((sigma[i]/restwave[i])**2)#find individual uncertainty      
    sumof2=np.sum(uncert)
    sqrtof=np.sqrt(sumof)
    uncertainty2=sqrtof/len(ilist)#total uncertainty
    uncertlist.append(uncertainty2)

xv = [1, 2, 3, 4, 5, 6, 7, 8]

plt.plot(xv, uncertlist)
plt.xlabel('Number of spectral lines')
plt.ylabel('Uncertainty in Redshift')
plt.title('Uncertainty Variation with Number of Spectral Lines Used')
plt.savefig('uncert.png')
plt.show()
plt.close()


c=300000.0 #km/sec speed of light
v=c*shift
h=71.0 #(km/s)/Mpc
d=v/h
