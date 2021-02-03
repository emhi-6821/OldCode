#import needed packages for image reduction

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
import glob
import os
import subprocess
import warnings
import numpy as np
import matplotlib.pyplot as plt
import photutils
import pyregion
import shutil
import sep

##------------------TEST DEPENDENCIES--------------------##

#print('---------------------TESTING DEPENDENCIES----------------------')

#Test that all required dependencies are installed on your computer
#def test_dependencies(dep, alternate_name=None):
"""
    Test external dependencies by running them as subprocesses"
    
    try:
        subprocess.check_output(dep, stderr=subprocess.PIPE, shell=True)
        print("%s is installed properly as %s. OK" % (dep, dep), '\n')
        return 1
    except subprocess.CalledProcessError:
        try:
            subprocess.check_output(alternate_name, stderr=subprocess.PIPE, shell=True)
            print("%s is installed properly as %s. OK" % (dep, alternate_name), '\n')
            return 1
        except subprocess.CalledProcessError:
            print("===%s/%s IS NOT YET INSTALLED PROPERLY===" % (dep, alternate_name))
            return 0
        
##-----------------SETUP WORKSPACE/VIEW RAW DATA------------------##

dependencies = [('sextractor', 'sex'), ('SWarp', 'swarp')]
i = 0
for dep_name1, dep_name2 in dependencies:
    i += test_dependencies(dep_name1, dep_name2)
print("%i out of %i external dependencies installed properly.\n" % (i, len(dependencies)))
if i != len(dependencies):
    print("Please correctly install these programs before continuing by following the instructions in README.md.")
else:
    print("You are ready to continue.\n")"""
            
#Define directories of data

curpath = os.path.abspath('.')                   #Working Directory (Top Level)
dataFolder = os.path.join(curpath, 'data')       #Directory of all data
biasFolder = os.path.join(dataFolder, 'bias')    #Directory of bias frames
flatFolder = os.path.join(dataFolder, 'flat')    #Directory of flat fields
sciFolder = os.path.join(dataFolder, 'science')  #Directory of science data
procFolder = os.path.join(curpath, 'processing') #Directory for processing data
if not os.path.isdir(procFolder):
    os.mkdir(procFolder)
else:
    for f in os.listdir(procFolder):
        try:
            os.remove(os.path.join(procFolder, f)) #remove processing folder from previous iterations
        except:
            print('Could not remove processing', f)

#Create lists of files in each directory/subdirectory

os.chdir(sciFolder)
fileList = sorted(glob.glob('*.fits'))
biasList = sorted([os.path.join(biasFolder, i) for i in os.listdir(biasFolder)])
flatList = sorted([os.path.join(flatFolder, i) for i in os.listdir(flatFolder)])
sciList = sorted([os.path.join(sciFolder, i) for i in os.listdir(sciFolder)])
procList = sorted([os.path.join(procFolder, file).replace('.fits','.proc.fits') for file in fileList])
combinedFile = os.path.join(procFolder, 'AT2018cow_combined.fits')
weightFile = os.path.join(procFolder, 'AT2018cow_weight.fits')
resampledFile = os.path.join(procFolder, 'AT2018cow_resampled.fits')

print(len(biasList), 'bias frames found,', len(flatList), 'flat fields found,', len(sciList), 'science images found\n')


#Turn off python warnings- many warning for .fits type files that we do not want to worry about
warnings.filterwarnings('ignore')

## Open an example FITS file to view raw data & header info

exampleFile = biasList[0]         # The first science image
HDUList = fits.open(exampleFile, ignore_missing_end=True) # Open the file
HDUList.info()                   # Print some information about the FITS file structure.

#Get & Display raw image data

data = HDUList[1].data #a numpy array of data from the first extension, each position of array is value of 

mean, median, std = sigma_clipped_stats(data[500:1600,500:1600]) #mean, median, standard deviation statistics of the image only from areas where the CCD is being used to avoid including blank pixels
plt.figure(figsize=(10,10))      #plot within a 10x10 panel
plt.imshow(data, vmin = median - 2*std, vmax = median + 20*std, origin = 'lower')  #plot data with value ranges set and location of [0,0] set with origin 
plt.colorbar()#add colorbar to image
plt.title('RAW SCIENCE IMAGE')
plt.close()    #open window of image

HDUList.close()



##------------CREATE BIAS FRAME----------------##
print('------------------------CREATING MASTER BIAS-------------------------')

#Bias frames are images with 0 exposure time to record the offset of the detector(CCD). Below I create the master bias frame by averaging multiple together using median-combination.

#Open up images and save standard deviations for checks
bias_std = [] 

for bias in biasList:
    biasHDR = fits.open(bias, ignore_missing_end=True)
    biasData = biasHDR[1].data
    biasHDR.close()
    mean, median, std = sigma_clipped_stats(biasData) #mean, median, standard deviation statistics of the image only from areas where the CCD is being used to avoid including blank pixels
    bias_std.append(std)
    plt.figure(figsize=(10,10))      #plot within a 10x10 panel
    plt.imshow(biasData, vmin = median - 2*std, vmax = median + 10*std, origin = 'lower')  #plot data with value ranges set and location of [0,0] set with origin 
    plt.colorbar()#add colorbar to image
    plt.close()

#create median-combined bias images (master bias)

#3D numpy array to store each of the bias images
ny = (fits.open(biasList[0], ignore_missing_end=True))[1].header['NAXIS2'] #gets ccd size from the header info to make correct size numpy array
nx = (fits.open(biasList[0], ignore_missing_end=True))[1].header['NAXIS1']
numBiasFiles = len(biasList)
biasImages = np.zeros((ny, nx, numBiasFiles))

for i in range(numBiasFiles):          #fill out array with data from bias images
    bias_HDR = fits.open(biasList[i], ignore_missing_end=True)
    biasImages[:,:,i] = bias_HDR[1].data
    bias_HDR.close()

#do a median combination on all leayers of array to create the master bias
masterBias = np.median(biasImages, axis=2)

mean, median, std = sigma_clipped_stats(masterBias) #mean, median, standard deviation of the master bias
plt.figure(figsize=(10,10))      #plot within a 10x10 panel
plt.imshow(masterBias, vmin = median - 2*std, vmax = median + 10*std, origin = 'lower')  #plot data with value ranges set and location of [0,0] set with origin 
plt.colorbar()#add colorbar to image
plt.title('MASTER BIAS')
plt.close()

print('Master Bias STD:' , std, 'Master Bias Median:' , median, '\n')

##---------------Create Flat Field----------------##
print('------------------------CHECKING %s FLAT IMAGES----------------------' % (str(len(flatList))))

#Image(x,y) = Brightness distribution of Sky(x,y)*telescope optics/detectors(x,y) + bias(x,y)
#Flats are created by taking image of uniform light source(illuminated spot on dome or twilight sky
count=0
      
for flat in flatList:
    flatHDR = fits.open(flat, ignore_missing_end=True)
    flatData = flatHDR[1].data
    flatHDR.close()
    mean, median, std = sigma_clipped_stats(flatData[500:1600,500:1600]) #mean, median, standard deviation statistics of the image only from areas where the CCD is being used to avoid including blank pixels
    plt.figure(figsize=(10,10))      #plot within a 10x10 panel
    plt.imshow(flatData, vmin = median - 5*std, vmax = median + 5*std, origin = 'lower')  #plot data with value ranges set and location of [0,0] set with origin 
    plt.colorbar()#add colorbar to image
    plt.title('FLAT')#add title
    plt.close()
    count+=1
    print('IMAGE %s--FILTER:' % (count), ((flatHDR)[0].header['ACAMFILT']), 'EXPOSURE TIME(s):', ((flatHDR)[0].header['EXPTIME']), 'MEDIAN PIX VAL:', median, '\n')
    

print('-------------------------CREATING MASTER FLAT-----------------------')

#Remove bias & normalize each flat image

#Create 3D array for flats
ny = (fits.open(flatList[0], ignore_missing_end=True))[1].header['NAXIS2'] #gets ccd size from the header info to make correct size numpy array
nx = (fits.open(flatList[0], ignore_missing_end=True))[1].header['NAXIS1']
numflatFiles = len(flatList) #number of layers to have in 3D array = number of flat images
flatImages = np.zeros((ny, nx, numflatFiles))  #create array of zeroes [x axis size of CCD, y axis size of CCD, # of flats)

for i in range(numflatFiles):          #fill out array with data from bias images
    #Open data
    flat_HDR = fits.open(flatList[i], ignore_missing_end=True)
    flatData = flat_HDR[1].data *1. #convert to floating point
    flat_HDR.close()
    flatData -= masterBias   #Subtract the master bias from the combined flats
    norm = np.median(flatData[500:1600,500:1600]) #find the median of counts per pixel in a square where the CCD is being used(actually collecting light)
    flatImages[:,:,i] = flatData /norm

#Median combination of the 3D array of all flat images to create a 2D array of the master flat
masterFlat = np.median(flatImages, axis=2)

mean, median, std = sigma_clipped_stats(masterFlat[500:1600,500:1600]) #mean, median, standard deviation of the master bias
plt.figure(figsize=(10,10))      #plot within a 10x10 panel
plt.imshow(masterFlat, vmin = median - 5*std, vmax = median + 5*std, origin = 'lower')  #plot data with value ranges set and location of [0,0] set with origin 
plt.colorbar()#add colorbar to image
plt.title('MASTER FLAT')
plt.close()
print('MASTER FLAT DATA-- MEAN:', mean, 'MEDIAN:', median, ' STD:', std, '\n')

#get rid of any unexposed pixels on the CCD by setting them to NaN (not a number)

masterFlatFixed = np.copy(masterFlat)
if np.any(masterFlat < 0.2):
    masterFlatFixed[masterFlat < 0.2] = float('NaN')

plt.figure(figsize=(10,10))
plt.imshow(masterFlatFixed, vmin = 0, vmax = median + 5*std, origin = 'lower')
plt.colorbar()
plt.title('MASTER FLAT')
plt.close()


##-------------------------SCIENCE IMAGE PROCESSING----------------------##
print('-----------------------------PROCESSING SCIENCE IMAGES--------------------------')

#Image(x,y) = Brightness distribution of Sky(x,y)*telescope optics/detectors(x,y) + bias(x,y)-----find Sky(x,y) below
#Sky(x,y) = (image(x,y) - bias(x,y))/telescope optics-detectors(x,y)

print('Processing %s science images' %(len(sciList)), '\n')

for i in range(len(sciList)):
    HDUList = fits.open(sciList[i], ignore_missing_end=True)
    primaryHeader = HDUList[0].header
    sciData = HDUList[1].data
    HDUList.close()

    procData = (sciData - masterBias)/masterFlatFixed

    procHDU = fits.PrimaryHDU(procData) #new HDU with processed science image
    procHDU.header = primaryHeader  
    procHDU.header.add_history('Bias corrected and flat-fielded')#Add this into the header for reference

    procHDU.writeto(procList[i], overwrite=True)#add the processed images to the processed folder

plt.figure(figsize=(14,14))
for i in range(len(sciList)):
    HDUList = fits.open(procList[i], ignore_missing_end=True)
    procData = HDUList[0].data
    HDUList.close()
               
    mean, median, std = sigma_clipped_stats(procData)
    plt.imshow(procData, vmin = median - 2*std, vmax = median + 10*std, origin = 'lower')
    plt.colorbar()
    plt.title('Proessed Science Image %s' % (i))
    plt.close()

##---------------------------ASTROMETRY-------------------------##
print('------------------------------BEGINNING ASTROMETRY----------------------------')

#Align images with reference star so that observed object lies at the same pixel position for each image

## Measure the precise position of a star in each image

# Set the approximate star coordinates of reference star chosen
estimated_star_x = [998,998,958]
estimated_star_y = [1161,1121,1121]
actual_star_x = [0,0,0] # dummy values
actual_star_y = [0,0,0]

for i in range(len(sciList)):
    HDUList = fits.open(procList[i], ignore_missing_end=True)
    procData = HDUList[0].data
    HDUList.close()
    (x0, x1) = (estimated_star_x[i]-20, estimated_star_x[i]+20) #create smaller area to cover the star
    (y0, y1) = (estimated_star_y[i]-20, estimated_star_y[i]+20)
    cutout = procData[y0:y1,x0:x1]
    mean, median, std = sigma_clipped_stats(cutout)
    cx, cy = photutils.centroid_com(cutout-median) #find the center position of the star
    actual_star_x[i] = cx+x0 #fill in actual calculated locations of star center from above
    actual_star_y[i] = cy+y0
    print('Image %s unshifted ref location:' % (i+1), actual_star_x[i], actual_star_y[i], '\n')

#offset images to match centroid locations

#default crop region
xmin, xmax = 50, 2050
ymin, ymax = 150, 2150
alignedImages = np.zeros((ymax-ymin, xmax-xmin, len(sciList))) #create empty array the size of [y pix, x pix, # images]

#calcualte the offset for each image and apply it then add to 3D array
for i in range(len(sciList)):
    xoffset = int(round(actual_star_x[i]-actual_star_x[0]))
    yoffset = int(round(actual_star_y[i]-actual_star_y[0]))
    print('Image %s offset:' % (i+1), xoffset, yoffset)

    HDUList = fits.open(procList[i], ignore_missing_end=True)
    procData = HDUList[0].data
    HDUList.close()

    shiftedData = procData[(ymin+yoffset):(ymax+yoffset), (xmin+xoffset):(xmax+xoffset)] #create 2D array of shifted data so star is in same pixel locaton for each image
    shiftedData -= np.nanmedian(shiftedData)  #remove the sky
    alignedImages[:,:,i] = shiftedData

for i in range(len(sciList)):
    (starX, starY) = (int(round(actual_star_x[0]-xmin)), int(round(actual_star_y[0]-ymin)))
    starCutout = alignedImages[starY-9:starY+9,starX-9:starX+9,i]
    mean, median, std = sigma_clipped_stats(alignedImages)
    plt.imshow(starCutout, vmin = 0, vmax = 10000, origin = 'lower')
    plt.title('Shifted Data %s' % (i))
    plt.close()

masterSci = np.nanmedian(alignedImages, axis=2)

HDUList = fits.open(sciList[i], ignore_missing_end=True)
primaryHeader = HDUList[0].header
HDUList.close()

sciHDU = fits.PrimaryHDU(masterSci) #new HDU with processed science image
sciHDU.header = primaryHeader
sciHDU.header.add_history('bias corrected, Sky bubtracted, flat fielded, combined science images')#Add this into the header for reference

sciHDU.writeto(combinedFile, overwrite=True)#add the processed images to the processed folder

## Confirm the stack worked correctly by reloading the combined image from disk and displaying it.

HDUList = fits.open(combinedFile, ignore_missing_end=True) # Open the first science file in order to retrieve its header
combinedImage = HDUList[0].data
HDUList.close()

plt.figure(figsize = (12,10))
mean, median, std = sigma_clipped_stats(combinedImage)
plt.imshow(combinedImage, vmin = median - 1*std, vmax = median + 100*std, origin='lower')
plt.colorbar()
plt.show()


##--------------------ASTROMETRIC CALIBRATION-----------------------##
print('--------------------BEGINNING ASTROMETRIC CALIBRATION-------------------------')
print('SEXTRACTOR NOT COMPATIBLE-END")
"""
#Pattern Match a group of stars to the catalog in order to calibrate the image, resulting in RA & Dec equal to pixel coordinates
#Utilize external python script Autoastrometry3.py

autoastrometry_script = os.path.join(dataFolder, 'autoastrometry3.py')
os.chdir(procFolder)
sextractorCommand = 'sextractor' # Change this if you get an error below

try:
    #Run the autoastrometry script using 2MASS as the reference catalog by specifying 'tmc' with the '-c' option
    astromFile = combinedFile.replace('.fits','.astrom.fits')
    shutil.copy('AT2018cow_combined.fits', 'AT2018cow_combined.astrom.fits')
    command = 'python3 %s %s -o %s -c tmc -px 0.253 -inv -sex %s' % (autoastrometry_script, combinedFile, astromFile, sextractorCommand)
    print('Executing command: %s' % command)
    print('Processing...')
    rval = subprocess.run(command.split(), check=True, capture_output=True)
    print('Process completed.')
    print(rval.stdout.decode())
except subprocess.CalledProcessError as err:
    print('Could not run autoastrometry with error %s. Check if file exists.'%err)
    
os.chdir(curpath)"""
