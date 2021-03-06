# USAGE:    python convert.3D-Cyl.py [same config file as gr.3D.py]
# python 2.7.x

import matplotlib.pyplot as plt
import numpy
import MDAnalysis
import time
from math import *
import sys


# FILE VARIABLES
configFile = sys.argv[1]

psf = None
outname = None
coordDcd = None
param = None
coord = None
temperature = None
dims = None
hdims = None

debug = False
junkCounter = 0     # counter used for debugging


# DEFUALT GLOBAL VARIABLES
rMin = 0
rMax = 25
binSize = 0.1
binCount = 0
elecPermittivity = 8.854e-12      # electrical permittivity of vacuum C^2/Jm
boltzmann = 1.9872041e-3          # boltzmann constant in kcal/(K mol)
rho = 0
epsilon = []        # list of epsilon values for all non-solvent atoms
lj_rMin = []          # list of rMin values for all non-solvent atoms
d = 0               # offset from central carbon to center of volume exclusion



# Set debug mode from config file
def setDebug(cF):
    global debug
    txt = open(cF, 'r')
    line = txt.next()
    while line != "END CONFIG\n":
        if line == "DEBUG MODE: ON\n":
            debug = True
        line = txt.next()

# Get name of PSF file from config file
def getPsf(cF):
    global psf
    print('\n\t***DCD Analysis***')
    if debug:
        print('\t\tDebug Mode ON')
    else:
        print('\t\tDebug Mode OFF')
    txt = open(cF, 'r')
    while psf is None:
        line = txt.next()
        if line == 'PSF FILE:\n':
            psf = txt.next()[:-1]
            if debug:
                print('PSF File: {}'.format(psf))
        elif line == 'END CONFIG FILE\n':
            print('No PSF file found in config.')
            break

# Get name of PSF file from config file
def getOut(cF):
    global outname
    txt = open(cF, 'r')
    while outname is None:
        line = txt.next()
        if line == 'OUTPUT FILE NAME:\n':
            outname = txt.next()[:-1]
            if debug:
                print('3D OUTPUT File: {}'.format(outname))
        elif line == 'END CONFIG FILE\n':
            print('No OUTPUT file found in config.')
            break

# Set coordinate max/min and binsize
def getCoordBounds(cF):
    global dims,hdims
    txt = open(cF,'r')
    line = txt.next()
    length1 = len("MIN DISTANCE: ")
    length2 = len("MAX DISTANCE: ")
    length3 = len("BIN SIZE: ")
    length4 = len("OFFSET: ")
    length5 = len("ATOM1: ")
    length6 = len("ATOM2: ")
    length7 = len("ATOM3: ")

    global rMin, rMax, binSize, binCount, d, atom1, atom2, atom3

    # scan config file for coord and bin values
    while line != "END CONFIG\n":
        line = txt.next()
        if line[:length1] == "MIN DISTANCE: ":
            rem = -1 * (len(line) - length1)
            rMin = int(line[rem:-1])
        elif line[:length2] == "MAX DISTANCE: ":
            rem = -1 * (len(line) - length2)
            rMax = int(line[rem:-1])
        elif line[:length3] == "BIN SIZE: ":
            rem = -1 * (len(line) - length3)
            binSize = float(line[rem:-1])
        elif line[:length4] == "OFFSET: ":
            rem = -1 * (len(line) - length4)
            d = float(line[rem:-1])
        elif line[:length5] == "ATOM1: ":
            rem = -1 * (len(line) - length5)
            atom1 = str(line[rem:-1])
        elif line[:length6] == "ATOM2: ":
            rem = -1 * (len(line) - length6)
            atom2 = str(line[rem:-1])
        elif line[:length7] == "ATOM3: ":
            rem = -1 * (len(line) - length7)
            atom3 = str(line[rem:-1])

def convertCylindrical():
    OutFile3D = numpy.loadtxt(outname+".gr3")
    x_axis = OutFile3D[:,0]
    y_axis = OutFile3D[:,1]
    z_axis = OutFile3D[:,2]
    gxyz = OutFile3D[:,3]
    fxyz = OutFile3D[:,4]
    binCount = int((x_axis[-1] - x_axis[0])/ binSize) + 1
    binCount2 = binCount * binCount
    hrMax = rMax / 2

    nr = numpy.zeros(binCount, dtype=int)
    gr = numpy.zeros((binCount,binCount), dtype=float)
    fr = numpy.zeros((binCount,binCount), dtype=float)

    for i in range(binCount):
        ix = int(((x_axis[i*binCount2]+binSize)-(x_axis[0]+binSize))/binSize)
        for j in range(binCount):
            index=i*binCount2+j*binCount
            for k in range(binCount):
                r = sqrt(y_axis[index]**2 + z_axis[index+k]**2)
                ir = int(r/binSize)
                nr[ir] += 1
                gr[ix][ir] += gxyz[index+k] #/(2*pi*r) * binSize # normalizing by the Jacobean doesn't work b/c cartesian bins dont translate smoothly into radial ones
                fr[ix][ir] += fxyz[index+k] *(2*pi*r) / binSize # multiply by jacobian because I divide by discritized jacobean below. FIXME

    im,m = min(enumerate(nr), key=lambda x: x[1] if x[1] > 0 else float('inf')) # find minimum value of nr that is > 0, m. and the index of that value, im.
    # Normalize gr and fr arrays
    for ir in range(binCount):
        if nr[ir] > 0:
            norm = binSize/(nr[ir]/m) # m vs binCount here? FIXME
            for ix in range(binCount):
                gr[ix][ir] *= norm
                fr[ix][ir] *= norm

    # the x-axis is the longitudinal axis
    # so new plot will be a function of x and r. Where r is the radius in the yz plane
    outFile = open(outname+".cyl.gr3", 'w')
    outFile.write("# 1: x Distance\n")
    outFile.write("# 2: r Distance\n")
    outFile.write("# 3: g(r) Density\n")
    outFile.write("# 4: Force\n")
    for ix in range(binCount):
        for ir in range(int(binCount/2)):
            outFile.write("{:7.3f} {:7.3f} {:18.12f} {:18.12f}\n".format( (ix+0.5)*binSize-hrMax, (ir+0.5)*binSize, gr[ix][ir], fr[ix][ir] ))
    outFile.close()


# main program
def main():

    # access global var for config file
    global configFile, pdb
    start = time.time()

    # Read config setting for debug mode
    setDebug(configFile)

    # Get name of PSF file from config file
    getPsf(configFile)
    
    # Get name of OUTPUT file from config file
    getOut(configFile)
    
    # Define coordinate min/max and bin size
    getCoordBounds(configFile)

    # Convert to cylindrical coordinates from cartesian
    convertCylindrical()


    end = time.time()
    t = end - start
    print "\nTotal running time: {:.2f} sec".format(t)

# Main program code
main()
