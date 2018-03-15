import matplotlib.pyplot as plt
import numpy
import MDAnalysis
import time
import math
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
                print('OUTPUT File: {}'.format(outname))
        elif line == 'END CONFIG FILE\n':
            print('No OUTPUT file found in config.')
            break


# Get name of Coordinate DCD files from config file
def getCoordDCDs(cF):
    global coordDcd
    coordDcd = []
    txt = open(cF, 'r')
    while len(coordDcd) == 0:
        line = txt.next()
        if line == 'COORD DCD FILES:\n':
            line = txt.next()
            while line != '\n':
                if line == 'END CONFIG\n':
                    print('NO DCD FILES FOUND IN CONFIG')
                coordDcd.append(line[:-1])
                line = txt.next()
            if debug:
                print('Coordinate DCD files: {}'.format(coordDcd))

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
            print 'ATOM3: ',atom3

# Define subset of data without solvent
def parseWater():
    # select all atoms that are not water or hydrogen
    if debug:
        print("\n--Reading DCD Data--\n\t- Parsing out WATER...\n")
    global ionsCoord
    global H2OCoord
    H2OCoord = coord.select_atoms("resname CL3")
    ionsCoord = coord.select_atoms("not resname CL3 and not resname Cl- and not resname Na+")

# Initialize MD Analysis
def initMDA():
    global coord, dims, hdims, rMax, binCount, debug

    m = False       # Swtiched to True if user requests an rMax value greater than the system allows
    # coordinate universe
    coord = MDAnalysis.Universe(psf, coordDcd)
    dims = [coord.dimensions[0], coord.dimensions[1], coord.dimensions[2]]
    hdims = [dims[0]/2,dims[1]/2,dims[2]/2]
    rMaxLimit = numpy.sqrt((dims[0]**2) + (dims[1]**2) + (dims[2]**2))
    if rMax > rMaxLimit:
        rMax = rMaxLimit
        m = True

    binCount = int((rMax - rMin)/binSize)

    if debug:
        print "--Dimensions of System--"
        print "\tTotal System Dimensions: {} A x {} A x {} A".format(dims[0], dims[1], dims[2])
        print "\tMin Interparticle Distance Considered: {} A".format(rMin)
        if m:
            print "\tMax Interparticle Distance Considered: {} A\tThe requested rMax value was bigger than the simulation box size permits, and was truncated".format(rMax)
        else:
            print "\tMax Interparticle Distance Considered: {} A".format(rMax)
        print "\tBin Size Used: {}".format(binSize)
        print "\tBin Count: {}".format(binCount)
        print "\tExcluded Volume Offset: {}".format(d)

    # Truncate solvent out of the data
    parseWater()


    # Print log data
    printLogData(debug)

# Print log data
def printLogData(d):
    if d:
        global ionsCoord
        print "--Simulation Log Info--"
        # print list of atoms being considered
        print "DCD coord universe:", len(ionsCoord), "atom(s)"
        #for i in range(0, len(ionsCoord)):
            #print "\t", ionsCoord[i]
        # some general log info
        print "\nNumber of time steps in coordinate trajectory:", len(coord.trajectory)

# Iterate through all pairs of particles in all simulations,
#    identifying each pair of particles, performing computations,
#    and storing the results in a data set
def iterate():
    global plots,dims,hdims
    hrMax=0.5*rMax
    pH2O=numpy.zeros((binCount,binCount,binCount,3))
    nH2O=numpy.zeros((binCount,binCount,binCount))

    if debug:
        print "-- Iterating through all particle pairs in first time step to establish pair types"
    for ts in coord.trajectory:                 # Iterate through all time steps
        dims = [coord.dimensions[0], coord.dimensions[1], coord.dimensions[2]]
        hdims = [x / 2. for x in dims] # EDIT: dims/2.

        sys.stdout.write("Progress: {0:.2f}% Complete\r".format((float(ts.frame) / float(len(coord.trajectory))) * 100))
        sys.stdout.flush()

# Compute radial vectors
        if ts.frame <= 0:
# Compute solute vectors
	    axes=numpy.zeros((3,3),dtype=float) # these dimensions because each atom position is a 3 element sequence
	    sel1 = atom1 # "resid 1 and name "+atom1
	    sel2 = atom2
            if atom3 != '.': # XXX: for LJ sphere dimer simulations use atom3 = . in the config file
                sel3 = atom3

	    sel1_univ = coord.select_atoms(sel1)
	    sel2_univ = coord.select_atoms(sel2)
            if atom3 != ".":
                sel3_univ = coord.select_atoms(sel3)
# Atom positions
            if atom3 != '.':
                atom1_pos = sel1_univ.atoms[0].position
                atom2_pos = sel2_univ.atoms[0].position
                atom3_pos = sel3_univ.atoms[0].position
            else:
                atom1_pos = sel1_univ.atoms[0].position
                atom2_pos = sel1_univ.atoms[1].position
                atom3_pos = numpy.zeros(3, dtype=float)
# Find 3 axes of solute
	    r1 = atom2_pos-atom1_pos
	    r1 /= math.sqrt(numpy.dot(r1,r1))
	    t1 = atom3_pos-atom1_pos
	    r3 = numpy.cross(r1,t1)
	    r3 /= math.sqrt(numpy.dot(r3,r3))
	    r2 = numpy.cross(r3,r1)
	    r2 /= math.sqrt(numpy.dot(r2,r2))
# Define 3 axes of solute
	    axes[0] = r1
	    axes[1] = r2
	    axes[2] = r3

            rCen=numpy.zeros(3)
            mtot=0
            for a in ionsCoord:
                rCen+=a.position*a.mass
                mtot+=a.mass
            rCen=rCen/mtot
            if debug:
                print "-- Printing the .crd file"
            outCrd = open(outname+".crd", 'w')
            ntyp=0
            i=-1
            typlist=[]
            atyp=numpy.zeros(len(ionsCoord),dtype=numpy.int)
            for a in ionsCoord:
                i+=1
                inew=1
                for jtyp in typlist:
                    if a.type == jtyp[1]:
                        atyp[i]=jtyp[0]
                        inew=0
                if inew == 1:
                    ntyp+=1
                    typlist.append([ntyp,a.type])
                    atyp[i]=ntyp
            i=0
            outCrd.write("{:4d} {:4d}\n".format(len(ionsCoord),ntyp))
            for a in ionsCoord:
		rNew = numpy.dot(axes,a.position-rCen)
                outCrd.write("{:3d} {:12.6f} {:12.6f} {:12.6f}\n".format(atyp[i],rNew[0],rNew[1],rNew[2])) # write rotated solute coordinates
                i+=1
            outCrd.close()
        for a in H2OCoord.residues:
            #rWat=a.atoms[1].position-rCen # selecting C1 in CL3 which is index 1
            rWat = (a.atoms[1].position + d * ((a.atoms[1].position - a.atoms[0].position)/1.1)) - rCen # selecting C1 in CL3 which is index 1 and H1 (0), to make new vector to center of volume exclusion.
            for i in range(3):
                while rWat[i] < -hdims[i]:
                    rWat[i]+=dims[i]
                while rWat[i] > hdims[i]:
                    rWat[i]-=dims[i]
	    rWat=numpy.dot(axes,rWat)
# Calculate x,y,z bin
            x=rWat[0]+hrMax
            if x < rMax and x > rMin:
                y=rWat[1]+hrMax
                if y < rMax and y > rMin:
                    z=rWat[2]+hrMax
                    if z < rMax and z > rMin:
                        ix=int(x/binSize)
                        iy=int(y/binSize)
                        iz=int(z/binSize)
                        nH2O[ix][iy][iz]+=1
			pNow=a.atoms[1].position-a.atoms[0].position
			pH2O[ix][iy][iz]+=pNow # pNow is in the Lab frame (unrotated frame, the frame of the box)
#
    dxH2O=1/(binSize**3/dims[0]/dims[1]/dims[2]*len(H2OCoord.residues)*len(coord.trajectory))
    outFile = open(outname+".gr3", 'w')
    for i in range(binCount):
        for j in range(binCount):
            for k in range(binCount):
		if nH2O[i][j][k] != 0: # so i dont get NaNs
		    pH2O[i][j][k]/=nH2O[i][j][k]*1.1 # normalize by number of bins and equilibrium bond value
		    pH2O[i][j][k]=numpy.dot(axes,pH2O[i][j][k]) # rotate into aligned frame (rotated frame, the frame of the solute axes)
                outFile.write("{:7.3f} {:7.3f} {:7.3f} {:18.12f} {:18.12f} {:18.12f} {:18.12f}\n".format((i+0.5)*binSize-hrMax,(j+0.5)*binSize-hrMax,(k+0.5)*binSize-hrMax,nH2O[i][j][k]*dxH2O,pH2O[i][j][k][0],pH2O[i][j][k][1],pH2O[i][j][k][2]))
    outFile.close()
####

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
    
    # Get names of Coord DCD files from config file
    getCoordDCDs(configFile)

    # Define coordinate min/max and bin size
    getCoordBounds(configFile)

    # Initialize MD Analysis
    initMDA()

    # Iterate over time steps, and perform MD calculations
    iterate()

    end = time.time()
    t = end - start
    print "\nTotal running time: {:.2f} sec".format(t)

# Main program code
main()
