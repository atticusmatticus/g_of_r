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

# Read prmtop file and populate global variables
def ParsePrmtopBonded(top_file):
    global bond_fc,bond_equil_values,angle_fc,angle_equil_values,dihedral_fc,dihedral_period,dihedral_phase,nbonh,nbona,ntheta,ntheth,nphia,nphih,bondsh,bondsa,anglesh,anglesa,dihedralsh,dihedralsa,n_atoms,n_types,atom_names,atom_type_index,nb_parm_index,lj_a_coeff,lj_b_coeff
        
    param = open(top_file,'r')
    pointers = numpy.zeros(31,dtype=numpy.int)
    lines = param.readlines()
    for i in range(len(lines)):	
        if lines[i][0:14] == '%FLAG POINTERS':
            for j in range(4):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    pointers[j*10+k] = int(temp[k])
            n_atoms = pointers[0]
            n_types = pointers[1]
            nbonh = pointers[2]
            nbona = pointers[12]
            ntheth = pointers[4]
            ntheta = pointers[13]
            nphih = pointers[6]
            nphia = pointers[14]
            numbnd = pointers[15]
            numang = pointers[16]
            numtra = pointers[17]
            n_type_lines = int(math.ceil(n_atoms/10.))
            n_name_lines = int(math.ceil(n_atoms/20.))
            n_nb_parm_lines = int(math.ceil(n_types*n_types/10.))
            n_lj_param_lines = int(math.ceil((n_types*(n_types+1)/2)/5.))
            n_bond_lines = int(math.ceil(numbnd/5.))
            n_angle_lines = int(math.ceil(numang/5.))
            n_dihedral_lines = int(math.ceil(numtra/5.))
            n_bondsh_lines = int(math.ceil(nbonh*3/10.))
            n_bondsa_lines = int(math.ceil(nbona*3/10.))
            n_anglesh_lines = int(math.ceil(ntheth*4/10.))
            n_anglesa_lines = int(math.ceil(ntheta*4/10.))
            n_dihedralsh_lines = int(math.ceil(nphih*5/10.))
            n_dihedralsa_lines = int(math.ceil(nphia*5/10.))
            bond_fc = numpy.zeros(numbnd,dtype=numpy.float)
            bond_equil_values = numpy.zeros(numbnd,dtype=numpy.float)
            angle_fc = numpy.zeros(numang,dtype=numpy.float)
            angle_equil_values = numpy.zeros(numang,dtype=numpy.float)
            dihedral_fc = numpy.zeros(numtra,dtype=numpy.float)
            dihedral_period = numpy.zeros(numtra,dtype=numpy.float)
            dihedral_phase = numpy.zeros(numtra,dtype=numpy.float)
            SCEE_factor = numpy.zeros(numtra,dtype=numpy.float)
            SCNB_factor = numpy.zeros(numtra,dtype=numpy.float)
            bondsh_linear = numpy.zeros(3*nbonh,dtype=int)
            bondsa_linear = numpy.zeros(3*nbona,dtype=int)
            bondsh = numpy.zeros((nbonh,3),dtype=int)
            bondsa = numpy.zeros((nbona,3),dtype=int)
            anglesh_linear = numpy.zeros(4*ntheth,dtype=int)
            anglesa_linear = numpy.zeros(4*ntheta,dtype=int)
            anglesh = numpy.zeros((ntheth,4),dtype=int)
            anglesa = numpy.zeros((ntheta,4),dtype=int)
            dihedralsh_linear = numpy.zeros(5*nphih,dtype=int)
            dihedralsa_linear = numpy.zeros(5*nphia,dtype=int)
            dihedralsh = numpy.zeros((nphih,5),dtype=int)
            dihedralsa = numpy.zeros((nphia,5),dtype=int)
            atom_names = []
            atom_type_index = numpy.zeros((n_atoms),dtype=int)
            nb_parm_index = numpy.zeros(n_types*n_types,dtype=int)
            lj_a_coeff = numpy.zeros((n_types*(n_types+1))/2,dtype=float)
            lj_b_coeff = numpy.zeros((n_types*(n_types+1))/2,dtype=float)

        if lines[i][0:25] == '%FLAG BOND_FORCE_CONSTANT':
            for j in range(n_bond_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    bond_fc[j*5+k] = float(temp[k])
        if lines[i][0:22] == '%FLAG BOND_EQUIL_VALUE':
            for j in range(n_bond_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    bond_equil_values[j*5+k] = float(temp[k])
        if lines[i][0:26] == '%FLAG ANGLE_FORCE_CONSTANT':
            for j in range(n_angle_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    angle_fc[j*5+k] = float(temp[k])
        if lines[i][0:23] == '%FLAG ANGLE_EQUIL_VALUE':
            for j in range(n_angle_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    angle_equil_values[j*5+k] = float(temp[k])
        if lines[i][0:29] == '%FLAG DIHEDRAL_FORCE_CONSTANT':
            for j in range(n_dihedral_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    dihedral_fc[j*5+k] = float(temp[k])
        if lines[i][0:26] == '%FLAG DIHEDRAL_PERIODICITY':
            for j in range(n_dihedral_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    dihedral_period[j*5+k] = float(temp[k])
        if lines[i][0:20] == '%FLAG DIHEDRAL_PHASE':
            for j in range(n_dihedral_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    dihedral_phase[j*5+k] = float(temp[k])
        if lines[i][0:23] == '%FLAG SCEE_SCALE_FACTOR':
            for j in range(n_dihedral_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    SCEE_factor[j*5+k] = float(temp[k])
        if lines[i][0:23] == '%FLAG SCNB_SCALE_FACTOR':
            for j in range(n_dihedral_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    SCNB_factor[j*5+k] = float(temp[k])
        if lines[i][0:24] == '%FLAG BONDS_INC_HYDROGEN':
            for j in range(n_bondsh_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    bondsh_linear[j*10+k] = int(temp[k])
            for j in range(nbonh):
                bondsh[j][0] = bondsh_linear[j*3]
                bondsh[j][1] = bondsh_linear[j*3+1]
                bondsh[j][2] = bondsh_linear[j*3+2]
        if lines[i][0:28] == '%FLAG BONDS_WITHOUT_HYDROGEN':
            for j in range(n_bondsa_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    bondsa_linear[j*10+k] = int(temp[k])			
            for j in range(nbona):
                bondsa[j][0] = bondsa_linear[j*3]
                bondsa[j][1] = bondsa_linear[j*3+1]
                bondsa[j][2] = bondsa_linear[j*3+2]
        if lines[i][0:25] == '%FLAG ANGLES_INC_HYDROGEN':
            for j in range(n_anglesh_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    anglesh_linear[j*10+k] = int(temp[k])
            for j in range(ntheth):
                anglesh[j][0] = anglesh_linear[j*4]
                anglesh[j][1] = anglesh_linear[j*4+1]
                anglesh[j][2] = anglesh_linear[j*4+2]
                anglesh[j][3] = anglesh_linear[j*4+3]
        if lines[i][0:29] == '%FLAG ANGLES_WITHOUT_HYDROGEN':
            for j in range(n_anglesa_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    anglesa_linear[j*10+k] = int(temp[k])			
            for j in range(ntheta):
                anglesa[j][0] = anglesa_linear[j*4]
                anglesa[j][1] = anglesa_linear[j*4+1]
                anglesa[j][2] = anglesa_linear[j*4+2]
                anglesa[j][3] = anglesa_linear[j*4+3]
        if lines[i][0:28] == '%FLAG DIHEDRALS_INC_HYDROGEN':
            for j in range(n_dihedralsh_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    dihedralsh_linear[j*10+k] = int(temp[k])
            for j in range(nphih):
                dihedralsh[j][0] = dihedralsh_linear[j*5]
                dihedralsh[j][1] = dihedralsh_linear[j*5+1]
                dihedralsh[j][2] = dihedralsh_linear[j*5+2]
                dihedralsh[j][3] = dihedralsh_linear[j*5+3]
                dihedralsh[j][4] = dihedralsh_linear[j*5+4]
        if lines[i][0:32] == '%FLAG DIHEDRALS_WITHOUT_HYDROGEN':
            for j in range(n_dihedralsa_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    dihedralsa_linear[j*10+k] = int(temp[k])			
            for j in range(nphia):
                dihedralsa[j][0] = dihedralsa_linear[j*5]
                dihedralsa[j][1] = dihedralsa_linear[j*5+1]
                dihedralsa[j][2] = dihedralsa_linear[j*5+2]
                dihedralsa[j][3] = dihedralsa_linear[j*5+3]
                dihedralsa[j][4] = dihedralsa_linear[j*5+4]
        
        if lines[i][0:15] == '%FLAG ATOM_NAME':
            for j in range(n_name_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    atom_names.append(temp[k])

        if lines[i][0:21] == '%FLAG ATOM_TYPE_INDEX':
            for j in range(n_type_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    atom_type_index[j*10+k] = float(temp[k])

        if lines[i][0:26] == '%FLAG NONBONDED_PARM_INDEX':
            for j in range(n_nb_parm_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    nb_parm_index[j*10+k] = float(temp[k])

        if lines[i][0:25] == '%FLAG LENNARD_JONES_ACOEF':
            for j in range(n_lj_param_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    lj_a_coeff[j*5+k] = float(temp[k])
            #print lj_a_coeff
        if lines[i][0:25] == '%FLAG LENNARD_JONES_BCOEF':
            for j in range(n_lj_param_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    lj_b_coeff[j*5+k] = float(temp[k])


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
    coord = MDAnalysis.Universe(psf, coordDcd[0])
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


# compute distance squared between two points taking into account periodic boundary conditions
def computePbcDist2(r1, r2, box, hbox):
    dr = r1-r2 # points from r2 to r1
    for j in range(3):## 0, 1, 2
        if dr[j] < -hbox[j]:
            dr[j] += box[j] ## is the box automatically wrapped in .mdcrd files? Not when you put them in VMD! What if the distance is more than 1 box length away?
        elif dr[j] > hbox[j]:
            dr[j] -= box[j]
    dist2 = numpy.dot(dr,dr)
    return dist2,dr;


# Iterate through all pairs of particles in all simulations,
#    identifying each pair of particles, performing computations,
#    and storing the results in a data set
def iterate():
    global plots,dims,hdims
    hrMax=0.5*rMax
    #pH2O=numpy.zeros((binCount,binCount,binCount,3))
    nH2O=numpy.zeros((binCount,binCount,binCount))
    fH2O=numpy.zeros((binCount,binCount,binCount))

    if debug:
        print "-- Iterating through all particle pairs in first time step to establish pair types"

    total_frames = 0
    for igo in range(len(coordDcd)):
        coord.load_new(coordDcd[igo])
        total_frames += len(coord.trajectory)
        print "Now analyzing trajectory file: ", coordDcd[igo]
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
            # calculate dR the LJ--LJ distance.
            ljDist2,ljdr = computePbcDist2(ionsCoord.atoms[0].position, ionsCoord.atoms[1].position, dims, hdims)
            for a in H2OCoord.residues:
                dist2,dr = computePbcDist2(ionsCoord.atoms[0].position, a.atoms[1].position, dims, hdims)
                dr /= numpy.sqrt(dist2) # normalize dr (vector from solvent to solute)
                dist = numpy.sqrt(dist2)
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
                            #pNow=a.atoms[1].position-a.atoms[0].position
                            #pH2O[ix][iy][iz]+=pNow # pNow is in the Lab frame (unrotated frame, the frame of the box)
                            solvAtom_force_vec = numpy.zeros((3), dtype=float)
                            for i in a.atoms:
                                index = n_types*(atom_type_index[ionsCoord.atoms[0].index]-1) + atom_type_index[i.index] - 1 # the '-1' at the end is to zero-index the array because python arrays are zero-indexed and AMBER arrays are one-indexed.
                                nb_index = nb_parm_index[index] - 1 # same reason as above for the '- 1' at the end.
                                solvAtom_dist2,solvAtom_dr = computePbcDist2(ionsCoord.atoms[0].position, i.position, dims, hdims)
                                
                                # define r^(-6) for faster calculation of force
                                r6 = solvAtom_dist2**(-3) 
                                # force vector along solvAtom_dr (vector from solute ATOM to solvent ATOM)
                                solvAtom_force_vec += ( r6 * ( 12. * r6 * lj_a_coeff[nb_index] - 6. * lj_b_coeff[nb_index] ) / solvAtom_dist2 ) * solvAtom_dr

                            force_var = numpy.dot( numpy.dot(solvAtom_force_vec, dr)*dr, ljdr)/dist # project force from solvent ATOM onto vector from solvent RESIDUE
                            fH2O[ix][iy][iz] += force_var
#
    dxH2O=1/(binSize**3/dims[0]/dims[1]/dims[2]*len(H2OCoord.residues)*total_frames) # normalize by total # of frames
    outFile = open(outname+".gr3", 'w')
    for i in range(binCount):
        for j in range(binCount):
            for k in range(binCount):
                #if nH2O[i][j][k] != 0: # so i dont get NaNs
                    #pH2O[i][j][k]/=nH2O[i][j][k]*1.1 # normalize by number of bins and equilibrium bond value
                    #pH2O[i][j][k]=numpy.dot(axes,pH2O[i][j][k]) # rotate into aligned frame (rotated frame, the frame of the solute axes)
                outFile.write("{:7.3f} {:7.3f} {:7.3f} {:18.12f} {:18.12f}\n".format((i+0.5)*binSize-hrMax, (j+0.5)*binSize-hrMax, (k+0.5)*binSize-hrMax, nH2O[i][j][k]*dxH2O, fH2O[i][j][k]*dxH2O)) #,pH2O[i][j][k][0],pH2O[i][j][k][1],pH2O[i][j][k][2]))
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
    
    # Parse topology file
    ParsePrmtopBonded(psf)
    
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
