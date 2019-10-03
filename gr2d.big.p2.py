# python3
# Compute Radial Distribution Function as a Function of r, cos(theta), and phi [ Spread into 3D ] around an LJ sphere pair

import numpy as np
import sys
import os
import MDAnalysis
from math import *


def compute_pbc_dr(r1,r2,box,hbox):
    dr = r1 - r2
    if dr < -hbox:
        dr += box
    elif dr > hbox:
        dr -= box
    return dr;


## Read configuration file and populate global variables
def parse_config_file(cfgFile):
        global topFile, trajFile, outFile, histDistMin, histDistMax, binDistSize, T, soluteResname, solventResname, d
        trajFile = []
        f = open(cfgFile)
        for line in f:
            # first remove comments
            if '#' in line:
                line, comment = line.split('#',1)
            if '=' in line:
                option, value = line.split('=',1)
                option = option.strip()
                value = value.strip()
                print("Option:", option, " Value:", value)
                # check value
                if option.lower()=='topfile':
                    topFile = value
                elif option.lower()=='trajfile':
                    trajFile.append(value)
                elif option.lower()=='outfile':
                    outFile = value
                elif option.lower()=='hist_dist_min':
                    histDistMin = float(value)
                elif option.lower()=='hist_dist_max':
                    histDistMax = float(value)
                elif option.lower()=='bin_dist_size':
                    binDistSize = float(value)
                elif option.lower()=='temperature':
                    T = float(value)
                elif option.lower()=='solute_resname':
                    soluteResname = value
                elif option.lower()=='solvent_resname':
                    solventResname = value
                elif option.lower()=='offset':
                    d = float(value)
                else :
                    print("Option:", option, " is not recognized")

        # set some extra global variables
        global kT, histDistMin2, histDistMax2, nDistBins

        # Boltzmann Constant in kcal/mol.K
        k_B = 0.0019872041
        kT = k_B * T

        # Distances [in Angstroms]
        histDistMin2= histDistMin*histDistMin
        histDistMax2= histDistMax*histDistMax

        # Histogram bins
        nDistBins = int((histDistMax - histDistMin)/binDistSize)

        f.close()


## Read prmtop file and populate global variables
def parse_prmtop_bonded(topFile):
    global bond_fc,bond_equil_values,angle_fc,angle_equil_values,dihedral_fc,dihedral_period,dihedral_phase,nbonh,nbona,ntheta,ntheth,nphia,nphih,bondsh,bondsa,anglesh,anglesa,dihedralsh,dihedralsa,n_atoms,n_types,atom_names,atom_type_index,nb_parm_index,lj_a_coeff,lj_b_coeff
    
    param = open(topFile,'r')
    pointers = np.zeros(31,dtype=int)
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
            n_type_lines = int(ceil(n_atoms/10.))
            n_name_lines = int(ceil(n_atoms/20.))
            n_nb_parm_lines = int(ceil(n_types*n_types/10.))
            n_lj_param_lines = int(ceil((n_types*(n_types+1)/2)/5.))
            n_bond_lines = int(ceil(numbnd/5.))
            n_angle_lines = int(ceil(numang/5.))
            n_dihedral_lines = int(ceil(numtra/5.))
            n_bondsh_lines = int(ceil(nbonh*3/10.))
            n_bondsa_lines = int(ceil(nbona*3/10.))
            n_anglesh_lines = int(ceil(ntheth*4/10.))
            n_anglesa_lines = int(ceil(ntheta*4/10.))
            n_dihedralsh_lines = int(ceil(nphih*5/10.))
            n_dihedralsa_lines = int(ceil(nphia*5/10.))
            bond_fc = np.zeros(numbnd,dtype=float)
            bond_equil_values = np.zeros(numbnd,dtype=float)
            angle_fc = np.zeros(numang,dtype=float)
            angle_equil_values = np.zeros(numang,dtype=float)
            dihedral_fc = np.zeros(numtra,dtype=float)
            dihedral_period = np.zeros(numtra,dtype=float)
            dihedral_phase = np.zeros(numtra,dtype=float)
            SCEE_factor = np.zeros(numtra,dtype=float)
            SCNB_factor = np.zeros(numtra,dtype=float)
            bondsh_linear = np.zeros(3*nbonh,dtype=int)
            bondsa_linear = np.zeros(3*nbona,dtype=int)
            bondsh = np.zeros((nbonh,3),dtype=int)
            bondsa = np.zeros((nbona,3),dtype=int)
            anglesh_linear = np.zeros(4*ntheth,dtype=int)
            anglesa_linear = np.zeros(4*ntheta,dtype=int)
            anglesh = np.zeros((ntheth,4),dtype=int)
            anglesa = np.zeros((ntheta,4),dtype=int)
            dihedralsh_linear = np.zeros(5*nphih,dtype=int)
            dihedralsa_linear = np.zeros(5*nphia,dtype=int)
            dihedralsh = np.zeros((nphih,5),dtype=int)
            dihedralsa = np.zeros((nphia,5),dtype=int)
            atom_names = []
            atom_type_index = np.zeros((n_atoms),dtype=int)
            nb_parm_index = np.zeros(n_types*n_types,dtype=int)
            lj_a_coeff = np.zeros((n_types*(n_types+1))//2,dtype=float)
            lj_b_coeff = np.zeros((n_types*(n_types+1))//2,dtype=float)

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
        if lines[i][0:25] == '%FLAG LENNARD_JONES_BCOEF':
            for j in range(n_lj_param_lines):
                temp = lines[i+2+j].split()
                for k in range(len(temp)):
                    lj_b_coeff[j*5+k] = float(temp[k])


# initialize arrays for 2D g(r,cos(theta))
def initialize_arrays():
    Gc = np.zeros(nDistBins, dtype=int)
    Gr = np.zeros(nDistBins, dtype=float)
    Fr = np.zeros(nDistBins, dtype=float)
    Pr = np.zeros(nDistBins, dtype=float)
    return Gc, Gr, Fr, Pr;


# loop through trajectory
def iterate(Gc, Fr, Pr):
    u = MDAnalysis.Universe(topFile, trajFile[0]) # initiate MDAnalysis Universe.
    soluSel = u.select_atoms('resname ' + soluteResname)
    solvSel = u.select_atoms('resname ' + solventResname)
    nSolv = len(solvSel) # number of solvent atoms
    wrapPbc = np.vectorize(compute_pbc_dr) # wrap vector difference in pbc box
    a = soluSel.atoms[0] # single solute in big box
    for igo in range(len(trajFile)):
        u.load_new(trajFile[igo])
        print("Now analyzing trajectory file: ", trajFile[igo])
        for ts in u.trajectory: # Loop over all time steps in the trajectory.
            # Progress Bar
            sys.stdout.write("Progress: {0:.2f}% Complete\r".format((ts.frame+len(u.trajectory)*(igo))/(len(u.trajectory)*len(trajFile))*100))
            sys.stdout.flush()

            box = u.dimensions[:3]## define box and half box here so we save division calculation for every distance pair when we calculate 'dist' below.
            hbox = u.dimensions[:3]/2

            # Calculate r bin and force
            pCH = wrapPbc(solvSel.atoms[np.arange(0,nSolv,5)].positions, solvSel.atoms[np.arange(1,nSolv,5)].positions, box, hbox) # points C -> H
            np.divide( pCH, np.sqrt(np.einsum('ij,ij->i',pCH,pCH))[:,None], out=pCH) # normalize pCH
            rSolv = solvSel.atoms[np.arange(1,nSolv,5)].positions - d*pCH
            rSolv = wrapPbc(a.position, rSolv, box, hbox) # points from solvent to solute

            ir = np.where( np.einsum('ij,ij->i',rSolv,rSolv) >= histDistMax2 )
            rSolv = np.delete(rSolv,ir,axis=0)
            pCH = np.delete(pCH,ir,axis=0)

            rSolvDist = np.sqrt(np.einsum('ij,ij->i',rSolv,rSolv))
            distBin = np.ndarray.astype( rSolvDist/binDistSize, np.int)
            distBin[distBin == nDistBins] = -1 # fix any rounding errors
            pr = np.divide( np.einsum('ij,ij->i',pCH,rSolv), rSolvDist) # polarization in direction of rSolv

            # compute coulombic force atom-by-atom
            solvAtomChg = solvSel.atoms[np.arange(0,nSolv,1)].charges
            solvAtomPos = solvSel.atoms[np.arange(0,nSolv,1)].positions
            # Delete all atoms associated with residues that have been deleted previously above before calculating force to save time
            ira = np.append(np.append(np.append(np.append( ir[0]*5, ir[0]*5+1), ir[0]*5+2), ir[0]*5+3), ir[0]*5+4)  # list of atoms to remove based on "residue" distance removal list
            solvAtomChg = np.delete(solvAtomChg,ira,axis=0) # remove atoms that belong to previously deleted residues: distance
            solvAtomPos = np.delete(solvAtomPos,ira,axis=0) # remove atoms that belong to previously deleted residues: distance

            rSolvAtom = wrapPbc(a.position, solvAtomPos, box, hbox)
            rSolvAtomDist3 = np.power(np.sqrt(np.einsum('ij,ij->i',rSolvAtom,rSolvAtom)), 3)
            fSolvAtom = np.divide(( 332.05595 * a.charge)*solvAtomChg[:,None]*rSolvAtom, rSolvAtomDist3[:,None])
            fSolv = fSolvAtom[np.arange(0,len(fSolvAtom),5)] + fSolvAtom[np.arange(1,len(fSolvAtom),5)] + fSolvAtom[np.arange(2,len(fSolvAtom),5)] + fSolvAtom[np.arange(3,len(fSolvAtom),5)] + fSolvAtom[np.arange(4,len(fSolvAtom),5)] # summed force vector from residue atoms
            fr = np.divide(np.einsum('ij,ij->i',fSolv,rSolv), rSolvDist) # force along r

            np.add.at(Gc, distBin, 1)
            np.add.at(Fr, distBin, fr)
            np.add.at(Pr, distBin, pr)


def average(Gc, Fr, Pr):
    # Average LJ radial force for each distance
    np.divide(Fr, Gc, out=Fr, where=Gc[:]!=0)
    np.divide(Pr, Gc, out=Pr, where=Gc[:]!=0)


def volume_correct(Gc, Gr):
    ## Volume Correct
    for i in range(nDistBins):
        Gr[i] = Gc[i] / (4*pi*((i+0.5)*binDistSize + histDistMin)**2)


def normalize_Gr(Gr):
    ## Normalize
    ## have to normalize after volume correction because the 'bulk' g(r) value changes after volume correction.
    norm_points = 10
    # normalize by the last 'norm_points' distance points
    g_norm = 0.
    for i in range(norm_points):
        g_norm += Gr[-(i+1)]

    g_norm /= float(norm_points)

    for i in range(nDistBins):
        Gr[i] /= g_norm


def write_out(outFile, Gc, Gr, Fr, Pr):
    ## Open Output File
    out = open(outFile,'w')

    out.write("##  1: Distance Bin\n")
    out.write("##  2: g(r)\n")
    out.write("##  3: <force . r>\n")
    out.write("##  4: pol.r H->C\n")
    out.write("##  5: g(r) Counts\n")
    for i in range(nDistBins):
        out.write("{:7.3f} {:18.12f} {:18.12f} {:18.12f} {:8d}\n".format((i+0.5)*binDistSize+histDistMin, Gr[i], Fr[i], Pr[i], Gc[i]))

    ## Close Output File
    out.close


# main program
def mainLJ():
    # read in command line argument (cfg file)
    cfgFile = sys.argv[1]

    print('Reading input and initializing')
    # read cfg file
    parse_config_file(cfgFile)

    # parse the prmtop file 
    parse_prmtop_bonded(topFile)

    ##########
    # initialize 2D arrays

    # initialize with total dist, theta, and phi bins
    Gc,Gr,Fr,Pr = initialize_arrays()

    # loop through trajectory and calculate g(r,cos[theta]), force(r,cos[theta]), boltzmann(r,cos[theta])
    print('Looping through trajectory time steps...')
    iterate(Gc, Fr, Pr)

    # average the force and boltzmann by the g(r,cos[theta])
    print('Volume correcting...')
    average(Gc, Fr, Pr)

    # volume correct g(r,cos[theta])
    volume_correct(Gc, Gr)

    # normalize g(r,cos[theta])
    normalize_Gr(Gr)

    # write 2D output file: Gc, frc, boltz, integrated_force
    print('Write 2D output file')
    write_out(outFile, Gc, Gr, Fr, Pr)

    print('All Done!')


# Run Main program code.
mainLJ()
