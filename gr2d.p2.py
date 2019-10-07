# python3
# Compute Radial Distribution Function as a Function of r, cos(theta), and phi [ Spread into 3D ] around an LJ sphere pair

import numpy as np
import sys
import os
import MDAnalysis
from math import *


def cosPhi_2_phi(cosPhi):
    if cosPhi > 1:
        return 0
    elif cosPhi < -1:
        return np.pi
    else:
        return np.arccos(cosPhi)


def wrap_phi(phi):
    if phi >= pi23:
        return phi-pi23, 1;
    elif pi3 < phi < pi23:
        return pi23-phi, -1;
    else:
        return phi, 1;


def compute_pbc_dr(r1,r2,box,hbox):
    dr = r1 - r2
    if dr < -hbox:
        dr += box
    elif dr > hbox:
        dr -= box
    return dr;


def t_dot_rcl(dot,sign):
    if dot < 0:
        return sign*(-1);
    else:
        return sign;


## Read configuration file and populate global variables
def parse_config_file(cfgFile):
        global topFile, trajFile, outFile, histDistMin, histDistMax, binDistSize, histThetaMin, histThetaMax, binThetaSize, T, soluteResname, solventResname, d, histPhiMin, histPhiMax, binPhiSize, nAtomTypes
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
                elif option.lower()=='hist_theta_min':
                    histThetaMin = float(value)
                elif option.lower()=='hist_theta_max':
                    histThetaMax = float(value)
                elif option.lower()=='bin_theta_size':
                    binThetaSize = float(value)
                elif option.lower()=='hist_phi_min':
                    histPhiMin = float(value)
                elif option.lower()=='hist_phi_max':
                    histPhiMax = float(value)
                elif option.lower()=='bin_phi_size':
                    binPhiSize = float(value)
                elif option.lower()=='temperature':
                    T = float(value)
                elif option.lower()=='solute_resname':
                    soluteResname = value
                elif option.lower()=='solvent_resname':
                    solventResname = value
                elif option.lower()=='offset':
                    d = float(value)
                elif option.lower()=='number_solute_atoms':
                    nAtomTypes = int(value)
                else :
                    print("Option:", option, " is not recognized")

        # set some extra global variables
        global kT, histDistMin2, histDistMax2, nDistBins, nThetaBins, nPhiBins, pi23, pi3

        # Boltzmann Constant in kcal/mol.K
        k_B = 0.0019872041
        kT = k_B * T

        # Distances [in Angstroms]
        histDistMin2= histDistMin*histDistMin
        histDistMax2= histDistMax*histDistMax

        # Histogram bins
        nDistBins = int((histDistMax - histDistMin)/binDistSize)

        # Cosine Theta Histogram bins
        nThetaBins = int((histThetaMax - histThetaMin)/binThetaSize)
        # Phi Histogram bins
        nPhiBins = int((histPhiMax - histPhiMin)/binPhiSize)

        # global constants
        pi23 = 2*pi/3. # FIXME: this should be incorporated into the config file somehow. Like a radian of symmetry and half of that value.
        pi3 = pi/3.

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


# initialize arrays for 3D g(r,cos(theta),phi)
def initialize_arrays():
    # NOTE
    # g(r) array has both the g(r) values and the counts for each atomtype.
    # Gc[atomtypes, distbin, thetabin, phibin]
    Gc = np.zeros((nAtomTypes, nDistBins, nThetaBins, nPhiBins), dtype=float)
    Gr = np.zeros((nAtomTypes, nDistBins, nThetaBins, nPhiBins), dtype=float)
    # NOTE
    # Force array has the force, its square, the std.dev. of the force, and its square for each atomtype.
    # FrLJ[atomtypes, <f.r>/<f.s>/<f.t>, distbin, thetabin, phibin]
    FrLJ = np.zeros((nAtomTypes, 3, nDistBins, nThetaBins, nPhiBins), dtype=float)
    # FrC[atomtypes, <f.r>/<f.s>/<f.t>, distbin, thetabin, phibin]
    FrC = np.zeros((nAtomTypes, 3, nDistBins, nThetaBins, nPhiBins), dtype=float)
    Pr = np.zeros((nAtomTypes, nDistBins, nThetaBins, nPhiBins), dtype=float)
    return Gc, Gr, FrLJ, FrC, Pr;



# loop through trajectory
def iterate(Gc, FrLJ, FrC, Pr):
    u = MDAnalysis.Universe(topFile, trajFile[0]) # initiate MDAnalysis Universe.
    soluSel = u.select_atoms('resname ' + soluteResname)
    solvSel = u.select_atoms('resname ' + solventResname)
    nSolv = len(solvSel) # number of solvent atoms
    cosPhi2Phi = np.vectorize(cosPhi_2_phi) # cosPhi -> phi conditions
    wrapPhi = np.vectorize(wrap_phi) # phi -> wrapped phi conditions
    wrapPbc = np.vectorize(compute_pbc_dr) # wrap vector difference in pbc box
    tDotRcl = np.vectorize(t_dot_rcl) # change sign based on t vector orientation
    for igo in range(len(trajFile)):
        u.load_new(trajFile[igo])
        print("Now analyzing trajectory file: ", trajFile[igo])
        for ts in u.trajectory: # Loop over all time steps in the trajectory.
            # Progress Bar
            sys.stdout.write("Progress: {0:.2f}% Complete\r".format((ts.frame+len(u.trajectory)*(igo))/(len(u.trajectory)*len(trajFile))*100))
            sys.stdout.flush()

            box = u.dimensions[:3]## define box and half box here so we save division calculation for every distance pair when we calculate 'dist' below.
            hbox = u.dimensions[:3]/2

            # Compute all pairwise distances
            if nAtomTypes == 2:
                ljDr = wrapPbc(soluSel.atoms[0].position, soluSel.atoms[1].position, box, hbox)# Calculate the vector (ljDr) between the two LJ particles.
            for a in soluSel.atoms:
                # Calculate r,cosTh,phi bin
                pCH = wrapPbc(solvSel.atoms[np.arange(0,nSolv,5)].positions, solvSel.atoms[np.arange(1,nSolv,5)].positions, box, hbox)
                np.divide( pCH, np.sqrt(np.einsum('ij,ij->i',pCH,pCH))[:,None], out=pCH) # normalize pCH
                rSolv = solvSel.atoms[np.arange(1,nSolv,5)].positions - d*pCH
                rSolv = wrapPbc(a.position, rSolv, box, hbox)
                pCCl = wrapPbc(solvSel.atoms[np.arange(2,nSolv,5)].positions, solvSel.atoms[np.arange(1,nSolv,5)].positions, box, hbox)

                if nAtomTypes == 2:
                    if a.index == 0:
                        ip = np.where( np.dot(rSolv,ljDr)>0 ) # far side from solute
                    elif a.index == 1:
                        ip = np.where( np.dot(rSolv,ljDr)<0 ) # far side from solute
                    rSolv = np.delete(rSolv,ip,axis=0)
                    pCH = np.delete(pCH,ip,axis=0)
                    pCCl = np.delete(pCCl,ip,axis=0)

                ir = np.where( np.einsum('ij,ij->i',rSolv,rSolv) >= histDistMax2 )
                rSolv = np.delete(rSolv,ir,axis=0)
                pCH = np.delete(pCH,ir,axis=0)
                pCCl = np.delete(pCCl,ir,axis=0)

                rSolvDist = np.sqrt(np.einsum('ij,ij->i',rSolv,rSolv))
                cosTh = np.divide( np.einsum('ij,ij->i',pCH,rSolv), rSolvDist)

                nLJCH = np.cross(pCH, rSolv)
                nHCCl1 = np.cross(pCH, pCCl)
                cosPhi = np.divide( np.einsum('ij,ij->i',nLJCH,nHCCl1), np.sqrt(np.einsum('ij,ij->i',nLJCH,nLJCH)*np.einsum('ij,ij->i',nHCCl1,nHCCl1)) )
                cosPhi = cosPhi2Phi( cosPhi ) # apply cosPhi_2_phi to cosPhi, cosPhi is now converted into phi
                phi,sign = wrapPhi( cosPhi ) # apply wrap_phi to phi

                distBin = np.ndarray.astype( rSolvDist/binDistSize, np.int)
                thetaBin = np.ndarray.astype( (cosTh-histThetaMin)/binThetaSize, np.int)
                phiBin = np.ndarray.astype( (phi-histPhiMin)/binPhiSize, np.int)
                distBin[distBin == nDistBins] = -1
                thetaBin[thetaBin == nThetaBins] = -1
                phiBin[phiBin == nPhiBins] = -1

                # compute lj and coulomb force atom-by-atom
                solvAtomInd = solvSel.atoms[np.arange(0,nSolv,1)].indices
                solvAtomPos = solvSel.atoms[np.arange(0,nSolv,1)].positions
                solvAtomChg = solvSel.atoms[np.arange(0,nSolv,1)].charges
                # Delete all atoms associated with residues that have been deleted previously above before calculating force to save time
                if nAtomTypes == 2:
                    ipa = np.append(np.append(np.append(np.append( ip[0]*5, ip[0]*5+1), ip[0]*5+2), ip[0]*5+3), ip[0]*5+4)  # list of atoms to remove based on "residue" dot product removal list
                    solvAtomInd = np.delete(solvAtomInd,ipa,axis=0) # remove atoms that belong to previously deleted residues: dot product
                    solvAtomPos = np.delete(solvAtomPos,ipa,axis=0) # remove atoms that belong to previously deleted residues: dot product
                    solvAtomChg = np.delete(solvAtomChg,ipa,axis=0) # remove atoms that belong to previously deleted residues: dot product
                ira = np.append(np.append(np.append(np.append( ir[0]*5, ir[0]*5+1), ir[0]*5+2), ir[0]*5+3), ir[0]*5+4)  # list of atoms to remove based on "residue" distance removal list
                solvAtomInd = np.delete(solvAtomInd,ira,axis=0) # remove atoms that belong to previously deleted residues: distance
                solvAtomChg = np.delete(solvAtomChg,ira,axis=0) # remove atoms that belong to previously deleted residues: distance
                solvAtomPos = np.delete(solvAtomPos,ira,axis=0) # remove atoms that belong to previously deleted residues: distance
                index = n_types*(atom_type_index[a.index]-1) + atom_type_index[solvAtomInd]-1
                nbIndex = nb_parm_index[index]-1
                rSolvAtom = wrapPbc(a.position, solvAtomPos, box, hbox)
                rSolvAtomDist2 = np.einsum('ij,ij->i',rSolvAtom,rSolvAtom)
                r6 = np.power(rSolvAtomDist2, -3)
                # LJ
                fSolvAtomLJ = np.einsum('i,ij->ij',(r6 * (12*r6*lj_a_coeff[nbIndex] - 6*lj_b_coeff[nbIndex]) / rSolvAtomDist2), rSolvAtom) # force vectors
                fSolvLJ = fSolvAtomLJ[np.arange(0,len(fSolvAtomLJ),5)] + fSolvAtomLJ[np.arange(1,len(fSolvAtomLJ),5)] + fSolvAtomLJ[np.arange(2,len(fSolvAtomLJ),5)] + fSolvAtomLJ[np.arange(3,len(fSolvAtomLJ),5)] + fSolvAtomLJ[np.arange(4,len(fSolvAtomLJ),5)] # summed force vector from residue atoms
                # Coulomb
                fSolvAtomC = 332.05595 * a.charge * solvAtomChg[:,None] * rSolvAtom * np.sqrt(r6)[:,None]
                fSolvC = fSolvAtomC[np.arange(0,len(fSolvAtomC),5)] + fSolvAtomC[np.arange(1,len(fSolvAtomC),5)] + fSolvAtomC[np.arange(2,len(fSolvAtomC),5)] + fSolvAtomC[np.arange(3,len(fSolvAtomC),5)] + fSolvAtomC[np.arange(4,len(fSolvAtomC),5)] # summed force vector from residue atoms
                tSolv = np.cross(rSolv,pCH) # t vector
                np.divide(tSolv, np.sqrt(np.einsum('ij,ij->i',tSolv,tSolv))[:,None], out=tSolv) # normalize t vector
                tDot = np.einsum('ij,ij->i',tSolv,pCCl)
                sign = tDotRcl(tDot,sign)
                sSolv = np.cross(tSolv,rSolv) # s vector
                np.divide(sSolv, np.sqrt(np.einsum('ij,ij->i',sSolv,sSolv))[:,None], out=sSolv) # normalize s vector
                fLJr = np.einsum('ij,ij->i',fSolvLJ,rSolv)/rSolvDist # force along r
                fLJs = np.einsum('ij,ij->i',fSolvLJ,sSolv)
                fLJt = np.einsum('ij,ij->i',fSolvLJ,tSolv)
                fCr = np.einsum('ij,ij->i',fSolvC,rSolv)/rSolvDist # force along r
                fCs = np.einsum('ij,ij->i',fSolvC,sSolv)
                fCt = np.einsum('ij,ij->i',fSolvC,tSolv)
                pSolv = np.einsum('ij,ij->i',pCH,rSolv)/rSolvDist # polarization along r

                np.add.at(Gc[a.index], tuple(np.stack((distBin,thetaBin,phiBin))), 1)
                np.add.at(FrLJ[a.index][0], tuple(np.stack((distBin,thetaBin,phiBin))), fLJr)
                np.add.at(FrLJ[a.index][1], tuple(np.stack((distBin,thetaBin,phiBin))), fLJs)
                np.add.at(FrLJ[a.index][2], tuple(np.stack((distBin,thetaBin,phiBin))), fLJt)
                np.add.at(FrC[a.index][0], tuple(np.stack((distBin,thetaBin,phiBin))), fCr)
                np.add.at(FrC[a.index][1], tuple(np.stack((distBin,thetaBin,phiBin))), fCs)
                np.add.at(FrC[a.index][2], tuple(np.stack((distBin,thetaBin,phiBin))), fCt)
                np.add.at(Pr[a.index],tuple(np.stack((distBin,thetaBin,phiBin))), pSolv)


def average_Fr(Gc, FrLJ, FrC, Pr):
    # Average LJ radial force for each distance and cos(theta) bin
    for a in range(nAtomTypes):
        np.divide(FrLJ[a][0], Gc[a], out=FrLJ[a][0], where=Gc[a][:,:,:]!=0)
        np.divide(FrLJ[a][1], Gc[a], out=FrLJ[a][1], where=Gc[a][:,:,:]!=0)
        np.divide(FrLJ[a][2], Gc[a], out=FrLJ[a][2], where=Gc[a][:,:,:]!=0)
        np.divide(FrC[a][0], Gc[a], out=FrC[a][0], where=Gc[a][:,:,:]!=0)
        np.divide(FrC[a][1], Gc[a], out=FrC[a][1], where=Gc[a][:,:,:]!=0)
        np.divide(FrC[a][2], Gc[a], out=FrC[a][2], where=Gc[a][:,:,:]!=0)
        np.divide(Pr[a], Gc[a], out=Pr[a], where=Gc[a][:,:,:]!=0)


def volume_correct(Gc, Gr):
    ## Volume Correct
    for a in range(nAtomTypes):
        for i in range(nDistBins):
            for j in range(nThetaBins):
                for k in range(nPhiBins):
                    Gr[a, i, j, k] = Gc[a,i,j,k] / (4*pi*((i+0.5)*binDistSize + histDistMin)**2)


def normalize_Gr(Gr):
    ## Normalize
    ## have to normalize after volume correction because the 'bulk' g(r) value changes after volume correction.
    norm_points = 10
    for a in range(nAtomTypes):
        # normalize by the last 'norm_points' distance points
        g_norm = 0.
        for k in range(nPhiBins):
            for j in range(nThetaBins):
                for i in range(norm_points):
                    g_norm += Gr[a, -(i+1), j, k]

        g_norm /= float(norm_points*nThetaBins*nPhiBins)

        for k in range(nPhiBins):
            for j in range(nThetaBins):
                for i in range(nDistBins):
                    Gr[a, i, j, k] /= g_norm


def write_out_1(outFile, Gc, Gr, FrLJ, FrC, Pr):
    ## Open Output File
    out = open(outFile,'w')

    out.write("##  1: Distance Bin\n")
    out.write("##  2: Cos(theta) Bin\n")
    out.write("##  3: Phi/3 Bin\n")
    out.write("##  4: g(r)\n")
    out.write("##  5: <fLJ . r>\n")
    out.write("##  6: <fLJ . s>\n")
    out.write("##  7: <fLJ . t>\n")
    out.write("##  8: <fC . r>\n")
    out.write("##  9: <fC . s>\n")
    out.write("## 10: <fC . t>\n")
    out.write("## 11: g(r) Counts\n")
    out.write("## 12: p(r) H->C\n")
    for i in range(nDistBins):
        for j in range(nThetaBins):
            for k in range(nPhiBins):
                out.write("%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n" %((i+0.5)*binDistSize+histDistMin, (j+0.5)*binThetaSize+histThetaMin, (k+0.5)*binPhiSize+histPhiMin, Gr[0,i,j,k], FrLJ[0,0,i,j,k], FrLJ[0,1,i,j,k], FrLJ[0,2,i,j,k], FrC[0,0,i,j,k], FrC[0,1,i,j,k], FrC[0,2,i,j,k], Gc[0,i,j,k], -Pr[0,i,j,k]))

    ## Close Output File
    out.close


def write_out_2(outFile, Gc, Gr, FrLJ, FrC, Pr):
    ## Open Output File
    out = open(outFile,'w')

    out.write("##  1: Distance Bin\n")
    out.write("##  2: Cos(theta) Bin\n")
    out.write("##  3: Phi/3 Bin\n")
    out.write("##  4: g(r) +\n")
    out.write("##  5: g(r) -\n")
    out.write("##  6: <fLJ . r> +\n")
    out.write("##  7: <fLJ . s> +\n")
    out.write("##  8: <fLJ . t> +\n")
    out.write("##  9: <fLJ . r> -\n")
    out.write("## 10: <fLJ . s> -\n")
    out.write("## 11: <fLJ . t> -\n")
    out.write("## 12: <fC . r> +\n")
    out.write("## 13: <fC . s> +\n")
    out.write("## 14: <fC . t> +\n")
    out.write("## 15: <fC . r> -\n")
    out.write("## 16: <fC . s> -\n")
    out.write("## 17: <fC . t> -\n")
    out.write("## 18: g(r) Counts +\n")
    out.write("## 19: g(r) Counts -\n")
    out.write("## 20: p(r) + H->C\n")
    out.write("## 21: p(r) - H->C\n")
    for i in range(nDistBins):
        for j in range(nThetaBins):
            for k in range(nPhiBins):
                out.write("%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n" %((i+0.5)*binDistSize+histDistMin, (j+0.5)*binThetaSize+histThetaMin, (k+0.5)*binPhiSize+histPhiMin, Gr[0,i,j,k], Gr[1,i,j,k], FrLJ[0,0,i,j,k], FrLJ[0,1,i,j,k], FrLJ[0,2,i,j,k], FrLJ[1,0,i,j,k], FrLJ[1,1,i,j,k], FrLJ[1,2,i,j,k], FrC[0,0,i,j,k], FrC[0,1,i,j,k], FrC[0,2,i,j,k], FrC[1,0,i,j,k], FrC[1,1,i,j,k], FrC[1,2,i,j,k], Gc[0,i,j,k], Gc[1,i,j,k], -Pr[0,i,j,k], -Pr[1,i,j,k]))

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
    Gc,Gr,FrLJ,FrC,Pr = initialize_arrays()

    # loop through trajectory and calculate g(r,cos[theta]), force(r,cos[theta]), boltzmann(r,cos[theta])
    print('Looping through trajectory time steps...')
    iterate(Gc, FrLJ, FrC, Pr)

    # average the force and boltzmann by the g(r,cos[theta])
    print('Volume correcting...')
    average_Fr(Gc, FrLJ, FrC, Pr)

    # volume correct g(r,cos[theta])
    volume_correct(Gc, Gr)

    # normalize g(r,cos[theta])
    normalize_Gr(Gr)

    # write 2D output file: Gc, frc, boltz, integrated_force
    print('Write 2D output file')
    if nAtomTypes == 1:
        write_out_1(outFile, Gc, Gr, FrLJ, FrC, Pr)
    elif nAtomTypes == 2:
        write_out_2(outFile, Gc, Gr, FrLJ, FrC, Pr)

    print('All Done!')


# Run Main program code.
mainLJ()
