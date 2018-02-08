# Compute Radial Distribution Function as a Function of Cos(theta) [ Spread into 2D ]
# python 2.7

# CONFIG FILE FORMAT:

#   TopFile = [topology file name (prmtop file)]

#   TrajFile = [trajectory file name (mdcrd file)]

#   OutFile = [output data file name]

import numpy as np
import sys
import os
import MDAnalysis
import math

########################################################################
############################# Sub Routines #############################
########################################################################

## compute distance squared between two points taking into accound periodic boundary conditions

def computePbcDist2(r1,r2,box,hbox):
    dr = r1-r2
    for j in range(0,3):## 0, 1, 2
        if dr[j] < -hbox[j]:
            dr[j] += box[j] ## is the box automatically wrapped in .mdcrd files? Not when you put them in VMD! What if the distance is more than 1 box length away?
        elif dr[j] > hbox[j]:
            dr[j] -= box[j]
    dist2 = np.dot(dr,dr)
    return dist2,dr;


## compute g[r,cos(th)], <f.r>[r,cos(th)], and <boltzmann_factor_of_LJ_potential>[r,cos(th)]
##      dist2       =   distance squared from solute atom to solvent residue
##      dr          =   vector from solute atom to solvent residue
##      g_count     =   g[r,cos(theta)] count array
##      lj_force    =   average LJ force dotted into radial vector
##      lj_boltz    =   average LJ boltzmann factor
def computeGr(solute_atom,solvent_residue,dist2,dr,g_count,lj_force,lj_force2,lj_boltz,lj_boltz2):
    dist = math.sqrt(dist2)
    ## Calculate Polarization Magnitude Along Radial Vector From Solute
    pNow = solvent_residue.atoms[1].position - solvent_residue.atoms[0].position # dipole points from H to C in CL3.
    pMag = math.sqrt(np.dot(pNow,pNow)) # Magnitude of solvent dipole vector. So pRad has range from -1 to +1, ie just Cos(theta)
    pRad = np.dot(pNow,dr) / (dist * pMag) # Projection of normalized solvent dipole onto radial vector. (So just a magnitude)
    #
    dist_bin = int((dist - hist_dist_min)/bin_dist_size)
    ang_bin = int((pRad - hist_ang_min)/bin_ang_size)

    ## Calculate non_bonded_index and LJ interaction energy
    lj_force_vec = np.zeros((3),dtype=float)
    energy = 0.0
    for i in solvent_residue.atoms: # loop through atoms of the solvent molecule 'b'.
        index = n_types*(atom_type_index[solute_atom.index]-1) + atom_type_index[i.index] - 1 # the '-1' at the end is to zero-index the array because python arrays are zero-indexed and AMBER arrays are one-indexed.
        nb_index = nb_parm_index[index] - 1 # same reason as above for the '- 1' at the end.
        lj_dist2,lj_dr = computePbcDist2(solute_atom.position,i.position,box,hbox)
        #lj_dist = math.sqrt(lj_dist2)
        
        # define r^(-6) for faster calculation of force and energy
        r6 = lj_dist2**(-3) 
        # force vector along lj_dr (vector from solute ATOM to solvent ATOM)
        lj_force_vec += ( r6 * ( 12. * r6 * lj_a_coeff[nb_index] - 6. * lj_b_coeff[nb_index] ) / lj_dist2 ) * lj_dr
        # LJ energy of atoms summed
        energy += r6 * ( r6 * lj_a_coeff[nb_index] - lj_b_coeff[nb_index] ) 
        #print i,lj_a_coeff[nb_index],lj_b_coeff[nb_index]

    energy_var = np.exp( - energy / kT )
    force_var = np.dot( lj_force_vec, dr)/dist # project force from solvent ATOM onto vector from solvent RESIDUE
    ## Sum the LJ energy and _then_ put it in the exponential. Not a sum of the individual boltzmanns.
    if ang_bin == num_ang_bins:
        g_count[dist_bin][-1] += 1.

        lj_boltz[dist_bin][-1] += energy_var
        lj_boltz2[dist_bin][-1] += energy_var * energy_var

        lj_force[dist_bin][-1] += force_var
        lj_force2[dist_bin][-1] += force_var * force_var
    else:
        g_count[dist_bin][ang_bin] += 1.

        lj_boltz[dist_bin][ang_bin] += energy_var 
        lj_boltz2[dist_bin][ang_bin] += energy_var * energy_var 

        lj_force[dist_bin][ang_bin] += force_var
        lj_force2[dist_bin][ang_bin] += force_var * force_var



## Read configuration file and populate global variables

def ParseConfigFile(cfg_file):
        global top_file, traj_file, out_file, collapsed_file, hist_dist_min, hist_dist_max, bin_dist_size, hist_ang_min, hist_ang_max, bin_ang_size, T, system, solute_resname, solvent_resname, d
        f = open(cfg_file)
        for line in f:
                # first remove comments
                if '#' in line:
                        line, comment = line.split('#',1)
                if '=' in line:
                        option, value = line.split('=',1)
                        option = option.strip()
                        value = value.strip()
                        print "Option:", option, " Value:", value
                        # check value
                        if option.lower()=='topfile':
                                top_file = value
                        elif option.lower()=='trajfile':
                                traj_file = value
                        elif option.lower()=='outfile':
                                out_file = value
                        elif option.lower()=='collapsed_outfile':
                                collapsed_file = value
                        elif option.lower()=='hist_dist_min':
                                hist_dist_min = float(value)
                        elif option.lower()=='hist_dist_max':
                                hist_dist_max = float(value)
                        elif option.lower()=='bin_dist_size':
                                bin_dist_size = float(value)
                        elif option.lower()=='hist_ang_min':
                                hist_ang_min = float(value)
                        elif option.lower()=='hist_ang_max':
                                hist_ang_max = float(value)
                        elif option.lower()=='bin_ang_size':
                                bin_ang_size = float(value)
                        elif option.lower()=='temperature':
                                T = float(value)
                        elif option.lower()=='solute_resname':
                                solute_resname = value
                        elif option.lower()=='solvent_resname':
                                solvent_resname = value
                        elif option.lower()=='system':
                                system = value
                        elif option.lower()=='offset':
                                d = float(value)
                        else :
                                print "Option:", option, " is not recognized"
        f.close()

## Read prmtop file and populate global variables

def ParsePrmtopBonded(top_file):
	global bond_fc,bond_equil_values,angle_fc,angle_equil_values,dihedral_fc,dihedral_period,dihedral_phase,nbonh,nbona,ntheta,ntheth,nphia,nphih,bondsh,bondsa,anglesh,anglesa,dihedralsh,dihedralsa,n_atoms,n_types,atom_names,atom_type_index,nb_parm_index,lj_a_coeff,lj_b_coeff
	
	param = open(top_file,'r')
	pointers = np.zeros(31,dtype=np.int)
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
			bond_fc = np.zeros(numbnd,dtype=np.float)
			bond_equil_values = np.zeros(numbnd,dtype=np.float)
			angle_fc = np.zeros(numang,dtype=np.float)
			angle_equil_values = np.zeros(numang,dtype=np.float)
			dihedral_fc = np.zeros(numtra,dtype=np.float)
			dihedral_period = np.zeros(numtra,dtype=np.float)
			dihedral_phase = np.zeros(numtra,dtype=np.float)
			SCEE_factor = np.zeros(numtra,dtype=np.float)
			SCNB_factor = np.zeros(numtra,dtype=np.float)
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
			lj_a_coeff = np.zeros((n_types*(n_types+1))/2,dtype=float)
			lj_b_coeff = np.zeros((n_types*(n_types+1))/2,dtype=float)


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


########################################################################
############################# Main Program #############################
########################################################################

cfg_file = sys.argv[1] ## read in command line argument (cfg file)

ParseConfigFile(cfg_file) ## read cfg file

u = MDAnalysis.Universe(top_file, traj_file)## initiate MDAnalysis Universe.

solute_sel = u.select_atoms('resname ' + solute_resname)
solvent_sel = u.select_atoms('resname ' + solvent_resname)

ParsePrmtopBonded(top_file)

## Histogram Array
# Distances [in Angstroms]
hist_dist_min2= hist_dist_min*hist_dist_min
hist_dist_max2= hist_dist_max*hist_dist_max
# Histogram bins
num_dist_bins = int((hist_dist_max - hist_dist_min)/bin_dist_size)
## There is no need for bin_centers because int(5.9)=5 so bin distances are being assigned to the appropriate mid point

# Angles [cosine(theta)]
# Cosine Theta Histogram bins
num_ang_bins = int((hist_ang_max - hist_ang_min)/bin_ang_size)
# Boltzmann Constant in kcal/mol.K
k_B = 0.0019872041
global kT
kT = k_B * T


## Initialize arrays based on system
if system == "LJ":
    # Initialize 2D g(r) histogram array for + and - LJ particles
    g2d_0 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    g2d_1 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    gcount_0 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    gcount_1 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    # Initialize 2D <e^(-u/T)> array for + and - LJ particles
    lj_boltz_0 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_boltz2_0 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_boltz_SD_0 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_boltz_SD2_0 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_boltz_1 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_boltz2_1 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_boltz_SD_1 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_boltz_SD2_1 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    # Initialize 2D <f.r> array for + and - LJ particles
    lj_force_0 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_force2_0 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_force_SD_0 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_force_SD2_0 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_force_1 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_force2_1 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_force_SD_1 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_force_SD2_1 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
elif system == "MOL":
    # Initialize 2D g(r) histogram array
    g2d = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    gcount = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    # Initialize 2D <e^(-u/T)> array
    lj_boltz = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_boltz2 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_boltz_SD = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_boltz_SD2 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    # Initialize 2D <f.r> array
    lj_force = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_force2 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_force_SD = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
    lj_force_SD2 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)


## Loop through trajectory
for ts in u.trajectory:## Loop over all time steps in the trajectory.
    if ts.frame >= 0:## ts.frame is the index of the timestep. Increase this to exclude "equilibration" time?

	## Progress Bar
        sys.stdout.write("Progress: {0:.2f}% Complete\r".format((float(ts.frame) / float(len(u.trajectory))) * 100))
        sys.stdout.flush()

        box = u.dimensions[:3]## define box and half box here so we save division calculation for every distance pair when we calculate 'dist' below.
        hbox = u.dimensions[:3]/2.0

        ## Select system type and compute all pairwise distances
        if system == "LJ": # two LJ particle testing simulations.
            dist2,R12 = computePbcDist2(solute_sel.atoms[0].position,solute_sel.atoms[1].position,box,hbox)# Calculate the vector (R12) between the two LJ particles.
            for a in solute_sel.atoms:
                for b in solvent_sel.residues:
                    rSolv = (b.atoms[1].position + d * ((b.atoms[1].position - b.atoms[0].position)/1.1)) # solvent vector to excluded volume center.
                    ## Bin the solvent for g(r) only if the solvent is on the far side of the respective solute atom
                    if a.index == 0: # if first LJ particle is selected...
                        dist2,dr = computePbcDist2(a.position,rSolv,box,hbox) # distance between solute and excluded volume center of CL3
                        if dist2 <= hist_dist_max2:
                            if np.dot(R12,dr) < 0: # Is dr dot product pointing right direction? Then compute sqrt
                                if hist_dist_min2 < dist2 < hist_dist_max2:
                                    computeGr(a,b,dist2,dr,gcount_0,lj_force_0,lj_force2_0,lj_boltz_0,lj_boltz2_0)

                    elif a.index == 1: # if second LJ particle is selected...
                        dist2,dr = computePbcDist2(a.position,rSolv,box,hbox) # distance between solute and excluded volume center of CL3
                        if dist2 <= hist_dist_max2:
                            if np.dot(R12,dr) > 0: # Is dr dot product pointing right direction? Then compute sqrt
                                if hist_dist_min2 < dist2 < hist_dist_max2:
                                    computeGr(a,b,dist2,dr,gcount_1,lj_force_1,lj_force2_1,lj_boltz_1,lj_boltz2_1)

        elif system == "MOL": # a belly simulation of a molecule.
            for a in solute_sel.atoms:
                for b in solvent_sel.residues:
                    rSolv = (b.atoms[1].position + d * ((b.atoms[1].position - b.atoms[0].position)/1.1)) # solvent vector to excluded volume center. 1.1 is roughly the distance between the H and C in Chloroform.
                    dist2,dr = computePbcDist2(a.position,rSolv,box,hbox) # distance between solute and C of CL3
                    if hist_dist_min2 < dist2 < hist_dist_max2:
                        computeGr(a,b,dist2,dr,gcount,lj_force,lj_force2,lj_boltz,lj_boltz2)

        else:
            print 'Please choose a viable \"system\" option in the configuration file.'


## Set 'count' array
if system == 'LJ':
    for j in range(num_ang_bins):
        for i in range(num_dist_bins):
            g2d_0[i][j] = gcount_0[i][j]
            g2d_1[i][j] = gcount_1[i][j]
if system == 'MOL':
    for j in range(num_ang_bins):
        for i in range(num_dist_bins):
            g2d[i][j] = gcount[i][j]

## Average LJ radial force for each distance and cos(theta) bin
if system == 'LJ':
    for j in range(num_ang_bins):
        for i in range(num_dist_bins):
            if gcount_0[i][j] > 0.5:
                lj_force_0[i][j] /= gcount_0[i][j]
                lj_boltz_0[i][j] /= gcount_0[i][j]
                lj_force2_0[i][j] /= gcount_0[i][j]
                lj_boltz2_0[i][j] /= gcount_0[i][j]
    for j in range(num_ang_bins):
        for i in range(num_dist_bins):
            if gcount_1[i][j] > 0.5:
                lj_force_1[i][j] /= gcount_1[i][j]
                lj_boltz_1[i][j] /= gcount_1[i][j]
                lj_force2_1[i][j] /= gcount_1[i][j]
                lj_boltz2_1[i][j] /= gcount_1[i][j]
elif system == 'MOL':
    for j in range(num_ang_bins):
        for i in range(num_dist_bins):
            if gcount[i][j] > 0.5:
                lj_force[i][j] /= gcount[i][j]
                lj_boltz[i][j] /= gcount[i][j]
                lj_force2[i][j] /= gcount[i][j]
                lj_boltz2[i][j] /= gcount[i][j]

## Standard Deviation of every bin Force and Boltzmann using the variance.
if system == 'LJ':
    for j in range(num_ang_bins):
        for i in range(num_dist_bins):
            lj_force_SD2_0[i][j] = lj_force2_0[i][j] - lj_force_0[i][j]*lj_force_0[i][j]
            lj_boltz_SD2_0[i][j] = lj_boltz2_0[i][j] - lj_boltz_0[i][j]*lj_boltz_0[i][j]
            lj_force_SD_0[i][j] = np.sqrt( lj_force_SD2_0[i][j] )
            lj_boltz_SD_0[i][j] = np.sqrt( lj_boltz_SD2_0[i][j] )

            lj_force_SD2_1[i][j] = lj_force2_1[i][j] - lj_force_1[i][j]*lj_force_1[i][j]
            lj_boltz_SD2_1[i][j] = lj_boltz2_1[i][j] - lj_boltz_1[i][j]*lj_boltz_1[i][j]
            lj_force_SD_1[i][j] = np.sqrt( lj_force_SD2_1[i][j] )
            lj_boltz_SD_1[i][j] = np.sqrt( lj_boltz_SD2_1[i][j] )
if system == 'MOL':
    for j in range(num_ang_bins):
        for i in range(num_dist_bins):
            lj_force_SD2[i][j] = lj_force2[i][j] - lj_force[i][j]*lj_force[i][j]
            lj_boltz_SD2[i][j] = lj_boltz2[i][j] - lj_boltz[i][j]*lj_boltz[i][j]
            lj_force_SD[i][j] = np.sqrt( lj_force_SD2[i][j] )
            lj_boltz_SD[i][j] = np.sqrt( lj_boltz_SD2[i][j] )

## Volume Correct
if system == 'LJ':
    for i in range(num_dist_bins):
        g2d_0[i][:] /= 4*math.pi*((i+0.5)*bin_dist_size + hist_dist_min)**2
        g2d_1[i][:] /= 4*math.pi*((i+0.5)*bin_dist_size + hist_dist_min)**2
elif system == 'MOL':
    for i in range(num_dist_bins):
        g2d[i][:] /= 4*math.pi*((i+0.5)*bin_dist_size + hist_dist_min)**2

## Normalize
## have to normalize after volume correction because the 'bulk' g(r) value changes after volume correction.
if system == 'LJ':
    for j in range(num_ang_bins):
        for i in range(num_dist_bins):
            g2d_0[i][j] /= g2d_0[-1][j]
            g2d_1[i][j] /= g2d_1[-1][j]
elif system == 'MOL':
    for j in range(num_ang_bins):
        for i in range(num_dist_bins):
            g2d[i][j] /= g2d[-1][j]

## Convert Histogram into Probability Density
#dist_hist /= (200*sel.n_atoms*bin_dist_size)

## Integrate the direct Solute--Solvent force
u_dir_0 = np.zeros((num_dist_bins, num_ang_bins),dtype=float)
u_dir_1 = np.zeros((num_dist_bins, num_ang_bins),dtype=float)
for j in range(num_ang_bins):
    for i in range(1,num_dist_bins+1):
        if i == 1:
            u_dir_0[-i][j] = lj_force_0[-i][j] * bin_dist_size
            u_dir_1[-i][j] = lj_force_1[-i][j] * bin_dist_size
        else:
            u_dir_0[-i][j] = u_dir_0[-(i-1)][j] + lj_force_0[-i][j] * bin_dist_size
            u_dir_1[-i][j] = u_dir_1[-(i-1)][j] + lj_force_1[-i][j] * bin_dist_size



## Open Output File
out = open(out_file,'w')

if system == 'LJ':
    out.write("## 1: Distance Bin\n")
    out.write("## 2: Cos(theta) Bin\n")
    out.write("## 3: g(r) +\n")
    out.write("## 4: g(r) -\n")
    out.write("## 5: <force . r> +\n")
    out.write("## 6: <force . r> Std Dev +\n")
    out.write("## 7: <force . r> -\n")
    out.write("## 8: <force . r> Std Dev -\n")
    out.write("## 9: <boltzmann> +\n")
    out.write("## 10: <boltzmann> Std Dev +\n")
    out.write("## 11: <boltzmann> -\n")
    out.write("## 12: <boltzmann> Std Dev -\n")
    out.write("## 13: Integrated force +\n")
    out.write("## 14: Integrated force -\n")
    for i in range(num_dist_bins):
        for j in range(num_ang_bins):
            out.write("%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n" %((i+0.5)*bin_dist_size+hist_dist_min, (j+0.5)*bin_ang_size+hist_ang_min, g2d_0[i][j], g2d_1[i][j], lj_force_0[i][j], lj_force_SD_0[i][j], lj_force_1[i][j], lj_force_SD_1[i][j], lj_boltz_0[i][j], lj_boltz_SD_0[i][j], lj_boltz_1[i][j], lj_boltz_SD_1[i][j], u_dir_0[i][j], u_dir_1[i][j]))
elif system == 'MOL':
    out.write("## 1: Distance Bin\n")
    out.write("## 2: Cos(theta) Bin\n")
    out.write("## 3: g(r)\n")
    out.write("## 4: <force . r>\n")
    out.write("## 5: <force . r> Std Dev\n")
    out.write("## 6: <boltzmann>\n")
    out.write("## 7: <boltzmann> Std Dev\n")
    for i in range(num_dist_bins):
        for j in range(num_ang_bins):
            out.write("%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n" %((i+0.5)*bin_dist_size+hist_dist_min, (j+0.5)*bin_ang_size+hist_ang_min, g2d[i][j], lj_force[i][j], lj_force_SD[i][j], lj_boltz[i][j], lj_boltz_SD[i][j]))

## Close Output File
out.close


## Initialize collapsed arrays
if system == 'LJ':
    g0 = np.zeros(num_dist_bins,dtype=float)
    g1 = np.zeros(num_dist_bins,dtype=float)
    force_0 = np.zeros(num_dist_bins,dtype=float)
    force_1 = np.zeros(num_dist_bins,dtype=float)
    force_SD2_0 = np.zeros(num_dist_bins,dtype=float)
    force_SD2_1 = np.zeros(num_dist_bins,dtype=float)
    force_SD1_0 = np.zeros(num_dist_bins,dtype=float)
    force_SD1_1 = np.zeros(num_dist_bins,dtype=float)
    force_SD_0 = np.zeros(num_dist_bins,dtype=float)
    force_SD_1 = np.zeros(num_dist_bins,dtype=float)
    boltz_0 = np.zeros(num_dist_bins,dtype=float)
    boltz_1 = np.zeros(num_dist_bins,dtype=float)
    boltz_SD2_0 = np.zeros(num_dist_bins,dtype=float)
    boltz_SD2_1 = np.zeros(num_dist_bins,dtype=float)
    boltz_SD1_0 = np.zeros(num_dist_bins,dtype=float)
    boltz_SD1_1 = np.zeros(num_dist_bins,dtype=float)
    boltz_SD_0 = np.zeros(num_dist_bins,dtype=float)
    boltz_SD_1 = np.zeros(num_dist_bins,dtype=float)
if system == 'MOL':
    g = np.zeros(num_dist_bins,dtype=float)
    force = np.zeros(num_dist_bins,dtype=float)
    force_SD = np.zeros(num_dist_bins,dtype=float)
    boltz = np.zeros(num_dist_bins,dtype=float)
    boltz_SD = np.zeros(num_dist_bins,dtype=float)

## Sum the Cos[theta] columns to collapse 2D in to 1D
if system == 'LJ':
    for i in range(num_dist_bins):
        for j in range(num_ang_bins):
            # sum of weighting factors for each distance
            g0[i] += g2d_0[i][j]
            g1[i] += g2d_1[i][j]
            # sum of weighted forces
            force_0[i] += lj_force_0[i][j] * g2d_0[i][j]
            force_1[i] += lj_force_1[i][j] * g2d_1[i][j]
            # sum of weighted (forces)**2. (The first term in sigma**2)
            force_SD2_0[i] += lj_force2_0[i][j] * g2d_0[i][j]
            force_SD2_1[i] += lj_force2_1[i][j] * g2d_1[i][j]
            # (sum of weighted forces)**2. (The second term in sigma**2)
            force_SD1_0[i] += lj_force_0[i][j] * g2d_0[i][j]
            force_SD1_1[i] += lj_force_1[i][j] * g2d_1[i][j]

            # sum of weighted boltzmann factors
            boltz_0[i] += lj_boltz_0[i][j] * g2d_0[i][j]
            boltz_1[i] += lj_boltz_1[i][j] * g2d_1[i][j]
            # sum of weighted (boltzmanns**2). (The first term in sigma**2)
            boltz_SD2_0[i] += lj_boltz2_0[i][j] * g2d_0[i][j]
            boltz_SD2_1[i] += lj_boltz2_1[i][j] * g2d_1[i][j]
            # (sum of boltmanns)**2 weighted. (The second term in sigma**2)
            boltz_SD1_0[i] += lj_boltz_0[i][j] * g2d_0[i][j]
            boltz_SD1_1[i] += lj_boltz_1[i][j] * g2d_1[i][j]
        ## End of j loop
        if g0[i] > 0.5:
            # sum of weighted forces divided by sum of weights gives weighted average.
            force_0[i] /= g0[i]
            # Std Dev using variance with weighted averages.
            force_SD_0[i] = np.sqrt( ( force_SD2_0[i] / g0[i] ) - ( force_SD1_0[i] / g0[i] )**2 )

            # sum of weighted forces divided by sum of weights gives weighted average.
            boltz_0[i] /= g0[i]
            # Std Dev using variance with weighted averages.
            boltz_SD_0[i] = np.sqrt( ( boltz_SD2_0[i] / g0[i] ) - ( boltz_SD1_0[i] / g0[i] )**2 )

            # So that bulk value is == 1.
            g0[i] /= num_ang_bins
        if g1[i] > 0.5:
            # sum of weighted forces divided by sum of weights gives weighted average.
            force_1[i] /= g1[i]
            # Std Dev using variance with weighted averages.
            force_SD_1[i] = np.sqrt( ( force_SD2_1[i] / g1[i] ) - ( force_SD1_1[i] / g1[i] )**2 )

            # sum of weighted forces divided by sum of weights gives weighted average.
            boltz_1[i] /= g1[i]
            # Std Dev using variance with weighted averages.
            boltz_SD_1[i] = np.sqrt( ( boltz_SD2_1[i] / g1[i] ) - ( boltz_SD1_1[i] / g1[i] )**2 )

            # So that bulk value is == 1.
            g1[i] /= num_ang_bins

if system == 'MOL':
    for i in range(num_dist_bins):
        for j in range(num_ang_bins):
            g[i] += g2d[i][j] / num_ang_bins

            force[i] += lj_force[i][j] * g2d[i][j] / num_ang_bins
            force_SD[i] += lj_force_SD2[i][j]

            boltz[i] += lj_boltz[i][j] * g2d[i][j] / num_ang_bins
            boltz_SD[i] += lj_boltz_SD2[i][j]

        ## Square root the SDs to give sigma instead of sigma**2
        force_SD[i] = np.sqrt( force_SD[i] )

        boltz_SD[i] = np.sqrt( boltz_SD[i] )


## Open Collapsed Output File
out = open(collapsed_file,'w')

if system == 'LJ':
    out.write("## 1: Distance Bin\n")
    out.write("## 2: g(r) +\n")
    out.write("## 3: g(r) -\n")
    out.write("## 4: <force . r> +\n")
    out.write("## 5: <force . r> Std Dev +\n")
    out.write("## 6: <force . r> -\n")
    out.write("## 7: <force . r> Std Dev -\n")
    out.write("## 8: <boltzmann> +\n")
    out.write("## 9: <boltzmann> Std Dev +\n")
    out.write("## 10: <boltzmann> -\n")
    out.write("## 11: <boltzmann> Std Dev -\n")
    for i in range(num_dist_bins):
        out.write("%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n" %((i+0.5)*bin_dist_size+hist_dist_min, g0[i], g1[i], force_0[i], force_SD_0[i], force_1[i], force_SD_1[i], boltz_0[i], boltz_SD_0[i], boltz_1[i], boltz_SD_1[i]))
elif system == 'MOL':
    out.write("## 1: Distance Bin\n")
    out.write("## 2: g(r)\n")
    out.write("## 3: <force . r>\n")
    out.write("## 4: <force . r> Std Dev\n")
    out.write("## 5: <boltzmann>\n")
    out.write("## 6: <boltzmann> Std Dev\n")
    for i in range(num_dist_bins):
        out.write("%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n" %((i+0.5)*bin_dist_size+hist_dist_min, g[i], force[i], force_SD[i], boltz[i], boltz_SD[i]))

## Close Output File
out.close
