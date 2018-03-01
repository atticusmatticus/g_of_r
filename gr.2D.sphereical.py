# Compute Radial Distribution Function as a Function of Cos(theta) [ Spread into 2D ] around LJ sphere
# Elliptical solvents
# python 2.7

import numpy as np
import sys
import os
import MDAnalysis
import math

class gr2D():

    ## compute distance squared between two points taking into accound periodic boundary conditions
    def computePbcDist2(self, r1, r2, box, hbox):
        dr = r1-r2
        for j in range(3):## 0, 1, 2
            if dr[j] < -hbox[j]:
                dr[j] += box[j] ## is the box automatically wrapped in .mdcrd files? Not when you put them in VMD! What if the distance is more than 1 box length away?
            elif dr[j] > hbox[j]:
                dr[j] -= box[j]
        dist2 = np.dot(dr,dr)
        return dist2,dr;


    '''
    compute g[r,cos(th)], <f.r>[r,cos(th)], and <boltzmann_factor_of_LJ_potential>[r,cos(th)]
        dist2       =   distance squared from solute atom to solvent residue
        dr          =   vector from solute atom to solvent residue
        gr          =   g[r,cos(theta)] count array
        fr          =   average LJ force dotted into radial vector
        bz          =   average LJ boltzmann factor
    '''
    def computeGr(self, solute_atom, solvent_residue, dist2, dr, AtomType, box, hbox, Gr, Fr, Bz):
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
            lj_dist2,lj_dr = self.computePbcDist2(solute_atom.position, i.position, box, hbox)
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
            Gr[AtomType, :, dist_bin, -1] += 1.

            Bz[AtomType, 0, dist_bin, -1] += energy_var
            Bz[AtomType, 1, dist_bin, -1] += energy_var * energy_var

            Fr[AtomType, 0, dist_bin, -1] += force_var
            Fr[AtomType, 1, dist_bin, -1] += force_var * force_var
        else:
            Gr[AtomType, :, dist_bin, ang_bin] += 1.

            Bz[AtomType, 0, dist_bin, ang_bin] += energy_var
            Bz[AtomType, 1, dist_bin, ang_bin] += energy_var * energy_var

            Fr[AtomType, 0, dist_bin, ang_bin] += force_var
            Fr[AtomType, 1, dist_bin, ang_bin] += force_var * force_var



    ## Read configuration file and populate global variables
    def ParseConfigFile(self, cfg_file):
            global top_file, traj_file, out_file, collapsed_file, hist_dist_min, hist_dist_max, bin_dist_size, hist_ang_min, hist_ang_max, bin_ang_size, T, solute_resname, solvent_resname, d
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
                            elif option.lower()=='offset':
                                    d = float(value)
                            else :
                                    print "Option:", option, " is not recognized"
            f.close()


    ## Read prmtop file and populate global variables
    def ParsePrmtopBonded(self, top_file):
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


    # initilize arrays for 2D g(r,cos(theta))
    def initilizeArrays2D(self):
        # NOTE
        # g(r) array has both the g(r) values and the counts for each atomtype.
        # Gr[atomtypes, gr/count, distbin, angbin]
        Gr = np.zeros((nAtomTypes, 2, num_dist_bins, num_ang_bins), dtype=float)
        # NOTE
        # Boltzmann factor array has the factor, its square, the std.dev. and its square for each atomtype.
        # Bz[atomtypes, factor/factor**2/std.dev./std.dev.**2, distbin, angbin]
        Bz = np.zeros((nAtomTypes, 4, num_dist_bins, num_ang_bins), dtype=float)
        # NOTE
        # Force array has the force, its square, the std.dev. of the force, and its square for each atomtype.
        # Fr[atomtypes, force/force**2/std.dev./std.dev.**2, distbin, angbin]
        Fr = np.zeros((nAtomTypes, 4, num_dist_bins, num_ang_bins), dtype=float)

        return Gr, Fr, Bz;

    # Loop through trajectory
    def iterate(self, Gr, Fr, Bz):
        ## initiate MDAnalysis Universe.
        u = MDAnalysis.Universe(top_file, traj_file)
        solute_sel = u.select_atoms('resname ' + solute_resname)
        solvent_sel = u.select_atoms('resname ' + solvent_resname)

        for ts in u.trajectory:## Loop over all time steps in the trajectory.
            if ts.frame >= 0:## ts.frame is the index of the timestep. Increase this to exclude "equilibration" time.

                ## Progress Bar
                sys.stdout.write("Progress: {0:.2f}% Complete\r".format((float(ts.frame) / float(len(u.trajectory))) * 100))
                sys.stdout.flush()

                box = u.dimensions[:3]## define box and half box here so we save division calculation for every distance pair when we calculate 'dist' below.
                hbox = u.dimensions[:3]/2.0

                ## Compute all pairwise distances
                dist2,R12 = self.computePbcDist2(solute_sel.atoms[0].position, solute_sel.atoms[1].position, box, hbox)# Calculate the vector (R12) between the two LJ particles.
                for a in solute_sel.atoms:
                    for b in solvent_sel.residues:
                        rSolv = (b.atoms[1].position + d * ((b.atoms[1].position - b.atoms[0].position)/1.1)) # solvent vector to excluded volume center.
                        ## Bin the solvent for g(r) only if the solvent is on the far side of the respective solute atom
                        if a.index == 0: # if first LJ particle is selected...
                            dist2,dr = self.computePbcDist2(a.position, rSolv, box, hbox) # distance between solute and excluded volume center of CL3
                            if dist2 <= hist_dist_max2:
                                if np.dot(R12,dr) < 0: # Is dr dot product pointing right direction? Then compute sqrt
                                    if hist_dist_min2 < dist2 < hist_dist_max2:
                                        self.computeGr(a, b, dist2, dr, a.index, box, hbox, Gr, Fr, Bz)

                        elif a.index == 1: # if second LJ particle is selected...
                            dist2,dr = self.computePbcDist2(a.position, rSolv, box, hbox) # distance between solute and excluded volume center of CL3
                            if dist2 <= hist_dist_max2:
                                if np.dot(R12,dr) > 0: # Is dr dot product pointing right direction? Then compute sqrt
                                    if hist_dist_min2 < dist2 < hist_dist_max2:
                                        self.computeGr(a, b, dist2, dr, a.index, box, hbox, Gr, Fr, Bz)


    def averageFrBz(self, Gr, Fr, Bz):
        ## Average LJ radial force for each distance and cos(theta) bin
        for a in range(nAtomTypes):
            for j in range(num_ang_bins):
                for i in range(num_dist_bins):
                    if Gr[a, 1, i, j] > 0.5:
                        Fr[a, 0, i, j] /= Gr[a, 1, i, j]
                        Fr[a, 1, i, j] /= Gr[a, 1, i, j]
                        Bz[a, 0, i, j] /= Gr[a, 1, i, j]
                        Bz[a, 1, i, j] /= Gr[a, 1, i, j]

    def calculateSD(self, Fr, Bz):
        ## Standard Deviation of every bin Force and Boltzmann using the variance.
        for a in range(nAtomTypes):
            for j in range(num_ang_bins):
                for i in range(num_dist_bins):
                    Fr[a, 3, i, j] = Fr[a, 1, i, j] - Fr[a, 0, i, j]*Fr[a, 0, i, j]
                    Bz[a, 3, i, j] = Bz[a, 1, i, j] - Bz[a, 0, i, j]*Bz[a, 0, i, j]
                    Fr[a, 2, i, j] = np.sqrt( Fr[a, 3, i, j] )
                    Bz[a, 2, i, j] = np.sqrt( Bz[a, 3, i, j] )

    def volumeCorrect(self, Gr):
        ## Volume Correct
        for a in range(nAtomTypes):
            for i in range(num_dist_bins):
                Gr[a, 0, i, :] /= 4*math.pi*((i+0.5)*bin_dist_size + hist_dist_min)**2

    def normalizeGr(self, Gr):
        ## Normalize
        ## have to normalize after volume correction because the 'bulk' g(r) value changes after volume correction.
        for a in range(nAtomTypes):
            for j in range(num_ang_bins):
                for i in range(num_dist_bins):
                    Gr[a, 0, i, j] /= Gr[a, 0, -1, j]


    def integrateDirectSoluteSolventFrc(self, Fr):
        ## Integrate the direct Solute--Solvent force
        U_dir = np.zeros((nAtomTypes, num_dist_bins, num_ang_bins), dtype=float)
        for a in range(nAtomTypes):
            for j in range(num_ang_bins):
                for i in range(1,num_dist_bins+1):
                    if i == 1:
                        U_dir[a, -i, j] = Fr[a, 0, -i, j] * bin_dist_size
                    else:
                        U_dir[a, -i, j] = U_dir[a, -(i-1), j] + Fr[a, 0, -i, j] * bin_dist_size
        return U_dir;


    def writeOutputGr2D(self, out_file, Gr, Fr, Bz):
        ## Open Output File
        out = open(out_file,'w')

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
                out.write("%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n" %((i+0.5)*bin_dist_size+hist_dist_min, (j+0.5)*bin_ang_size+hist_ang_min, Gr[0,0,i,j], Gr[1,0,i,j], Fr[0,0,i,j], Fr[0,2,i,j], Fr[1,0,i,j], Fr[1,2,i,j], Bz[0,0,i,j], Bz[0,2,i,j], Bz[1,0,i,j], Bz[1,2,i,j], U_dir[0,i,j], U_dir[1,i,j]))

        ## Close Output File
        out.close


    ## Initialize collapsed arrays
    def initilizeArraysCollapsed(self):
        # NOTE
        # g_collapsed(r) array has the g(r).
        # GrC[atomtypes, gr, distbin]
        GrC = np.zeros((nAtomTypes, num_dist_bins), dtype=float)
        # NOTE
        # Boltzmann factor collapsed array has the factor, its square, the std.dev. and its square for each atomtype.
        # BzC[atomtypes, factor/std.dev./std.dev.**2, distbin]
        BzC = np.zeros((nAtomTypes, 3, num_dist_bins), dtype=float)
        # NOTE
        # Force array has the force, its square, the std.dev. of the force, and its square for each atomtype.
        # Fr[atomtypes, force/std.dev./std.dev.**2, distbin]
        FrC = np.zeros((nAtomTypes, 3, num_dist_bins), dtype=float)

        return GrC, FrC, BzC;


    ## Sum the Cos[theta] columns to collapse 2D into 1D
    def collapseArrays(self, GrC, FrC, BzC):
        for a in range(nAtomTypes):
            for i in range(num_dist_bins):
                for j in range(num_ang_bins):
                    # sum of weighting factors for each distance
                    GrC[a, i] += Gr[a, 0, i, j]
                    # sum of weighted forces
                    FrC[a, 0, i] += Fr[a, 0, i, j] * Gr[a, 0, i, j]
                    # sum of weighted (forces)**2. (The first term in sigma**2)
                    FrC[a, 2, i] += Fr[a, 1, i, j] * Gr[a, 0, i, j]
                    # (sum of weighted forces)**2. (The second term in sigma**2)
                    FrC[a, 1, i] += Fr[a, 0, i, j] * Gr[a, 0, i, j]

                    # sum of weighted boltzmann factors
                    BzC[a, 0, i] += Bz[a, 0, i, j] * Gr[a, 0, i, j]
                    # sum of weighted (boltzmanns**2). (The first term in sigma**2)
                    BzC[a, 2, i] += Bz[a, 1, i, j] * Gr[a, 0, i, j]
                    # (sum of boltmanns)**2 weighted. (The second term in sigma**2)
                    BzC[a, 1, i] += Bz[a, 0, i, j] * Gr[a, 0, i, j]
                ## XXX End of j loop

                if GrC[a, i] > 0.5:
                    # sum of weighted forces divided by sum of weights gives weighted average.
                    FrC[a, 0, i] /= GrC[a, i]
                    # Std Dev using variance with weighted averages.
                    FrC[a, 1, i] = np.sqrt( ( FrC[a, 2, i] / GrC[a, i] ) - ( FrC[a, 1, i] / GrC[a, i] )**2 )

                    # sum of weighted forces divided by sum of weights gives weighted average.
                    BzC[a, i] /= GrC[a, i]
                    # Std Dev using variance with weighted averages.
                    BzC[a, 1, i] = np.sqrt( ( BzC[a, 2, i] / GrC[a, i] ) - ( BzC[a, 1, i] / GrC[a, i] )**2 )

                    # So that bulk value is == 1.
                    GrC[a, i] /= num_ang_bins


    def writeOutputCollapsed(self, collapsed_file, GrC, FrC, BzC):
        ## Open Collapsed Output File
        out = open(collapsed_file,'w')

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
            out.write("%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n" %((i+0.5)*bin_dist_size+hist_dist_min, GrC[0,i], GrC[1,i], FrC[0,0,i], FrC[0,1,i], FrC[1,0,i], FrC[1,1,i], BzC[0,0,i], BzC[0,1,i], BzC[1,0,i], BzC[1,1,i]))

        ## Close Output File
        out.close



    # main program
    def mainLJ(self):
        # read in command line argument (cfg file)
        cfg_file = sys.argv[1]

        print 'Reading input and initilizing'
        # read cfg file
        self.ParseConfigFile(cfg_file)

        # set some extra global variables
        global kT, hist_dist_min, hist_dist_min2, hist_dist_max, hist_dist_max2, num_dist_bins, num_ang_bins
        # Boltzmann Constant in kcal/mol.K
        k_B = 0.0019872041
        kT = k_B * T
        # Distances [in Angstroms]
        hist_dist_min2= hist_dist_min*hist_dist_min
        hist_dist_max2= hist_dist_max*hist_dist_max
        # Histogram bins
        num_dist_bins = int((hist_dist_max - hist_dist_min)/bin_dist_size)
        # Cosine Theta Histogram bins
        num_ang_bins = int((hist_ang_max - hist_ang_min)/bin_ang_size)

        # parse the prmtop file 
        self.ParsePrmtopBonded(top_file)

        # initilize 2D arrays
        global nAtomTypes
        nAtomTypes = 2 # Number of solute atom types (+1z,-1z)
        Gr,Fr,Bz = self.initilizeArrays2D()

        # loop through trajectory and calculate g(r,cos[theta]), force(r,cos[theta]), boltzmann(r,cos[theta])
        print 'Looping through trajectory time steps...'
        self.iterate(Gr, Fr, Bz)

        # average the force and boltzmann by the g(r,cos[theta])
        print 'Calculating standard deviations and volume correcting...'
        self.averageFrBz(Gr, Fr, Bz)

        # calculate standard deviation of force and boltzmann
        self.calculateSD(Fr, Bz)

        # volume correct g(r,cos[theta])
        self.volumeCorrect(Gr)

        # normalize g(r,cos[theta])
        self.normalizeGr(Gr)

        # integrate the direct Solute--Solvent force
        U_dir = self.integrateDirectSoluteSolventFrc(Fr)

        # write 2D output files: gr, frc, boltz, integrated_force
        print 'Write 2D output files'
        self.writeOutputGr2D(out_file, Gr, Fr, Bz)
        print 'Done with 2D'





# Run Main program code.
g2d = gr2D()

g2d.mainLJ()

