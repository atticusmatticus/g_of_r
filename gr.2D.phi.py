# Compute Radial Distribution Function as a Function of r, cos(theta), and phi [ Spread into 3D ] around and LJ sphere
# python 2.7

import numpy as np
import sys
import os
import MDAnalysis
from math import *

class gr2D():

    # compute distance squared between two points taking into account periodic boundary conditions
    def computePbcDist2(self, r1, r2, box, hbox):
        dr = r1-r2 # points from r2 to r1
        for j in range(3):## 0, 1, 2
            if dr[j] < -hbox[j]:
                dr[j] += box[j] ## is the box automatically wrapped in .mdcrd files? Not when you put them in VMD! What if the distance is more than 1 box length away?
            elif dr[j] > hbox[j]:
                dr[j] -= box[j]
        dist2 = np.dot(dr,dr)
        return dist2,dr;


    # compute g[r,cos(th)], <f.r>[r,cos(th)], and <boltzmann_factor_of_LJ_potential>[r,cos(th)]
    def computeGr(self, solute_atom, solvent_residue, solvRes_dist2, solvRes_dr, AtomType, box, hbox, Gr, Fr, Bz):
        solvRes_dist = sqrt(solvRes_dist2)
        ## Calculate Polarization Magnitude Along Radial Vector From Solute
        pCH = solvent_residue.atoms[0].position - solvent_residue.atoms[1].position # dipole points from C to H in CL3.
        pCH_norm = sqrt(np.dot(pCH,pCH)) # Magnitude of solvent dipole vector.
        cosTh = np.dot(pCH,solvRes_dr) / (solvRes_dist * pCH_norm) # Projection of two normalized vectors == cos[theta]
        #
        pCCl = solvent_residue.atoms[2].position - solvent_residue.atoms[1].position # vector points from C to Cl1 in CL3.
        pCCl_norm = sqrt(np.dot(pCCl,pCCl)) # Magnitude of solvent CCl vector.
        # XXX: what i want is the angle between the LJ-C-H plane and the H-C-Cl1 plane. Find the angle between planes by measuring the angle between the plane normal vectors.
        n_a = np.cross(pCH, solvRes_dr) # vector normal to the LJ-C-H plane
        n_b = np.cross(pCH, pCCl) # vector normal to the H-C-Cl1 plane
        cosPhi = ( np.dot(n_a,n_b) / ( sqrt( np.dot(n_a,n_a)*np.dot(n_b,n_b) ) ) )
        if cosPhi > 1.0:
            phi = 0.
        elif cosPhi < -1.0:
            phi = pi
        else:
            phi = acos( cosPhi )

        if phi >= pi23:
            phi -= pi23
            sign = 1
        elif pi3 < phi < pi23:
            phi = pi23 - phi
            sign = -1
        else:
            sign = 1

        #
        # bin for spherical
        dist_bin = int((solvRes_dist - hist_dist_min)/bin_dist_size)
        theta_bin = int((cosTh - hist_theta_min)/bin_theta_size)
        phi_bin = int((phi - hist_phi_min)/bin_phi_size)
        if theta_bin == num_theta_bins:
            theta_bin = -1
        if phi_bin == num_phi_bins:
            phi_bin = -1

        ## Calculate non_bonded_index, LJ interaction energy, and the LJ force
        solvAtom_force_vec = np.zeros((3),dtype=float)
        energy = 0.0
        for i in solvent_residue.atoms: # loop through atoms of the solvent molecule 'b'.
            index = n_types*(atom_type_index[solute_atom.index]-1) + atom_type_index[i.index] - 1 # the '-1' at the end is to zero-index the array because python arrays are zero-indexed and AMBER arrays are one-indexed.
            nb_index = nb_parm_index[index] - 1 # same reason as above for the '- 1' at the end.
            solvAtom_dist2,solvAtom_dr = self.computePbcDist2(solute_atom.position, i.position, box, hbox)
            
            # define r^(-6) for faster calculation of force and energy
            r6 = solvAtom_dist2**(-3) 
            # force vector along solvAtom_dr (vector from solute ATOM to solvent ATOM)
            solvAtom_force_vec += ( r6 * ( 12. * r6 * lj_a_coeff[nb_index] - 6. * lj_b_coeff[nb_index] ) / solvAtom_dist2 ) * solvAtom_dr
            # LJ energy of atoms summed
            energy += r6 * ( r6 * lj_a_coeff[nb_index] - lj_b_coeff[nb_index] ) 

        force_r = np.dot( solvAtom_force_vec, solvRes_dr)/solvRes_dist # project force from solvent ATOM onto vector from solvent RESIDUE
        t_vec = np.cross(solvRes_dr, pCH)
        t_norm = sqrt(np.dot(t_vec,t_vec))
        tdotrcl = np.dot(t_vec,pCCl)#/(t_norm*pCCl_norm)
        if tdotrcl < 0:
            sign = -sign
        #print "t:", t_vec, sqrt(np.dot(t_vec,t_vec))/t_norm
        s_vec = np.cross(t_vec, solvRes_dr)
        s_norm = sqrt(np.dot(s_vec,s_vec))
        #print "s, p:", s_vec, pCH/pCH_norm, sqrt(np.dot(s_vec,s_vec))
        #print "r:", solvRes_dr/solvRes_dist
        #print "r.s", np.dot(solvRes_dr/solvRes_dist,s_vec/s_norm)
        #print "r.t", np.dot(solvRes_dr/solvRes_dist,t_vec/t_norm)
        #print "s.t", np.dot(s_vec/s_norm,t_vec/t_norm)
        #print "\n"
        force_s = np.dot(solvAtom_force_vec, s_vec)/s_norm # force projected along s, orthogonal to r and coplanar with p.
        force_t = sign*np.dot(solvAtom_force_vec, t_vec)/t_norm # force projected along t, orthogonal to r and s and p.
        #
        energy_var = exp( - energy / kT )

        ## Sum the LJ energy and _then_ put it in the exponential. Not a sum of the individual boltzmanns.
        Gr[AtomType, :, dist_bin, theta_bin, phi_bin] += 1.

        Bz[AtomType, 0, dist_bin, theta_bin, phi_bin] += energy_var
        Bz[AtomType, 1, dist_bin, theta_bin, phi_bin] += energy_var * energy_var

        Fr[AtomType, 0, 0, dist_bin, theta_bin, phi_bin] += force_r
        Fr[AtomType, 0, 1, dist_bin, theta_bin, phi_bin] += force_r * force_r
        Fr[AtomType, 1, 0, dist_bin, theta_bin, phi_bin] += force_s
        Fr[AtomType, 1, 1, dist_bin, theta_bin, phi_bin] += force_s * force_s
        Fr[AtomType, 2, 0, dist_bin, theta_bin, phi_bin] += force_t
        Fr[AtomType, 2, 1, dist_bin, theta_bin, phi_bin] += force_t * force_t


    ## Read configuration file and populate global variables
    def ParseConfigFile(self, cfg_file):
            global top_file, traj_file, out_file, collapsed_file, hist_dist_min, hist_dist_max, bin_dist_size, hist_theta_min, hist_theta_max, bin_theta_size, T, solute_resname, solvent_resname, d, hist_phi_min, hist_phi_max, bin_phi_size
            traj_file = []
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
                        traj_file.append(value)
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
                    elif option.lower()=='hist_theta_min':
                        hist_theta_min = float(value)
                    elif option.lower()=='hist_theta_max':
                        hist_theta_max = float(value)
                    elif option.lower()=='bin_theta_size':
                        bin_theta_size = float(value)
                    elif option.lower()=='hist_phi_min':
                        hist_phi_min = float(value)
                    elif option.lower()=='hist_phi_max':
                        hist_phi_max = float(value)
                    elif option.lower()=='bin_phi_size':
                        bin_phi_size = float(value)
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

            # set some extra global variables
            global kT, hist_dist_min2, hist_dist_max2, num_dist_bins, num_theta_bins, num_phi_bins, pi23, pi3

            # Boltzmann Constant in kcal/mol.K
            k_B = 0.0019872041
            kT = k_B * T

            # Distances [in Angstroms]
            hist_dist_min2= hist_dist_min*hist_dist_min
            hist_dist_max2= hist_dist_max*hist_dist_max

            # Histogram bins
            num_dist_bins = int((hist_dist_max - hist_dist_min)/bin_dist_size)

            # Cosine Theta Histogram bins
            num_theta_bins = int((hist_theta_max - hist_theta_min)/bin_theta_size)
            # Phi Histogram bins
            num_phi_bins = int((hist_phi_max - hist_phi_min)/bin_phi_size)

            # global constants
            pi23 = 2*pi/3. # FIXME: this should be incorporated into the config file somehow. Like a radian of symmetry and half of that value.
            pi3 = pi/3.

            f.close()


    ## Read prmtop file and populate global variables
    def ParsePrmtopBonded(self, top_file):
            global bond_fc,bond_equil_values,angle_fc,angle_equil_values,dihedral_fc,dihedral_period,dihedral_phase,nbonh,nbona,ntheta,ntheth,nphia,nphih,bondsh,bondsa,anglesh,anglesa,dihedralsh,dihedralsa,n_atoms,n_types,atom_names,atom_type_index,nb_parm_index,lj_a_coeff,lj_b_coeff
            
            param = open(top_file,'r')
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
                    if lines[i][0:25] == '%FLAG LENNARD_JONES_BCOEF':
                            for j in range(n_lj_param_lines):
                                    temp = lines[i+2+j].split()
                                    for k in range(len(temp)):
                                            lj_b_coeff[j*5+k] = float(temp[k])


    # initialize arrays for 2D g(r,cos(theta))
    def initializeArrays2D(self):
        # NOTE
        # g(r) array has both the g(r) values and the counts for each atomtype.
        # Gr[atomtypes, gr/count, distbin, thetabin, phibin]
        Gr = np.zeros((nAtomTypes, 2, num_dist_bins, num_theta_bins, num_phi_bins), dtype=float)
        # NOTE
        # Boltzmann factor array has the factor, its square, the std.dev. and its square for each atomtype.
        # Bz[atomtypes, factor/factor**2/std.dev./std.dev.**2, distbin, thetabin, phibin]
        Bz = np.zeros((nAtomTypes, 4, num_dist_bins, num_theta_bins, num_phi_bins), dtype=float)
        # NOTE
        # Force array has the force, its square, the std.dev. of the force, and its square for each atomtype.
        # Fr[atomtypes, <f.r>/<f.s>/<f.t>, force/force**2/std.dev./std.dev.**2, distbin, thetabin, phibin]
        Fr = np.zeros((nAtomTypes, 3, 4, num_dist_bins, num_theta_bins, num_phi_bins), dtype=float)
        # NOTE
        # direct interaction potential
        U_dir = np.zeros((nAtomTypes, num_dist_bins, num_theta_bins, num_phi_bins), dtype=float)

        return Gr, Fr, Bz, U_dir;



    # loop through trajectory
    def iterate(self, Gr, Fr, Bz):
        ## initiate MDAnalysis Universe.
        u = MDAnalysis.Universe(top_file, traj_file[0])
        solute_sel = u.select_atoms('resname ' + solute_resname)
        solvent_sel = u.select_atoms('resname ' + solvent_resname)
        for igo in range(len(traj_file)):
            u.load_new(traj_file[igo])
            print "Now analyzing trajectory file: ", traj_file[igo]
            for ts in u.trajectory:## Loop over all time steps in the trajectory.
                if ts.frame >= 0:## ts.frame is the index of the timestep. Increase this to exclude "equilibration" time.

                    ## Progress Bar
                    sys.stdout.write("Progress: {0:.2f}% Complete\r".format((float(ts.frame) / float(len(u.trajectory))) * 100))
                    sys.stdout.flush()

                    box = u.dimensions[:3]## define box and half box here so we save division calculation for every distance pair when we calculate 'dist' below.
                    hbox = u.dimensions[:3]/2.0

                    ## Compute all pairwise distances
                    lj2_dist2,lj2_dr = self.computePbcDist2(solute_sel.atoms[0].position, solute_sel.atoms[1].position, box, hbox)# Calculate the vector (lj2_dr) between the two LJ particles.
                    for a in solute_sel.atoms:
                        for b in solvent_sel.residues:
                            #rSolv = (b.atoms[1].position + d * ((b.atoms[1].position - b.atoms[0].position)/1.1)) # solvent vector to excluded volume center.
                            rSolv = b.atoms[1].position
                            ## Bin the solvent for g(r) only if the solvent is on the far side of the respective solute atom
                            solvRes_dist2,solvRes_dr = self.computePbcDist2(a.position, rSolv, box, hbox) # distance between solute and excluded volume center of CL3
                            if a.index == 0: # if first LJ particle is selected...
                                if solvRes_dist2 <= hist_dist_max2:
                                    if np.dot(lj2_dr,solvRes_dr) < 0: # Is dr dot product pointing right direction? Then compute sqrt
                                        if hist_dist_min2 < solvRes_dist2 < hist_dist_max2:
                                            self.computeGr(a, b, solvRes_dist2, solvRes_dr, a.index, box, hbox, Gr, Fr, Bz)

                            elif a.index == 1: # if second LJ particle is selected...
                                if solvRes_dist2 <= hist_dist_max2:
                                    if np.dot(lj2_dr,solvRes_dr) > 0: # Is dr dot product pointing right direction? Then compute sqrt
                                        if hist_dist_min2 < solvRes_dist2 < hist_dist_max2:
                                            self.computeGr(a, b, solvRes_dist2, solvRes_dr, a.index, box, hbox, Gr, Fr, Bz)


    def averageFrBz(self, Gr, Fr, Bz):
        ## Average LJ radial force for each distance and cos(theta) bin
        for a in range(nAtomTypes):
            for k in range(num_phi_bins):
                for j in range(num_theta_bins):
                    for i in range(num_dist_bins):
                        if Gr[a, 1, i, j, k] > 0.5:
                            Fr[a, 0, 0, i, j, k] /= Gr[a, 1, i, j, k]
                            Fr[a, 1, 0, i, j, k] /= Gr[a, 1, i, j, k]
                            Fr[a, 2, 0, i, j, k] /= Gr[a, 1, i, j, k]
                            Fr[a, 0, 1, i, j, k] /= Gr[a, 1, i, j, k]
                            Fr[a, 1, 1, i, j, k] /= Gr[a, 1, i, j, k]
                            Fr[a, 2, 1, i, j, k] /= Gr[a, 1, i, j, k]
                            Bz[a, 0, i, j, k] /= Gr[a, 1, i, j, k]
                            Bz[a, 1, i, j, k] /= Gr[a, 1, i, j, k]


    def calculateSD(self, Fr, Bz):
        ## Standard Deviation of every bin Force and Boltzmann using the variance.
        for a in range(nAtomTypes):
            for k in range(num_phi_bins):
                for j in range(num_theta_bins):
                    for i in range(num_dist_bins):
                        Fr[a, 0, 3, i, j, k] = Fr[a, 0, 1, i, j, k] - Fr[a, 0, 0, i, j, k]*Fr[a, 0, 0, i, j, k]
                        Bz[a, 3, i, j, k] = Bz[a, 1, i, j, k] - Bz[a, 0, i, j, k]*Bz[a, 0, i, j, k]
                        Fr[a, 0, 2, i, j, k] = sqrt( Fr[a, 0, 3, i, j, k] )
                        Bz[a, 2, i, j, k] = sqrt( Bz[a, 3, i, j, k] )

    def volumeCorrect(self, Gr):
        ## Volume Correct
        for a in range(nAtomTypes):
            for i in range(num_dist_bins):
                Gr[a, 0, i, :, :] /= 4*pi*((i+0.5)*bin_dist_size + hist_dist_min)**2

    def normalizeGr(self, Gr):
        ## Normalize
        ## have to normalize after volume correction because the 'bulk' g(r) value changes after volume correction.
        norm_points = 10
        for a in range(nAtomTypes):
            # normalize by the last 'norm_points' distance points
            g_norm = 0.
            for k in range(num_phi_bins):
                for j in range(num_theta_bins):
                    for i in range(norm_points):
                        g_norm += Gr[a, 0, -(i+1), j, k]

            g_norm /= float(norm_points*num_theta_bins*num_phi_bins)

            for k in range(num_phi_bins):
                for j in range(num_theta_bins):
                    for i in range(num_dist_bins):
                        Gr[a, 0, i, j, k] /= g_norm


    def integrateDirectSoluteSolventFrc(self, Fr, U_dir):
        ## Integrate the direct Solute--Solvent force
        for a in range(nAtomTypes):
            for k in range(num_phi_bins):
                for j in range(num_theta_bins):
                    for i in range(1,num_dist_bins+1):
                        if i == 1:
                            U_dir[a, -i, j, k] = Fr[a, 0, 0, -i, j, k] * bin_dist_size
                        else:
                            U_dir[a, -i, j, k] = U_dir[a, -(i-1), j, k] + Fr[a, 0, 0, -i, j, k] * bin_dist_size


    def writeOutputGr2D(self, out_file, Gr, Fr, Bz, U_dir):
        ## Open Output File
        out = open(out_file,'w')

        out.write("## 1: Distance Bin\n")
        out.write("## 2: Cos(theta) Bin\n")
        out.write("## 3: Phi/3 Bin\n")
        out.write("## 4: g(r) +\n")
        out.write("## 5: g(r) -\n")
        out.write("## 6: <force . r> +\n")
        out.write("## 7: <force . s> +\n")
        out.write("## 8: <force . t> +\n")
        out.write("## 9: g(r) Counts +\n")
        out.write("## 10: g(r) Counts -\n")
#        out.write("## 9: <force . r> Std Dev +\n")
#        out.write("## 10: <force . r> -\n")
#        out.write("## 11: <force . r> Std Dev -\n")
#        out.write("## 12: <boltzmann> +\n")
#        out.write("## 13: <boltzmann> Std Dev +\n")
#        out.write("## 14: <boltzmann> -\n")
#        out.write("## 15: <boltzmann> Std Dev -\n")
#        out.write("## 16: Integrated force +\n")
#        out.write("## 17: Integrated force -\n")
        for i in range(num_dist_bins):
            for j in range(num_theta_bins):
                for k in range(num_phi_bins):
                    out.write("%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n" %((i+0.5)*bin_dist_size+hist_dist_min, (j+0.5)*bin_theta_size+hist_theta_min, (k+0.5)*bin_phi_size+hist_phi_min, Gr[0,0,i,j,k], Gr[1,0,i,j,k], Fr[0,0,0,i,j,k], Fr[0,1,0,i,j,k], Fr[0,2,0,i,j,k], Gr[0,1,i,j,k], Gr[1,1,i,j,k]))#, Fr[0,0,2,i,j,k], Fr[1,0,0,i,j,k], Fr[1,0,2,i,j,k], Bz[0,0,i,j,k], Bz[0,2,i,j,k], Bz[1,0,i,j,k], Bz[1,2,i,j,k], U_dir[0,i,j,k], U_dir[1,i,j,k]))

        ## Close Output File
        out.close


    ## Initialize collapsed arrays
    def initializeArraysCollapsed(self):
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
    def collapseArrays(self, GrC, FrC, BzC, Gr, Fr, Bz):
        for a in range(nAtomTypes):
            for i in range(num_dist_bins):
                for j in range(num_theta_bins):
                    for k in range(num_phi_bins):
                        # sum of weighting factors for each distance
                        GrC[a, i] += Gr[a, 0, i, j, k]
                        # sum of weighted forces
                        FrC[a, 0, i] += Fr[a, 0, 0, i, j, k] * Gr[a, 0, i, j, k]
                        # sum of weighted (forces)**2. (The first term in sigma**2)
                        FrC[a, 2, i] += Fr[a, 0, 1, i, j, k] * Gr[a, 0, i, j, k]
                        # (sum of weighted forces)**2. (The second term in sigma**2)
                        FrC[a, 1, i] += Fr[a, 0, 0, i, j, k] * Gr[a, 0, i, j, k]

                        # sum of weighted boltzmann factors
                        BzC[a, 0, i] += Bz[a, 0, i, j, k] * Gr[a, 0, i, j, k]
                        # sum of weighted (boltzmanns**2). (The first term in sigma**2)
                        BzC[a, 2, i] += Bz[a, 1, i, j, k] * Gr[a, 0, i, j, k]
                        # (sum of boltmanns)**2 weighted. (The second term in sigma**2)
                        BzC[a, 1, i] += Bz[a, 0, i, j, k] * Gr[a, 0, i, j, k]
                ## XXX End of j loop

                if GrC[a, i] > 0.0000001:
                    # sum of weighted forces divided by sum of weights gives weighted average.
                    FrC[a, 0, i] /= GrC[a, i]
                    # Std Dev using variance with weighted averages.
                    print FrC[a, 2, i], GrC[a, i], FrC[a, 1, i], GrC[a, i]
                    print sqrt( ( FrC[a, 2, i] / GrC[a, i] ) - ( FrC[a, 1, i] / GrC[a, i] )**2 )
                    print "\n"
                    FrC[a, 1, i] = sqrt( ( FrC[a, 2, i] / GrC[a, i] ) - ( FrC[a, 1, i] / GrC[a, i] )**2 )

                    # sum of weighted forces divided by sum of weights gives weighted average.
                    BzC[a, 0, i] /= GrC[a, i]
                    # Std Dev using variance with weighted averages.
                    BzC[a, 1, i] = sqrt( ( BzC[a, 2, i] / GrC[a, i] ) - ( BzC[a, 1, i] / GrC[a, i] )**2 )

                    # So that bulk value is == 1.
                    GrC[a, i] /= num_theta_bins*num_phi_bins


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

        print 'Reading input and initializing'
        # read cfg file
        self.ParseConfigFile(cfg_file)

        # parse the prmtop file 
        self.ParsePrmtopBonded(top_file)

        ##########
        # initialize 2D arrays
        global nAtomTypes
        nAtomTypes = 2 # Number of solute atom types (+1z,-1z)

        # initialize with total dist, theta, and phi bins
        Gr,Fr,Bz,U_dir = self.initializeArrays2D()

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
        self.integrateDirectSoluteSolventFrc(Fr, U_dir)

        # write 2D output file: gr, frc, boltz, integrated_force
        print 'Write 2D output file'
        self.writeOutputGr2D(out_file, Gr, Fr, Bz, U_dir)

        ##########
#        print 'Starting Collapsed'
#        # initialize collapsed arrays
#        GrC,FrC,BzC = self.initializeArraysCollapsed()
#
#        # collapse the 2D arrays into the 1D collapsed arrays
#        print 'Collapsing arrays'
#        self.collapseArrays(GrC, FrC, BzC, Gr, Fr, Bz)
#
#        # write collapsed output file
#        print 'Writing collapsed output file'
#        self.writeOutputCollapsed(collapsed_file, GrC, FrC, BzC)
#
        print 'All Done!'




# Run Main program code.
g2d = gr2D()

g2d.mainLJ()
