# Compute Radial Distribution Function and Seperately Compute the Average Polarization as a Function of Distance [ Collapsed into 1D]
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

## Read configuration file and populate global variables

def ParseConfigFile(cfg_file):
        global top_file, traj_file, out_file
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
                        else :
                                print "Option:", option, " is not recognized"
        f.close()

########################################################################
############################# Main Program #############################
########################################################################

cfg_file = sys.argv[1] ## read in command line argument (cfg file)

ParseConfigFile(cfg_file) ## read cfg file

#print "Topology file: ", top_file
#print "Trajectory file: ", traj_file
#print "Output data file: ", out_file

u = MDAnalysis.Universe(top_file, traj_file)## initiate MDAnalysis coordinate universe

# create array of atomgroups with center ring of PDI core.
res_sel = u.select_atoms('resname PDI')
n_res = len(res_sel.residues) # number of solute residues
solute_sel = np.empty(n_res,dtype=object)
for i in range(n_res):
    mda_str = 'resname PDI and resid ' + str(i+1) + ' and (name C12 or name C13 or name C16 or name C17 or name C18 or name C19)'
    solute_sel[i] = u.select_atoms(mda_str)

#print solute_sel[1]

#solute_sel = u.select_atoms('resname PDI and (name C12 or name C13 or name C16 or name C17 or name C18 or name C19)')
#!solvent_sel = u.select_atoms('resname CL3 and name C1')
#!solvent_sel = u.select_atoms('resname CL3')

## Histogram Array
# Distances [in Angstroms]
hist_min = 0
hist_max = 50
hist_min2= hist_min*hist_min
hist_max2= hist_max*hist_max
bin_size = 0.1
# Histogram bins
num_bins = int((hist_max - hist_min)/bin_size)
dist_hist = np.zeros(num_bins,dtype=float)## This will be the 'counts' array for the histogram.
## There is no need for bin_centers because int(5.9)=5 so bin distances are being assigned to the appropriate mid point
#print u.dimensions[:3]
## Loop through trajectory
for ts in u.trajectory:## Loop over all time steps in the trajectory.
    if ts.frame >= 0:## ts.frame is the index of the timestep. Increase this to exclude "equilibration" time?
        box = u.dimensions[:3]## define box and half box here so we save division calculation for every distance pair when we calculate 'dist' below.
        hbox = u.dimensions[:3]/2.0
#!        dist2,R12 = computePbcDist2(solute_sel.atoms[0].position,solute_sel.atoms[1].position,box,hbox)# Calculate the vector (R12) between the two LJ particles.
	
	## Progress Bar
        sys.stdout.write("Progress: {0:.2f}% Complete\r".format((float(ts.frame) / float(len(u.trajectory))) * 100))
        sys.stdout.flush()

#!        for i in range(n_res):
#!            print '--------Option 1--------'
#!            print solute_sel[i].center_of_mass()
#!            #print i.center_of_mass()
#!            print '------------------------'

        ## Compute all pairwise distances 
        for a in range(n_res):
            for b in range(n_res):
                if a != b:
                    dist2,dr = computePbcDist2(solute_sel[a].center_of_mass(),solute_sel[b].center_of_mass(),box,hbox)
                    dist = math.sqrt(dist2)
                    #
                    dist_bin = int((dist - hist_min)/bin_size)
                    dist_hist[dist_bin] += 1


## Volume Correct
for i in range(num_bins):
    dist_hist[i] /= 4*math.pi*((i+0.5)*bin_size + hist_min)**2

## Normalize
for i in range(num_bins): ## have to normalize after volume correction because the 'bulk' g(r) value changes after volume correction.
    dist_hist[i] /= dist_hist[num_bins - 1]

## Open Output File
out = open(out_file,'w')

out.write("#  dist       g_of_r\n")
for i in range(num_bins):
    out.write("%10.5f %10.5f\n" %(i*bin_size+hist_min,dist_hist[i]))

## Close Output File
out.close
