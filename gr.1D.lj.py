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

solute_sel = u.select_atoms('resname LJ2 and name LJ')
solvent_sel = u.select_atoms('resname CL3 and name C1')
solvent_sel = u.select_atoms('resname CL3')

## Histogram Array
# Distances [in Angstroms]
hist_min = 0
hist_max = 25
hist_min2= hist_min*hist_min
hist_max2= hist_max*hist_max
bin_size = 0.1
# Histogram bins
num_bins = int((hist_max - hist_min)/bin_size)
dist_hist_0 = np.zeros(num_bins,dtype=float)## This will be the 'counts' array for the histogram.
dist_hist_1 = np.zeros(num_bins,dtype=float)## This will be the 'counts' array for the histogram.
pol_hist_0 = np.zeros(num_bins,dtype=float)## This will be the 'counts' array for the histogram.
pol_hist_1 = np.zeros(num_bins,dtype=float)## This will be the 'counts' array for the histogram.
## There is no need for bin_centers because int(5.9)=5 so bin distances are being assigned to the appropriate mid point
#print u.dimensions[:3]
## Loop through trajectory
for ts in u.trajectory:## Loop over all time steps in the trajectory.
    if ts.frame >= 0:## ts.frame is the index of the timestep. Increase this to exclude "equilibration" time?
        box = u.dimensions[:3]## define box and half box here so we save division calculation for every distance pair when we calculate 'dist' below.
        hbox = u.dimensions[:3]/2.0
        dist2,R12 = computePbcDist2(solute_sel.atoms[0].position,solute_sel.atoms[1].position,box,hbox)# Calculate the vector (R12) between the two LJ particles.
	
	## Progress Bar
        sys.stdout.write("Progress: {0:.2f}% Complete\r".format((float(ts.frame) / float(len(u.trajectory))) * 100))
        sys.stdout.flush()

        ## Compute all pairwise distances
        for a in solute_sel.atoms:
            for b in solvent_sel.residues:
                #print b.atoms[1]
                #print b.residue.H1
                dist2,dr = computePbcDist2(a.position,b.atoms[1].position,box,hbox)

                ## Bin the solvent for g(r) only if the solvent is on the far side of the respective solute atom
                if a.index == 0: # if first LJ particle is selected...
                    #print np.dot(R12,dr)
                    if np.dot(R12,dr) < 0: # Is dr dot product pointing right direction? Then compute sqrt
                        if hist_min2 < dist2 < hist_max2:
                            dist = math.sqrt(dist2)
                            ## Calculate Polarization Magnitude Along Radial Vector From Solute
                            pNow = b.atoms[1].position - b.atoms[0].position # Direction of solvent molecule 'b' dipole
                            pRad = np.dot(pNow,dr) / dist # Projection of solvent dipole onto radial vector. (So just a magnitude)
                            #
                            dist_bin = int((dist - hist_min)/bin_size)
                            dist_hist_0[dist_bin] += 1
                            pol_hist_0[dist_bin] += pRad
                elif a.index == 1: # if second LJ particle is selected...
                    if np.dot(R12,dr) > 0: # Is dr dot product pointing right direction? Then compute sqrt
                        if hist_min2 < dist2 < hist_max2:
                            dist = math.sqrt(dist2)
                            ## Calculate Polarization Magnitude Along Radial Vector From Solute
                            pNow = b.atoms[1].position - b.atoms[0].position # Direction of solvent molecule 'b' dipole
                            pRad = np.dot(pNow,dr) / dist # Projection of solvent dipole onto radial vector. (So just a magnitude)
                            #
                            dist_bin = int((dist - hist_min)/bin_size)
                            dist_hist_1[dist_bin] += 1
                            pol_hist_1[dist_bin] += pRad

                #dist_bin = int((dist - hist_min)/bin_size)
                #if hist_min < dist < hist_max:
                #    dist_hist[dist_bin] += 1
                #else:
                #    print 'Distance value out of bounds: %d' %dist

for i in range(num_bins): ## Divide polarization in each bin by the counts in that bin. Effectively averaging the polarization of that bin.
    if dist_hist_0[i] > 0.5: # 0.5 instead of 0.0 because if a bit flips and 0.0 turns into 0.0...01 the normalization factor will be huge.
        pol_hist_0[i] /= dist_hist_0[i]
    if dist_hist_1[i] > 0.5:
        pol_hist_1[i] /= dist_hist_1[i]


## Volume Correct
for i in range(num_bins):
    dist_hist_0[i] /= 4*math.pi*((i+0.5)*bin_size + hist_min)**2
    dist_hist_1[i] /= 4*math.pi*((i+0.5)*bin_size + hist_min)**2

## Normalize
for i in range(num_bins): ## have to normalize after volume correction because the 'bulk' g(r) value changes after volume correction.
    dist_hist_0[i] /= dist_hist_0[num_bins - 1]
    dist_hist_1[i] /= dist_hist_1[num_bins - 1]

#print dist_hist[num_bins-1]

## Convert Histogram into Probability Density
#dist_hist /= (200*sel.n_atoms*bin_size)

## Open Output File
out = open(out_file,'w')

out.write("#  distBin    distPos    distNeg    polPos     polNeg\n")
for i in range(num_bins):
    out.write("%10.5f %10.5f %10.5f %10.5f %10.5f\n" %(i*bin_size+hist_min,dist_hist_0[i],dist_hist_1[i],pol_hist_0[i],pol_hist_1[i]))

## Close Output File
out.close
