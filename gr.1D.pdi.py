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
                        #elif option.lower()=='trajfile':
                        #        traj_file = value
                        elif option.lower()=='outfile':
                                out_file = value
                        else :
                                print "Option:", option, " is not recognized"
        f.close()


# Get name of Coordinate files from config file
def getCoordFiles(cfg_file):
    global coordFiles
    coordFiles = []
    txt = open(cfg_file, 'r')
    while len(coordFiles) == 0:
        line = txt.next()
        if line == 'COORD FILES:\n':
            line = txt.next()
            while line != '\n':
                if line == 'END CONFIG\n':
                    print('NO COORD FILES FOUND IN CONFIG')
                coordFiles.append(line[:-1])
                line = txt.next()
            #if debug:
                #print('Coordinate files: {}'.format(coordFiles))


########################################################################
############################# Main Program #############################
########################################################################

cfg_file = sys.argv[1] ## read in command line argument (cfg file)

ParseConfigFile(cfg_file) ## read cfg file
getCoordFiles(cfg_file)

#print "Topology file: ", top_file
#print "Trajectory file: ", traj_file
#print "Output data file: ", out_file

## Histogram Array
# Distances [in Angstroms]
hist_min = 0
hist_max = 25
hist_min2= hist_min*hist_min
hist_max2= hist_max*hist_max
bin_size = 0.1
# Histogram bins
num_bins = int((hist_max - hist_min)/bin_size)
dist_hist = np.zeros(num_bins,dtype=float)## This will be the 'counts' array for the histogram.

for i in coordFiles:
    traj_file = i
    print 'Analyzing '+traj_file

    u = MDAnalysis.Universe(top_file, traj_file)## initiate MDAnalysis coordinate universe

    # create array of atomgroups with center ring of PDI core.
    res_sel = u.select_atoms('resname PDI')
    n_res = len(res_sel.residues) # number of solute residues
    solute_sel = np.empty(n_res,dtype=object)
    for i in range(n_res):
        mda_str = 'resname PDI and resid ' + str(i+1) + ' and (name C12 or name C13 or name C16 or name C17 or name C18 or name C19)'
        solute_sel[i] = u.select_atoms(mda_str)

    ## Loop through trajectory
    for ts in u.trajectory:## Loop over all time steps in the trajectory.
        if ts.frame >= 0:## ts.frame is the index of the timestep. Increase this to exclude "equilibration" time?
            box = u.dimensions[:3]## define box and half box here so we save division calculation for every distance pair when we calculate 'dist' below.
            hbox = u.dimensions[:3]/2.0
            
            ## Progress Bar
            sys.stdout.write("Progress: {0:.2f}% Complete\r".format((float(ts.frame) / float(len(u.trajectory))) * 100))
            sys.stdout.flush()

            ## Compute all pairwise distances 
            for a in range(n_res):
                for b in range(n_res):
                    if a != b:
                        dist2,dr = computePbcDist2(solute_sel[a].center_of_mass(),solute_sel[b].center_of_mass(),box,hbox)
                        if hist_min2 < dist2 < hist_max2:
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
