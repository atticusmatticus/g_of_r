import numpy as np
import sys
import math

########################################################################
############################# Sub Routines #############################
########################################################################

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


########################################################################
############################# Main Program #############################
########################################################################

cfg_file = sys.argv[1] ## read in command line argument (cfg file)

ParseConfigFile(cfg_file) ## read cfg file

## Histogram Array
# Distances [in Angstroms]
hist_dist_min2= hist_dist_min*hist_dist_min
hist_dist_max2= hist_dist_max*hist_dist_max
# Histogram bins
num_dist_bins = int((hist_dist_max - hist_dist_min)/bin_dist_size)

# Angles [cosine(theta)]
# Cosine Theta Histogram bins
num_ang_bins = int((hist_ang_max - hist_ang_min)/bin_ang_size)
# Boltzmann Constant in kcal/mol.K
k_B = 0.0019872041
global kT
kT = k_B * T

infile = np.loadtxt(out_file)
col_file = np.loadtxt(collapsed_file)

lj_force_0 = np.zeros((num_dist_bins,num_ang_bins),dtype=float)
for i in range(num_dist_bins):
    for j in range(num_ang_bins):
        lj_force_0[i][j] = infile[i*num_ang_bins+j][4]


u_dir_0 = np.zeros((num_dist_bins, num_ang_bins),dtype=float)
for j in range(num_ang_bins):
    for i in range(1,num_dist_bins+1):
        if i == 1:
            u_dir_0[-i][j] = lj_force_0[-i][j] * bin_dist_size
        else:
            u_dir_0[-i][j] = u_dir_0[-(i-1)][j] + lj_force_0[-i][j] * bin_dist_size



col_force_0 = np.zeros(num_dist_bins,dtype=float)
for i in range(num_dist_bins):
    col_force_0[i] = col_file[i][3]

u_dir_collapsed = np.zeros(num_dist_bins,dtype=float)
for i in range(1,num_dist_bins+1):
    if i == 1:
        u_dir_collapsed[-i] = col_force_0[-i] * bin_dist_size
    else:
        u_dir_collapsed[-i] = u_dir_collapsed[-(i-1)] + col_force_0[-i] * bin_dist_size

## Open Output File
out = open("integrated_force.log",'w')

out.write("## 1: Distance Bin\n")
out.write("## 2: Cos(theta) Bin\n")
out.write("## 3: Integrated force +\n")
for i in range(num_dist_bins):
    for j in range(num_ang_bins):
        out.write("%10.5f %10.5f %10.5f\n" %((i+0.5)*bin_dist_size+hist_dist_min, (j+0.5)*bin_ang_size+hist_ang_min, u_dir_0[i][j]))
out.close()

## Open Output File
out = open("integrated_force.collapsed.log",'w')

out.write("## 1: Distance Bin\n")
out.write("## 2: Integrated force +\n")
for i in range(num_dist_bins):
    out.write("%10.5f %10.5f\n" %((i+0.5)*bin_dist_size+hist_dist_min, u_dir_collapsed[i]))
out.close()
