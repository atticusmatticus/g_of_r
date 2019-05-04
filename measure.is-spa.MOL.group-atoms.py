# python3 measure.is-spa.MOL.group-atoms.py [cfg_file]

## Config File Format:
#
#   file_before_atom_number = atom-
#   file_after_atom_number_lj = _d0.00_lj_PDI.gr1
#   file_after_atom_number_coulomb = _d0.00_coulomb_PDI.gr1
#   grouped_file_suffix = _d0.00_grouped_PDI.gr1
#   
#   ## Atom lists:
#   # core
#   O = 39,40,41,42
#   N = 7,27
#   bay-C = 24,29,15,32
#   carbonyl-C = 26,28,12,8
#   4thRow-C = 22,25,11,9
#   bay-H = 43,45,47,48
#   3rdRow_in-C = 10,21
#   2ndRow_in-C = 13,18
#   out_bay-H = 44,46,49,50
#   1stRow-C = 14,17,20,19
#   3rdRow_out-C = 31,16,30,23
#   # r-group
#   o-C = 3,5,34,38
#   m-C = 2,6,35,37
#   1stBranch-C = 51,52,56,57
#   ipso-C = 33,4
#   p-C = 1,36
#   p-H = 55,60
#   2ndBranch-C = 61,62,63,64,79,80,81,82
#   m-H = 53,54,58,59
#   1stBranch-H = 65,72,83,90
#   2ndBranch-H = 66,67,68,69,70,71, 73,74,75,76,77,78, 84,85,86,87,88,89, 91,92,93,94,95,96
#

import numpy as np
import sys

def parse_cfg_file(cfg_file):
    names = [] # list of new file names after combining
    atomString = []
    f = open(cfg_file)
    for line in f:
        # first remove comments
        if '#' in line:
            line, comment = line.split('#',1)
        if '=' in line:
            option, value = line.split('=',1)
            option = option.strip()
            value = value.strip()
            # check value
            if option.lower() == 'file_before_atom_number':
                file_pre = str(value)
            elif option.lower() == 'file_after_atom_number_lj':
                file_post1 = str(value)
            elif option.lower() == 'file_after_atom_number_coulomb':
                file_post2 = str(value)
            elif option.lower() == 'grouped_file_suffix':
                out_file_suffix = str(value)
            else:
                print("Group name:", option, " Atoms:", value)
                names.append(option)
                atomString.append(value)
    i=0
    atoms = []
    for group in atomString:
        atoms.append([])    # add a dimension (list) to atoms for every different atom group
        atoms[i] = [x.strip() for x in group.split(',')]    # make the 'group' which is a CSV string into a list
        i+=1
    return file_pre, file_post1, file_post2, out_file_suffix, names, atoms;

def average_grouped_atoms(file_pre,file_post1,file_post2,names,atoms):
    nBins = len(np.loadtxt(file_pre + str(1).zfill(2) + file_post1)) # load the first atom file and get the number of bins
    grFr = np.zeros( (len(names),2,nBins), dtype=float)
    gSum = np.zeros( nBins, dtype=int)
    i=0 # 'i' will be used to represent which new 'name' group we are in.
    for name in names:  # loop through new 'name' groups
        gSum[:] = 0
        for atomIndex in atoms[i]:
            inFile1 = file_pre + str(atomIndex).zfill(2) + file_post1
            r1,g1,f1 = np.loadtxt(inFile1, unpack=True)
            inFile2 = file_pre + str(atomIndex).zfill(2) + file_post2
            r2,g2,f2 = np.loadtxt(inFile2, unpack=True)
            for j in range(nBins):  # 'j' is the r-bin.
                grFr[i,0,j] += g1[j]*f1[j]  # Weighted average. Force weighted by the # of times it was observed. LJ
                grFr[i,1,j] += g2[j]*f2[j]  # Weighted average. Force weighted by the # of times it was observed. Coulomb
                gSum[j] += g1[j]   # Denominator of weighted average.
        # Divide by sum of weights to complete the average for a group.
        for j in range(nBins):
            if gSum[j] > 0.5:
                grFr[i,0,j] /= float(gSum[j])
                grFr[i,1,j] /= float(gSum[j])
        i+=1    # next new 'name' group
    return r1, grFr;

def write_output(r,grFr,names,file_post1,file_post2,out_file_suffix):
    i=0
    for name in names:
        out = open(name + out_file_suffix, 'w')
        out.write('## 1.  r\n')
        out.write('## 2.  LJ Force\n')
        out.write('## 3.  Coulomb Force\n')
        for j in range(len(grFr[0,0])):
            out.write('%10.5f %10.5f %10.5f\n' %(r[j], grFr[i,0,j], grFr[i,1,j]))
        out.close
        i+=1

def main():
    # Get cfg_file name from command line
    cfg_file = sys.argv[1]

    # Retrieve atom lists from config file
    file_pre,file_post1,file_post2,out_file_suffix,names,atoms = parse_cfg_file(cfg_file)

    # Average the grouped atoms together
    r,grFr = average_grouped_atoms(file_pre,file_post1,file_post2,names,atoms)

    # Write output file
    write_output(r,grFr,names,file_post1,file_post2,out_file_suffix)

main()
