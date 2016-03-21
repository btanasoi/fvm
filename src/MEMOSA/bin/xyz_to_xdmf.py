#!/usr/bin/env python

"""
This program converts XYZ molecule files to Xdmf files.

Usage: xyz_to_xdmf [options] filename.xyz

'filename.xyz' is the input file.  The output will have the same
name with the extension 'xmf'. 

Options:
  -v      Verbose.
 """

import os, sys
try:
    import h5py
except:
    print "ERROR: Could not find h5py module in path."
    sys.exit(1)
from numpy import *
from optparse import OptionParser
import mpm.xdmf as xdmf
import memosa.elements as elements

def element_number(symbol):
    try:
        return elements.sym2elt[symbol].ano
    except:
        return 0
    
def read_atoms(fpin, xmf, atom_count, step):
    alist = []
    tlist = []
    a = array([], float32).reshape(0, 3)
    t = array([], int8)
    group = hdf.create_group(str(step))       
    while atom_count > 0:
        atom, x, y, z = fpin.readline().split()
        atom_count -= 1
        if atom_count % 100000 == 0:
            print atom_count
            a = concatenate((a, array(alist, float32)))
            alist = []
            t = concatenate((t, array(tlist, int8)))
            tlist = []
        alist.append([float(x), float(y), float(z)])
        tlist.append(element_number(atom))
    a = concatenate((a, array(alist, float32)))
    t = concatenate((t, array(tlist, int8)))
    # write to hdf5  and xdmf files
    group.create_dataset('data', data=a, compression='gzip')
    xmf.pv_geo('data', a)
    # now atom type
    group.create_dataset('element', data=t, compression='gzip')
    xmf.attr('element', t)
    return alen


if __name__ == '__main__':
    parser = OptionParser(__doc__)
    parser.add_option("-v", action="store_true", help="Verbose")
    (options, args) = parser.parse_args()
    if len(args) < 1:
        parser.error('No filename given.')

    step = 0    
    fpin = open(args[0])
    basename = args[0].split('.')[0]
    hdf = h5py.File(basename + '.h5', 'w')
    xmf = xdmf.Xdmf(basename + '.xmf', basename + '.h5', options.v)
    xmf.temporal_grid('Atomic Data')
                    
    while True:
        try:
            atom_count = int(fpin.readline())
        except ValueError:
            break
        comment = fpin.readline()
        xmf.grid(step)
        xmf.time(step)
        read_atoms(fpin, xmf, atom_count, step)
        step += 1
        
        
