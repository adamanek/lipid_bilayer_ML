#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 10:30:23 2020

@author: adam
"""
import MDAnalysis
import numpy as np
import tqdm
structure = 'test.tpr'
trajectory = ''
#u= MDAnalysis.Universe(structure, trajectory)

lipid_resnames = ['DAPE','DLPE','DOPE','DPPE', 'POPE', 'PIPE', 'DPPC', 'PIPC', 'PAPC', 'POPC', 'PAPS', 'POPS', 'PGPS', 'DBSM', 'DXSM', 'DPSM']

def find_sn(lipid_resnames, structure):
    u = MDAnalysis.Universe(structure)
    sn_dict = dict.fromkeys(lipid_resnames)
    for key in tqdm.tqdm(sn_dict.keys()):
        fir_lip = u.select_atoms(f'resname {key}')[0].resid
        m = u.select_atoms(f'resid {fir_lip}')
        f = m.bonds
        l = []
        for i in f:
            l.append(i.indices)
        #Finds the bead with the most number of bonds (this is the first bead of sn1 chain)
        l = np.concatenate(l)
        sn1_atom = np.bincount(l).argmax()
        sn2_atom = sn1_atom+1
        
        sn1_sel = ''
        sn2_sel = ''
              
        sn1_sel += str(m.select_atoms(f'index {sn1_atom}').names[0])+' '
        sn2_sel += str(m.select_atoms(f'index {sn2_atom}').names[0])+' '
        
        for i in m.names:
            if i[2] == 'A':
                sn1_sel += i + ' '
            elif i[2] == 'B':
                sn2_sel += i + ' '
        sn_dict[key] = sn1_sel, sn2_sel
    return sn_dict

Dic = find_sn(lipid_resnames,structure)