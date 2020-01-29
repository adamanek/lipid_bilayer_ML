#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 10:30:23 2020

@author: adam
"""
import MDAnalysis
import numpy as np
from numpy.linalg import norm
import tqdm
import os
structure = os.path.sep.join(["test_inputs/coarse_step8_production_3.gro"])
tpr = os.path.sep.join(["test_inputs/coarse_step8_production_3.tpr"])
trajectory = ''

#lipid_resnames = ['DAPE','DLPE','DOPE','DPPE', 'POPE', 'PIPE', 'DPPC', 'PIPC', 'PAPC', 'POPC', 'PAPS', 'POPS', 'PGPS', 'DBSM', 'DXSM', 'DPSM']
lipid_resnames= ['DOPC', 'DPPC']
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

Dic = find_sn(lipid_resnames,tpr)

u = MDAnalysis.Universe(structure)

def find_thickness(lipid_resnames, tpr_file):
    thickness_dictionary = dict.fromkeys(lipid_resnames)
    for key in tqdm.tqdm(Dic.keys()):
        lipids = u.select_atoms(f'resname {key}', updating=True)
        thicknesses = {res.resid:[] for res in lipids.residues}
        for res in lipids.residues:
    
            sn1_atoms = res.atoms.select_atoms(f'name {Dic.get(key)[0]}')
            sn2_atoms = res.atoms.select_atoms(f'name {Dic.get(key)[1]}')
        
            sn1_thickness = np.max(sn1_atoms.positions[:,2]) -\
                np.min(sn1_atoms.positions[:,2])
            sn2_thickness = np.max(sn2_atoms.positions[:,2]) -\
                np.min(sn2_atoms.positions[:,2])
        
            thicknesses[res.resid].append(
                (sn1_thickness + sn2_thickness) / 2.0
            )
        thickness_dictionary[key] = thicknesses
    return thickness_dictionary

def find_angle(lipid_resnames, tpr_file):
    angles_dictionary = dict.fromkeys(lipid_resnames)
    for key in tqdm.tqdm(Dic.keys()):
        lipids = u.select_atoms(f'resname {key}', updating=True)
        angles = {res.resid:[] for res in lipids.residues}
        for res in lipids.residues:
    
            sn1_first = res.atoms.select_atoms(f'name {Dic.get(key)[0].split()[0]}').center_of_geometry()
            sn1_last = res.atoms.select_atoms(f'name {Dic.get(key)[0].split()[-1]}').center_of_geometry()
            sn2_last = res.atoms.select_atoms(f'name {Dic.get(key)[1].split()[-1]}').center_of_geometry()
            
            vec_sn1 = sn1_first - sn1_last
            vec_sn2 = sn1_first - sn2_last
                
            angle = np.arccos(np.dot(vec_sn1, vec_sn2)/(norm(vec_sn1) * norm(vec_sn2)))
            
            angles[res.resid].append(np.rad2deg(angle))
            
        angles_dictionary[key] = angles
        
    return angles_dictionary