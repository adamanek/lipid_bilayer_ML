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
import math
import pandas as pd
from MDAnalysis.analysis.distances import distance_array

structure = os.path.sep.join(["test_inputs/step7_production.gro"])
tpr = os.path.sep.join(["test_inputs/step7_production.tpr"])
trajectory = os.path.sep.join(["test_inputs/step7_production.xtc"])

#lipid_resnames = ['DAPE','DLPE','DOPE','DPPE', 'POPE', 'PIPE', 'DPPC', 'PIPC', 'PAPC', 'POPC', 'PAPS', 'POPS', 'PGPS', 'DBSM', 'DXSM', 'DPSM']
lipid_resnames= ['DOPC']
def find_sn(lipid_resnames, tpr_file):
    
    """
    
    This function identifies the two sn chains in phospholipids and saves
    their atomnames for later use.
    
    """
    u = MDAnalysis.Universe(tpr_file)
    sn_dict = dict.fromkeys(lipid_resnames)
    for key in (sn_dict.keys()):
        fir_lip = u.select_atoms(f'resname {key}')[0].resid
        m = u.select_atoms(f'resid {fir_lip}')
        f = m.bonds
        l = []
        for i in f:
            l.append(i.indices)
        #Finds the bead with the most number of bonds (this is the first bead of sn1 chain)
        l = np.concatenate(l)
        sn1_atom = np.bincount(l).argmax()
        #T he first atom of the sn2 chain has one higher index number than 
        # the first atom of the sn1 chain by MARTINI convention.
        sn2_atom = sn1_atom+1
        
        sn1_sel = ''
        sn2_sel = ''
              
        sn1_sel += str(m.select_atoms(f'bynum {sn1_atom + 1}').names[0])+' '
        sn2_sel += str(m.select_atoms(f'bynum {sn2_atom + 1}').names[0])+' '
        
        # By MARTINI convention carbon atomnames end in 'A' and 'B' for sn1
        # and sn2 chains respectively. 
        for i in m.names:
            if i[2] == 'A':
                sn1_sel += i + ' '
            elif i[2] == 'B':
                sn2_sel += i + ' '
        sn_dict[key] = sn1_sel, sn2_sel
    return sn_dict



def calc_dist(p1, p2):
    x_dist = (p2[0] - p1[0])
    y_dist = (p2[1] - p1[1])
    return math.sqrt(x_dist * x_dist + y_dist * y_dist)
  
def find_thickness(lipid_resnames, lipids, sn_dic, thic):
    """
    
    Calculates the thickness of the bilayer by calculating the distances
    the highest and lowest atoms in both sn chains and returns their average.
    
    """
    thickness_dictionary = dict.fromkeys(lipid_resnames)
    for key in (sn_dic.keys()):
        
        
        for res in lipids.residues:
    
            sn1_atoms = res.atoms.select_atoms(f'name {sn_dic.get(key)[0]}')
            sn2_atoms = res.atoms.select_atoms(f'name {sn_dic.get(key)[1]}')
        
            sn1_thickness = np.max(sn1_atoms.positions[:,2]) -\
                np.min(sn1_atoms.positions[:,2])
            sn2_thickness = np.max(sn2_atoms.positions[:,2]) -\
                np.min(sn2_atoms.positions[:,2])
        
            thicknesses[res.resid].append(
                (sn1_thickness + sn2_thickness) / 2.0
            )
        thickness_dictionary[key] = thicknesses
    return thickness_dictionary

def find_angle(lipid_resnames, lipids, sn_dic, ang):
    """
    Calculates the angle between the last atom in the sn1 chain, the first atom
    in the sn1-chain and the last atom of the sn2 chain.
    
    Represents the angle between the two chains.
    
    """
    angles_dictionary = dict.fromkeys(lipid_resnames)
    for key in (sn_dic.keys()):
        for res in lipids.residues:
    
            sn1_first = res.atoms.select_atoms(f'name {sn_dic.get(key)[0].split()[0]}').center_of_geometry()
            sn1_last = res.atoms.select_atoms(f'name {sn_dic.get(key)[0].split()[-1]}').center_of_geometry()
            sn2_last = res.atoms.select_atoms(f'name {sn_dic.get(key)[1].split()[-1]}').center_of_geometry()
            
            vec_sn1 = sn1_first - sn1_last
            vec_sn2 = sn1_first - sn2_last
                
            angle = np.arccos(np.dot(vec_sn1, vec_sn2)/(norm(vec_sn1) * norm(vec_sn2)))
            
            angles[res.resid].append(np.rad2deg(angle))
            
        angles_dictionary[key] = angles
        
    return angles_dictionary

def find_dist(lipid_resnames, lipids, sn_dic, dist):
    
    """
    Finds the relative distance in X-Y space between the first and last atoms
    of both sn chains.
    
    """
    
    dist_dictionary = dict.fromkeys(lipid_resnames)
    
    for key in (sn_dic.keys()):
        
        for res in lipids.residues:
            
            sn1_atoms = res.atoms.select_atoms(f'name {sn_dic.get(key)[0]}')
            sn2_atoms = res.atoms.select_atoms(f'name {sn_dic.get(key)[1]}')
            
            sn1_dist = (calc_dist(sn1_atoms[0].position, sn1_atoms[-1].position) /
                        np.concatenate(distance_array(sn1_atoms[0].position, sn1_atoms[-1].position))[0])
            sn2_dist = (calc_dist(sn2_atoms[0].position, sn2_atoms[-1].position) /
                        np.concatenate(distance_array(sn2_atoms[0].position, sn2_atoms[-1].position))[0])
            
            # Need to decide whether to do average of both sn chains or save both
            distances[res.resid].append((sn1_dist + sn2_dist) / 2)
            #distances[res.resid].append((sn1_dist, sn2_dist))              
        dist_dictionary[key] = distances
        
    return dist_dictionary

def make_array_var(dictionary, array):
    Q = dictionary[lipid_type]
    dis_array = []
    
    for key in Q:
        dis_array.append(Q[key])
    dis_array = np.concatenate(dis_array)
    
    return array.append(dis_array)


Dic = find_sn(lipid_resnames,tpr)
u = MDAnalysis.Universe(structure, trajectory)
for lipid_type in lipid_resnames:
    
    lipids = u.select_atoms(f'resname {lipid_type}', updating=True)
    array_all_var = []
    thicknesses = {res.resid:[] for res in lipids.residues}
    angles = {res.resid:[] for res in lipids.residues}
    distances = {res.resid:[] for res in lipids.residues}        
    
    for tf in tqdm.tqdm(u.trajectory[-10:-1:1]):
        
        dis_thick = find_thickness(lipid_resnames,lipids,Dic, thicknesses)
        dis_ang = find_angle(lipid_resnames,lipids,Dic, angles)
        dis_dis = find_dist(lipid_resnames,lipids,Dic, distances)
    
    
    make_array_var(dis_thick,array_all_var)
    make_array_var(dis_ang,array_all_var)
    make_array_var(dis_dis,array_all_var)

    df = pd.DataFrame(np.transpose(array_all_var))
    df.to_csv(os.path.sep.join(["output", f"training_set_{lipid_type}.csv"]), index = False, header = False)
    

    
        

    
