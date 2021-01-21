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
from MDAnalysis.analysis.leaflet import LeafletFinder

structure = os.path.sep.join(["/media/adam/My Passport/Data Backup/Data/simulations/ML/Training datsets/DOPC_CHOL/step7_production.gro"])
tpr = os.path.sep.join(["/media/adam/My Passport/Data Backup/Data/simulations/ML/Training datsets/DOPC_CHOL/step7_production.tpr"])
trajectory = os.path.sep.join(["/media/adam/My Passport/Data Backup/Data/simulations/ML/Training datsets/DOPC_CHOL/step7_production.xtc"])



#lipid_resnames = ['DAPE','DLPE','DOPE','DPPE', 'POPE', 'PIPE', 'DPPC', 'PIPC', 'PAPC', 'POPC', 'PAPS', 'POPS', 'PGPS', 'DBSM', 'DXSM', 'DPSM']
lipid_resnames= ['DOPC','DPPC']
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
  
def find_thickness(lipid_resnames, lipids, sn_dic, thicknesses):
    """
    
    Calculates the thickness of the bilayer by calculating the distances
    the highest and lowest atoms in both sn chains and returns their average.
    
    """
        
    for res in lipids.residues:

        sn1_atoms = res.atoms.select_atoms(f'name {sn_dic.get(res.resname)[0]}')
        sn2_atoms = res.atoms.select_atoms(f'name {sn_dic.get(res.resname)[1]}')
        sn1_len = []
        sn2_len = []
        for i in range(len(sn1_atoms.positions)-1):
            dis = distance_array(sn1_atoms.positions[i], sn1_atoms.positions[i+1])
            sn1_len.append(dis)
        for i in range(len(sn2_atoms.positions)-1):
            dis = distance_array(sn2_atoms.positions[i], sn2_atoms.positions[i+1])
            sn2_len.append(dis)
    
        sn1_thickness = (np.max(sn1_atoms.positions[:,2]) -\
            np.min(sn1_atoms.positions[:,2])) / np.concatenate(sum(sn1_len))[0]
        sn2_thickness = (np.max(sn2_atoms.positions[:,2]) -\
            np.min(sn2_atoms.positions[:,2])) / np.concatenate(sum(sn2_len))[0]
    
        thicknesses[res.resid].append(
            (sn1_thickness + sn2_thickness) / 2.0
        )
    return thicknesses

def find_angle(lipid_resnames, lipids, sn_dic, angles):
    """
    Calculates the angle between the last atom in the sn1 chain, the first atom
    in the sn1-chain and the last atom of the sn2 chain.
    
    Represents the angle between the two chains.
    
    """
    

    for res in lipids.residues:

        sn1_first = res.atoms.select_atoms(f'name {sn_dic.get(res.resname)[0].split()[0]}').center_of_geometry()
        sn1_last = res.atoms.select_atoms(f'name {sn_dic.get(res.resname)[0].split()[-1]}').center_of_geometry()
        sn2_last = res.atoms.select_atoms(f'name {sn_dic.get(res.resname)[1].split()[-1]}').center_of_geometry()
        
        vec_sn1 = sn1_first - sn1_last
        vec_sn2 = sn1_first - sn2_last
            
        angle = np.arccos(np.dot(vec_sn1, vec_sn2)/(norm(vec_sn1) * norm(vec_sn2)))
        
        angles[res.resid].append(np.rad2deg(angle))
            
        
    return angles

def find_dist(lipid_resnames, lipids, sn_dic, dist):
    
    """
    Finds the relative distance in X-Y space between the first and last atoms
    of both sn chains.
    
    """
    

    for res in lipids.residues:
            
        sn1_atoms = res.atoms.select_atoms(f'name {sn_dic.get(res.resname)[0]}')
        sn2_atoms = res.atoms.select_atoms(f'name {sn_dic.get(res.resname)[1]}')
        sn1_len = []
        sn2_len = []
        for i in range(len(sn1_atoms.positions)-1):
            dis = distance_array(sn1_atoms.positions[i], sn1_atoms.positions[i+1])
            sn1_len.append(dis)
        for i in range(len(sn2_atoms.positions)-1):
            dis = distance_array(sn2_atoms.positions[i], sn2_atoms.positions[i+1])
            sn2_len.append(dis)

        
        sn1_dist = calc_dist(sn1_atoms[0].position, sn1_atoms[-1].position) / np.concatenate(sum(sn1_len))[0]
        sn2_dist = calc_dist(sn2_atoms[0].position, sn2_atoms[-1].position) / np.concatenate(sum(sn2_len))[0]
        
        # Need to decide whether to do average of both sn chains or save both
        distances[res.resid].append((sn1_dist + sn2_dist) / 2)
        #distances[res.resid].append((sn1_dist, sn2_dist))              
        
    return distances

def make_array_var(dictionary, array):
    dis_array = []
    
    for key in dictionary:
        dis_array.append(dictionary[key])
    dis_array = np.concatenate(dis_array)
    
    return array.append(dis_array)


Dic = find_sn(lipid_resnames,tpr)
u = MDAnalysis.Universe(structure, trajectory)
for lipid_type in lipid_resnames:
    
    lipids = u.select_atoms(f'resname {lipid_type}', updating=True)
    thicknesses = {res.resid:[] for res in lipids.residues}
    angles = {res.resid:[] for res in lipids.residues}
    distances = {res.resid:[] for res in lipids.residues}        
    alla = []
    for tf in tqdm.tqdm(u.trajectory[-200:-1:1]):
        
        dis_thick = find_thickness(lipid_resnames,lipids,Dic, thicknesses)
        dis_ang = find_angle(lipid_resnames,lipids,Dic, angles)
        dis_dis = find_dist(lipid_resnames,lipids,Dic, distances)
    
    
    #This commented out section is for getting all the values in 2D Matrix
    
#    make_array_var(dis_thick,alla)
#    make_array_var(dis_ang,alla)
#    make_array_var(dis_dis,alla)
#    alla.append(np.full([len(alla[0])], f'{lipid_type}'))
#    df = pd.DataFrame(np.transpose(alla))
    
    #Getting all the values in a 3D matrix
    
    matrix3D = [list(dis_thick.values()),
                 list(dis_ang.values()),
                 list(dis_dis.values()),
                 ]
    label = ['DOPC CHOL'] * len(dis_ang.values())
    np.save(f'output/train_set_{lipid_type}_CHOL_3D.npy',matrix3D)
    np.save(f'output/train_set_{lipid_type}_CHOL_3D_label.npy', label)
    #following code comment is for mean of each value for each lipid
    all_data = np.transpose(np.vstack((np.mean(list(dis_thick.values()), axis = 1),
                          np.mean(list(dis_ang.values()), axis = 1),
                          np.mean(list(dis_dis.values()), axis = 1),
                          np.full([len(dis_thick)], 'DOPC CHOL')
                          )))
    df = pd.DataFrame(all_data)   
    df.to_csv(os.path.sep.join(["output", f"train_set_{lipid_type}_CHOL_mean.csv"]), index = False, header = False)
    
structure = os.path.sep.join(["/media/adam/Black 4TB1/CG dian_laurdan/laurdan/md_laurdan_coarse_part5.gro"])
tpr = os.path.sep.join(["/media/adam/Black 4TB1/CG dian_laurdan/laurdan/md_laurdan_coarse_part5.tpr"])
trajectory = os.path.sep.join(["/media/adam/Black 4TB1/CG dian_laurdan/laurdan/last5ns_part5.xtc"])

Dic = find_sn(lipid_resnames,tpr)
u = MDAnalysis.Universe(structure, trajectory)
L = LeafletFinder(u, 'type P', cutoff=10, pbc=True, sparse = False)
L0 = L.group(0)
L1 = L.group(1)
Leaflets = [L0,L1]
i=0
L0 = u.select_atoms('type P and (prop z >40)')
L1 = u.select_atoms('type P and (prop z <40)')
Leaflets = [L0,L1]

for group in Leaflets:
    group_string = ''
    for bead in group:
        group_string += f'{bead.resid}' + ' '
    print('got here0')
    append_data = []
    lip_string = ''
    for lipid_type in lipid_resnames:
        lip_string += lipid_type + ' '
    lipids = u.select_atoms(f'resname {lip_string} and byres resid {group_string}', updating=True)
    print('got here')
    array_all_var = []
    thicknesses = {res.resid:[] for res in lipids.residues}   
    angles = {res.resid:[] for res in lipids.residues}
    distances = {res.resid:[] for res in lipids.residues} 
    
    #If i want to do it over a trajectory uncomment the next line and indent everything between that loop and the next comment
    print('got here2')

    for tf in tqdm.tqdm(u.trajectory[-200:-1:1]):
        dis_thick = find_thickness(lipid_resnames,lipids,Dic, thicknesses)
        dis_ang = find_angle(lipid_resnames,lipids,Dic, angles)
        dis_dis = find_dist(lipid_resnames,lipids,Dic, distances)
    #stop indent here
    all_data = np.transpose(np.vstack((np.mean(list(dis_thick.values()), axis = 1),
                          np.mean(list(dis_ang.values()), axis = 1),
                          np.mean(list(dis_dis.values()), axis = 1),
                          )))
#    df = pd.DataFrame(all_data)
    lip_resnames = group.resnames
    lip_resids = group.resids
#    df = pd.concat([df, pd.DataFrame(lip_resnames),pd.DataFrame(lip_resids)], axis = 1)
#    df.to_csv(os.path.sep.join(["output", f"CG_dian_leaflet{i}.csv"]), index = False, header = False)
    
    matrix3D = [list(dis_thick.values()),
             list(dis_ang.values()),
             list(dis_dis.values()),
             ]
    labels = [list(lip_resnames),
             list(lip_resids)]
    np.save(f'output/large_real_set_dian_DOPC_DPPC_3D_leaflet{i}.npy',matrix3D)
    np.save(f'output/large_real_set_dian_DOPC_DPPC_3D_leaflet{i}_labels.npy',labels)


    
    positions = group.positions[:,0:2]
    pf = pd.concat([pd.DataFrame(positions), pd.DataFrame(lip_resnames),pd.DataFrame(lip_resids)], axis = 1)    
    pf.to_csv(os.path.sep.join(["output", f"large_real_dian_DOPC_DPPC_3D_positions_leaflet{i}.csv"]), index = False, header = False)    
    i +=1