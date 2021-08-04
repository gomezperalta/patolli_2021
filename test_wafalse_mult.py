# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:03:53 2021

@author: iG
"""

import pandas as pd
import numpy as np
import keras.models as models
from patolli import inout_creator
from patolli import compute_quotients
from patolli import append_local_functions, add_rad_elec, append_density, compute_diffelec
import os

def test_all_false(directorio = str(), database = './support/red_cod-db.pkl', 
                   local_function = './support/fij_2.0_25_diccio',
                   use_rad_elec=True, use_elecdiff=False,use_density=True):
    """
    This function tests all models once the training finished with the remaining
    false samples of red_cod-db.pkl.
    Parameters:
        directorio: A string with the name of the directory where the models are.
        database: a pickle file containing the entire database
        local_function: a numpy dictionary with the local functions to append.
    Returns:
        a txt - file with the name test_with_all_false, which is inside the given directory.
    """
    df = pd.read_pickle(database)
    collection = pd.read_csv(directorio + '/compounds_collection.csv')
    
    cifs = [i for i in collection['cif']]
    maxsites = np.max(collection['sitios'])
    
    df = df[df['sitios'] > 0][df['sitios'] <= maxsites].reset_index(drop=True)
    df = df.loc[~df['cif'].isin(cifs)].reset_index(drop=True)
    
    x, _, df = inout_creator(df=df)
    
    x = compute_quotients(X=x)
    x, df = append_local_functions(X = x,df=df)

    if use_rad_elec:
        xraw, _, _ = inout_creator(df=df)
        xre = add_rad_elec(X=xraw)
        
        if use_elecdiff:
            xde = compute_diffelec(X=xraw)
            xre = np.concatenate((xre, xde), axis=2)
            
        x = np.concatenate((xre, x), axis=2)
    
    if use_density:
        xdensity = append_density(df=df)
        x = np.concatenate((x, xdensity), axis=2)
    
    busqueda = "ls " + directorio + "/*.h5 > model_names.txt"
    os.system(busqueda)
    
    diccio = np.load(directorio + '/feature_standarisation.npy').item()
    
    X = (x - diccio['mean'])/diccio['std']
    x = np.reshape(X,(X.shape[0],X.shape[2]))
    
    with open('model_names.txt','r') as f:
        for line in f:
            modelo = models.load_model(line[:len(line)-1])
            nombre = line.split('/')[1]
            
            outpred = modelo.predict(x)
            prediction = np.argmax(outpred, axis=1)
            df['y_pred'] = np.ravel(prediction)
            
            with open(directorio+'/test_with_all_false.txt','a') as tr:
                
                tr.write(nombre + '\n')
                for sitio in range(1, max(df['sitios']) + 1):
                    subset = df[df['sitios'] == sitio].shape[0]
                    tr.write('With '+ str(sitio) + ' sites (' + str(subset) + '): \n')
                    for item in df['y_pred'].unique():
                        count = df[df['sitios'] == sitio][df['y_pred'] == item].shape[0]
                        recall = np.round(100*count/subset,2)
                        tr.write('Output '+ str(item) + ':' + str(recall) + '% (' + str(count) + ') \n')
                tr.write('\n')
                    
                tr.close()
    return

def test_ann_all_false(directorio = str(), ann= 'ANN001',
                   database = './support/red_cod-db.pkl', 
                   local_function = './support/fij_2.0_25_diccio',
                   use_rad_elec=True, use_elecdiff=False,use_density=True):
    """
    This function tests all models once the training finished with the remaining
    false samples of red_cod-db.pkl.
    Parameters:
        directorio: A string with the name of the directory where the models are.
        database: a pickle file containing the entire database
        local_function: a numpy dictionary with the local functions to append.
    Returns:
        a txt - file with the name test_with_all_false, which is inside the given directory.
    """
    df = pd.read_pickle(database)
    collection = pd.read_csv(directorio + '/compounds_collection.csv')
    
    cifs = [i for i in collection['cif']]
    maxsites = np.max(collection['sitios'])
    
    df = df[df['sitios'] > 0][df['sitios'] <= maxsites].reset_index(drop=True)
    df = df.loc[~df['cif'].isin(cifs)].reset_index(drop=True)
    
    x, _, df = inout_creator(df=df)
    
    x = compute_quotients(X=x)
    x, df = append_local_functions(X = x,df=df)

    if use_rad_elec:
        xraw, _, _ = inout_creator(df=df)
        xre = add_rad_elec(X=xraw)
        
        if use_elecdiff:
            xde = compute_diffelec(X=xraw)
            xre = np.concatenate((xre, xde), axis=2)
            
        x = np.concatenate((xre, x), axis=2)
    
    if use_density:
        xdensity = append_density(df=df)
        x = np.concatenate((x, xdensity), axis=2)
    
    #busqueda = "ls " + directorio + "/*.h5 > model_names.txt"
    #os.system(busqueda)
    
    diccio = np.load(directorio + '/feature_standarisation.npy').item()
    
    X = (x - diccio['mean'])/diccio['std']
    x = np.reshape(X,(X.shape[0],X.shape[2]))
    
    modelo = models.load_model(directorio + '/' + ann + '.h5')
    output_nodes = modelo.output_shape[1]
    
    outpred = modelo.predict(x)
    #prediction = np.argmax(outpred, axis=1)
    if output_nodes != 1:
        y_pred = np.argmax(outpred,axis=1)
    else:
        y_pred = (outpred >= 0.5)
    
    df['y_pred'] = np.ravel(y_pred)
            
    df.to_csv(directorio + '/df_test_wafalse-best.csv', index=None)
    return