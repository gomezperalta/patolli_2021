# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:24:33 2018
Patolli 2021 created on Sun Jan 24 14:41 2021

@author: iG
Contact: gomezperalta.ai@gmail.com
"""

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.utils as kutils
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.optimizers as optimizer
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as PRFS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
import copy
import os


np.random.seed(10)

def lab2symm(sgnum='001',label='a'):
    """
    This is an auxiliary function. It loads a Python dictionary which transforms
    the Wyckoff symbol, given a space group, into a point-symmetry group.
    Parameters:
        sgnum: string in format 000, the number of the space group.
        label: The Wyckoff symbol to tranform for the given sgnum
    Returns:
        A string with the point-symmetry group
    """
    wyck_dic=np.load('./support/WyckoffSG_dict.npy', allow_pickle=True).item()['wycksym']
    return wyck_dic[sgnum].get(label)

def create_dictionary(file='dictionary'):
    """
    Parameters:
        file: a txt-file which contains the spacegroups with their symmetry-site
        -occupations that define a structure. This file is transformed into a 
        python dictionary
        
    Return:
        diccio: Python dictionary
    """
    
    start=time.time()
    f=list(filter(None,open(str(file)+'.txt','r').read().split('\n')))

    sg_ikeys=[f.index(sg) for sg in f if 'spacegroup' in sg]+[len(f)]
    sg_keys=[str(int(sg.split(':')[1])).zfill(3) for sg in f if 'spacegroup' in sg]
    
    diccio={}

    for item in range(len(sg_ikeys)-1):
        text=f[sg_ikeys[item]+1:sg_ikeys[item+1]]
        option=[text.index(i) for i in text if 'option' in i]+[len(text)]

        dicc_1={}
        for inneritem in range(len(option)-1):
            innertext=text[option[inneritem]+1:option[inneritem+1]]
                    
            values=[]
            for letra in innertext:
                
                temp_values=[i for j in [s.split(',') for s in [letra.split(':')[1].replace(' ','')]] for i in j]
                values=values+temp_values
            dicc_1[inneritem]=[lab2symm(sgnum=sg_keys[item],label=j) for j in sorted(values)]
        
        diccio[sg_keys[item]]=dicc_1
    print('Crystal structure definition text file already converted into a Python dictionary in', 
          round(time.time()-start,2),' s')    
    return diccio

def ctrl_dictionary(archivo='model_control_file'):
    """
    Parameters: A txt - file which has all the hyperparameters for the 
                trainings of  the artificial neural networks. 
                This txt - file is converted into
                a python dictionary for being used in create_patolli - function.
    Return: A dictionary for create_patolli function
    """
    
    f=list(filter(None,open(str(archivo)+'.txt','r').read().split('\n')))

    sg_ikeys=[f.index(sg) for sg in f if 'NAME' in sg]+[len(f)]
    
    diccio={}
    for item in range(len(sg_ikeys)-1):
        text = f[sg_ikeys[item]:sg_ikeys[item+1]]
        key = [entry.split(':')[0] for entry in text]
        value = [entry.split(':')[1] for entry in text]
        diccio[item] = {k:v for k,v in zip(key,value)}

    return diccio

def create_binary_collection(database='./support/red_cod-db.pkl', sites=-1, elements=-1,
                    maxatoms=-1, dictionary='diccionario'):
    """
    This function creates the collection of compounds needed to train the
    ANNs. After its execution, a file named as 'compounds_collection.csv'
    is automatically saved. 
    This function also calls lab2symm and create_dictionary.
    Parameters: 
        database: A pickle file which summarizes the crytal information of
                the Crystallography Open Database. The pickle file
                must contain the occupation of the Wyckoff sites 
                as a Python dictionary in order that patolly can normally
                executes. 
        sites: int, it constrains the creation of the compounds collection
                to samples with a maximum number of Wyckoff sites. 
                By default, it uses the maximum number of sites found for True samples.
        elements: int, it constrains the true samples of the compounds collection
                to have at least a number of different elements in the formula.
                By default
        maxatoms: int, it constrains the false samples of the compounds collection
                to have at most a maxatoms-number of atoms within the unit cell.
        dictionary: A txt - file which contains the spacegroups and the symmetry
                    sites which define a given structure.
    Returns:
        final_data: A pandas DataFrame which contains True and False samples. 
                    This DataFrame is saved as 'compounds_collection.csv', as well.
    """
    diccionario = create_dictionary(file=str(dictionary))
    
    sitios=max([len(j) for i in diccionario.values() for j in i.values()])
    start=time.time()
    print('Loading database. This may take some time...')
    df = pd.read_pickle(database)
    print('Database loaded. This process took ',np.round(time.time()-start,2))
   
    if sites == -1:
        df = df[df['sitios'] <= sitios].reset_index(drop=True)
    else:
        df = df[df['sitios'] <= sites].reset_index(drop=True)
    
    if maxatoms != -1:
        df = df[df['atoms'] <= maxatoms].reset_index(drop=True)

    data = df
    #Next line gets compounds having candidate spacegroups
    trydata=data.loc[data['sgnum'].isin([int(i) for i in \
                     list(diccionario.keys())])].reset_index(drop=True)
    wyck_dic=np.load('./support/WyckoffSG_dict.npy').item()['wycksym']
    
    #Next line creates the column labels, related to Wyckoff sites.
    trydata['labels'] = [[list(item.keys())[0] for item in \
           list(trydata['WyckOcc'][row].values())] for row in range(len(trydata))]
    
    #Next line creates the column symmetry, which transforms Wyckoff labels to point symmetry groups.
    trydata['symmetry'] = [[wyck_dic[str(trydata['sgnum'][row]).zfill(3)].get(letra) \
           for letra in trydata['labels'][row]] for row in range(len(trydata))]
    
    #Next for-loop identifies the compounds matching the crystal structure that we are looking for.
    #The crystal structure was defined in the structure_dictionary.txt
    target=[]
    for row in range(len(trydata)):
    
        spacegroup = str(trydata['sgnum'][row]).zfill(3)
        comparador=list(diccionario[spacegroup].values())
    
        if trydata['symmetry'][row] in comparador:
            target.append(True)
        else:
            target.append(False)
        
    target=pd.Series(target,name='target')
    trydata=trydata.join(target)
    print('True data identified')
    #Next line preserves the compounds matching with the crystal structure
    data_true=trydata[trydata['target'] == True].reset_index(drop=True)
    exclude_cif=trydata[trydata['target'] == True]['cif']
    sitios=list(set(data_true['sitios']))
    #Next line restores the not-matching candidate compounds to the whole database.
    data_rem=df.loc[~df['cif'].isin(exclude_cif)].reset_index(drop=True)
    
    final_data=data_true[['cif','formula','WyckOcc','sgnum','sitios',
                          'atoms','elements','target']]
    
    #Next for-loop picks randomly up some crystal compounds to represent false crystal compounds.
    #Those false crystal compounds have a site distribution similar to the true crystal compounds.
    for item in sitios:
        data_temp=data_rem[data_rem['sitios'] == item].reset_index(drop=True)
        
        cantidad=len(data_true[data_true['sitios'] == item])

        vector=np.random.permutation(np.random.permutation(np.random.permutation(np.arange(len(data_temp)))))[:cantidad]
        data_temp=data_temp.take(vector)
        data_temp['target']=False
        final_data=pd.concat((final_data, data_temp), ignore_index=True)
        
    print('Crystal compounds collection to train the ANNs created in',
          round(time.time()-start,2),' s')
    
    final_data = final_data[['cif','formula','sgnum','WyckOcc','sitios',
                             'elements','atoms','target']]
    
    #If minimal number of elements was established, the next statements drops
    #those true samples. A same amount of false sample is dropped randomly.
    if elements != -1:
        false_positive = final_data[final_data['target'] == True][final_data['elements'] < elements].index
        if len(false_positive) != 0:
            numcif_todrop = len(false_positive)
            false_todrop = np.random.choice(final_data[final_data['target'] == False].index,numcif_todrop)
            idx_todrop = np.append(false_positive, false_todrop)
            
            final_data = final_data.drop(final_data.index[idx_todrop])
            final_data = final_data.reset_index(drop=True)            
            
    final_data.to_csv('compounds_collection.csv', index=None)
        
    return final_data

def create_collection(database='./support/red_cod-db.pkl', sites=-1, elements=-1,
                    maxatoms=-1, dictionary='diccionario', not_identified=True):
    """
    This function creates the collection of compounds needed to train the
    ANNs. After its execution, a file named as 'compounds_collection.csv'
    is automatically saved. 
    This function also calls lab2symm and create_dictionary.
    Parameters: 
        database: A pickle file which summarizes the crytal information of
                the Crystallography Open Database. The pickle file
                must contain the occupation of the Wyckoff sites 
                as a Python dictionary in order that patolly can normally
                executes. 
        sites: int, it constrains the creation of the compounds collection
                to samples with a maximum number of Wyckoff sites. 
                By default, it uses the maximum number of sites found for True samples.
        elements: int, it constrains the true samples of the compounds collection
                to have at least a number of different elements in the formula.
                By default
        maxatoms: int, it constrains the false samples of the compounds collection
                to have at most a maxatoms-number of atoms within the unit cell.
        dictionary: A txt - file which contains the spacegroups and the symmetry
                    sites which define a given structure.
        not_identified: boolean. If True, a set of compound not matching the crystal
        		structure types in the dictionary will be added as example of 
        		not identified compounds.

    Returns:
        final_data: A pandas DataFrame which contains True and False samples. 
                    This DataFrame is saved as 'compounds_collection.csv', as well.
    """

    dictionaries = dictionary.split(',')
    dictionaries = [i.strip() for i in dictionaries]
        
    start=time.time()
    print('Loading database. This may take some time...')
    df = pd.read_pickle(database)
    print('Database loaded. This process took ',np.round(time.time()-start,2))
    
    #cst stands for "Crystal Structure type"
    #Next for-loop gets the compounds matching the crystal structure types.    
    #In fact, this first part resembles to the function create_binary_collection
    for item, cst in enumerate(dictionaries):
        diccionario = create_dictionary(file=str(cst))

        sitios=max([len(j) for i in diccionario.values() for j in i.values()])
        
        if sites == -1:
            data = df[df['sitios'] <= sitios].reset_index(drop=True)
        else:
            data = df[df['sitios'] <= sites].reset_index(drop=True)
        
        if maxatoms != -1:
            data = df[df['atoms'] <= maxatoms].reset_index(drop=True)
    
        trydata=data.loc[data['sgnum'].isin([int(i) for i in \
                         list(diccionario.keys())])].reset_index(drop=True)
        wyck_dic=np.load('./support/WyckoffSG_dict.npy', allow_pickle=True).item()['wycksym']
    
        trydata['labels'] = [[list(item.keys())[0] for item in \
               list(trydata['WyckOcc'][row].values())] for row in range(len(trydata))]
    
        trydata['symmetry'] = [[wyck_dic[str(trydata['sgnum'][row]).zfill(3)].get(letra) \
               for letra in trydata['labels'][row]] for row in range(len(trydata))]
    
        target=[]
        for row in range(len(trydata)):
        
            spacegroup = str(trydata['sgnum'][row]).zfill(3)
            comparador=list(diccionario[spacegroup].values())
        
            if trydata['symmetry'][row] in comparador:
                target.append(cst)
            else:
                target.append(False)
            
        target=pd.Series(target,name='target')
        trydata=trydata.join(target)
        print(cst, ' identified')
        if item == 0:
            df_cst = trydata[trydata['target'] == cst].reset_index(drop=True)
            exclude_cif = trydata[trydata['target'] == cst]['cif'].to_list()
        else:    
            data_true=trydata[trydata['target'] == cst].reset_index(drop=True)
            df_cst = pd.concat((df_cst, data_true), ignore_index=True)
            exclude_cif += trydata[trydata['target'] == cst]['cif'].to_list()
    
    df=df.loc[~df['cif'].isin(exclude_cif)].reset_index(drop=True)
    sitios=list(set(df_cst['sitios']))
    final_data=df_cst[['cif','formula','WyckOcc','sgnum','sitios',
                          'atoms','elements','target']]
    
    idx = {k:v for v,k in enumerate(sorted(final_data['target'].unique()))}
    
    for row in range(final_data.shape[0]):
        final_data.at[row, 'target'] = idx.get(final_data['target'][row],None)
    if not_identified:
        for item in sitios:
            data_temp=df[df['sitios'] == item].reset_index(drop=True)
            
            cantidad=len(df_cst[df_cst['sitios'] == item])
    
            vector=np.random.permutation(np.random.permutation(np.random.permutation(np.arange(len(data_temp)))))[:cantidad]
            data_temp=data_temp.take(vector)
            data_temp['target']=len(idx)
            final_data=pd.concat((final_data, data_temp), ignore_index=True)
        idx['not_identified_examples'] = len(idx)
        
    final_data = final_data[['cif','formula','sgnum','WyckOcc','sitios',
                             'elements','atoms','target']]
    '''
    if elements != -1:
        false_positive = final_data[final_data['target'] == True][final_data['elements'] < elements].index
        if len(false_positive) != 0:
            numcif_todrop = len(false_positive)
            false_todrop = np.random.choice(final_data[final_data['target'] == False].index,numcif_todrop)
            idx_todrop = np.append(false_positive, false_todrop)
            
            final_data = final_data.drop(final_data.index[idx_todrop])
            final_data = final_data.reset_index(drop=True)
    '''            
    output_str = open('output_str.txt','w')    
    output_str.write(str(idx).replace(',',',\n'))
    output_str.close()
    
    final_data.to_csv('compounds_collection.csv', index=None)
    
    print('Crystal compounds collection to train the ANNs created in',
          round(time.time()-start,2),' s')
    
    return final_data

def raw_features_extractor(database='./support/red_cod-db.pkl', sites=-1, 
                           elements = -1, maxatoms= -1,
                           dictionary='diccionario', features='./support/datosrahm.csv',
                           not_identified=True):
    
    """
    This function computes the average electronegativity and atomic radius for
    each Wyckoff sites. Since in the collection of compounds can be compounds
    with a different number of occupied Wyckoff sites, this function also 
    homogenizes the number of sites of the collection of compounds. This homogenization
    is done adding extra sites with empty occupation; i.e, zero electronegativity
    and atomic radius.
    This function calls the functions lab2symm, create_dictionary and create_collection.
    The outputs of this function are saved. 
    Parameters: 
        database: A pickle file which summarizes the crytal information of
                the Crystallography Open Database. The pickle file
                must contain the occupation of the Wyckoff sites 
                as a Python dictionary in order that patolly can normally
                executes. 
        sites: int, it constrains the creation of the compounds collection
                to samples with a maximum number of Wyckoff sites.
        elements: int, it constrains the true samples of the compounds collection
                to have at least a number of different elements in the formula.
        maxatoms: int, it constrains the false samples of the compounds collection
                to have at most a maxatoms-number of atoms within the unit cell.
        dictionary: A txt - file which contains the spacegroups and the symmetry
                    sites which define a given structure.
        features: A csv-file containing the atomic radius (published by Ashcroft, Rahm
                    and Hoffmann) and the Pauling electronegativity for each element.
        not_identified: boolean. If True, a set of compound not matching the crystal
        		structure types in the dictionary will be added as example of 
        		not identified compounds.
    Returns:
        X: A tensor which dimensions are the samples x sites x pair atomic radius and electronegativity.
            This is saved as raw_features.npy
        y: A tensor with the output values. This is saved as output_values.npy
        S: A tensor which dimensions are the samples x sites x multiplicity of the site
            This is saved as multiplicites.npy
        fracsum: A tensor which dimensions are the samples x sites x the occupation fraction of the site
            This is saved as occupation_fractions.npy
        df: The collection of the compounds created with the function create_collection.
    """
    
    length_dict = len(dictionary.split(','))
    
    if length_dict != 1:
        df=create_collection(database=database,sites=sites, elements=elements, maxatoms=maxatoms, 
                       dictionary=dictionary, not_identified = not_identified)
    else:
        df=create_binary_collection(database=database,sites=sites, elements=elements, maxatoms=maxatoms, 
                       dictionary=dictionary)
    
    start=time.time()
    
    datos=pd.read_csv(features)
    datos=datos.fillna(-1)

    dicc=dict(datos[['Symbol','Z']].values)

    dicc['D']=1
    dicc['Bk']=97
    dicc['Cf']=98
    dicc['Es']=99
    dicc['Fm']=100
    dicc['Md']=101
    dicc['No']=102
    dicc['Lr']=103
    
    max_sitios = max(df['sitios'].values)

    df=df[df['sitios'] <= max_sitios].reset_index(drop=True)
    
    X=np.zeros((len(df),max_sitios,104))
    mult=np.zeros((len(df),max_sitios))
    wyckmul=np.load('./support/WyckoffSG_dict.npy', allow_pickle=True).item()['wyckmul']
    
    if length_dict != 1:
        y=np.zeros((len(df), len(df['target'].unique())))
    else:
        y=np.zeros((len(df),1))

    todelete = list()    
    for row in range(len(df)):
        
        item=df['WyckOcc'][row]
        sitios=list(item.values()) 
        sitocc=np.zeros((len(sitios),104))
        spacegroup = str(df['sgnum'][row]).zfill(3)
        
        try:
        
            s=[int(wyckmul[spacegroup][i]) for j in [list(item.keys()) for item in \
               sitios] for i in j]
        
        except:
            print('The compound with the cif ', df['cif'][row], ' will be deleted')
            print('The database will be updated')
            todelete += [row]
            continue
            
        occs=[]
        for i in range(len(sitios)):

            for j in list(sitios[i].values()):
                
                ocupacion=np.array(list(j.values()))
                llaves=[llave.replace('+','').replace('-','').replace('1',
                        '').replace('2','').replace('3','').replace('4',
                                   '') for llave in np.array(list(j.keys()))]
                llaves=[llave.replace('.','') for llave in llaves]
                llaves=[llave.replace('5','').replace('6','').replace('7',
                        '').replace('8','').replace('9','').replace('0',
                                   '') for llave in llaves]
                vector=np.zeros((1,104))
                occs=[sum(ocupacion)]+occs
                
                try:
                    
                    idx=[dicc[k] for k in llaves]
                
                except:
                    
                    print('The compound with the cif ', df['cif'][row], ' will be deleted')
                    print('The database will be updated')
                    todelete += [row]
                    continue
                    
                for k in idx:
                    vector[0][k-1] = ocupacion[idx.index(k)]
                
            sitocc[i]=vector
    
        while sitocc.shape[0] != max_sitios:
            sitocc=np.concatenate((np.zeros((1,104)),sitocc))
            s=[0]+s
        
        X[row,:,:]=sitocc
        y[row, df['target'][row]] = 1
        mult[row]=s
    
    S = np.expand_dims(mult,axis=2)
    features=datos.iloc[:,2:5].values
    x=X[:,:,:96]
    print(x.shape)
    fracsum = np.expand_dims(np.sum(x,axis=2), axis=2)
    
    x=np.dot(x,features)    
    
    x = np.delete(x, todelete, axis=0)
    y = np.delete(y, todelete, axis=0)
    S = np.delete(S, todelete, axis=0)
    fracsum = np.delete(fracsum, todelete, axis=0)
    df = df.drop(df.index[todelete]).reset_index(drop=True)
    
    print('Atomic radii and electronegativities for each Wyckoff site extracted in',
          round(time.time()-start,2),' s')   
    
    np.save('raw_features', x)
    np.save('output_values', y)
    np.save('multiplicities', S)
    np.save('occupation_fractions', fracsum)
    
    return x, y, S, fracsum, df

def add_rad_elec(X = np.ones((1,1,1))):
    """
    This function gets the average atomic radii and electronegativities per site for each compound. 
    Parameters:
	X: a Numpy array
    Returns:
	x: a Numpy array. The dimensions of the array correspond to a tensor of shape 
	(samples,1,atomic radii and electronegativites).
    """
    X = X[:,:,:2]
    x = X.reshape((X.shape[0], 1, X.shape[1]*X.shape[2]))
    return x
    

def inout_creator(df = pd.DataFrame(), features='./support/datosrahm.csv'):
    """
    This is a function similar to raw_features_extractor, but it does not
    call previous functions. The User must provide a Pandas DataFrame
    with the collection of compounds.The outputs of this function are not
    saved.
    
    Parameters: 
        df:  A pandas DataFrame which contains information about the spacegroups and
                    the occupied symmetry sites. This must be specified with extension.
        features: A csv - file which contains the features to be use for each present 
                element in the sites of the structure.
    Returns:
        X: A tensor which dimensions are the samples x sites x pair atomic radius and electronegativity.
            This is saved as raw_features.npy
        fracsum: A tensor which dimensions are the samples x sites x the occupation fraction of the site
            This is saved as occupation_fractions.npy
        df: The collection of the compounds created with the function create_collection.
    """
    df = df
   
    start=time.time()
    
    datos=pd.read_csv(features)
    datos=datos.fillna(-1)

    dicc=dict(datos[['Symbol','Z']].values)

    dicc['D']=1
    dicc['Bk']=97
    dicc['Cf']=98
    dicc['Es']=99
    dicc['Fm']=100
    dicc['Md']=101
    dicc['No']=102
    dicc['Lr']=103
    
    max_sitios = max(df['sitios'].values)
    
    X=np.zeros((len(df),max_sitios,104))

    mult=np.zeros((len(df),max_sitios))
    wyckmul=np.load('./support/WyckoffSG_dict.npy', allow_pickle=True).item()['wyckmul']
    
    todelete = list()
    
    for row in range(len(df)):
        item=df['WyckOcc'][row]
        sitios=list(item.values()) 
        sitocc=np.zeros((len(sitios),104)) 
        spacegroup = str(df['sgnum'][row]).zfill(3)
        
        try:
        
            s=[int(wyckmul[spacegroup][i]) for j in [list(item.keys()) for item in \
               sitios] for i in j]
        
        except:
            print('The compound with the cif ', df['cif'][row], ' will be deleted')
            print('The database will be updated')
            todelete += [row]
            continue

        occs=[]
        for i in range(len(sitios)):

            for j in list(sitios[i].values()):
                
                ocupacion=np.array(list(j.values()))
                llaves=[llave.replace('+','').replace('-','').replace('1',
                        '').replace('2','').replace('3','').replace('4',
                                   '') for llave in np.array(list(j.keys()))]
                llaves=[llave.replace('.','') for llave in llaves]
                llaves=[llave.replace('5','').replace('6','').replace('7',
                        '').replace('8','').replace('9','').replace('0',
                                   '') for llave in llaves]
                vector=np.zeros((1,104))
                occs=[sum(ocupacion)]+occs
                
                try:
                    
                    idx=[dicc[k] for k in llaves]
                
                except:
                    print('The compound with the cif ', df['cif'][row], ' will be deleted')
                    print('The database will be updated')
                    todelete += [row]
                    continue
                
                for k in idx:
                    vector[0][k-1] = ocupacion[idx.index(k)]
                        
            sitocc[i]=vector
        
        while sitocc.shape[0] != max_sitios:
            sitocc=np.concatenate((np.zeros((1,104)),sitocc))
            s=[0]+s
        
        X[row,:,:]=sitocc
        mult[row]=s
    
    features=datos.iloc[:,2:5].values
    x=X[:,:,:96]
    
    fracsum = np.expand_dims(np.sum(x,axis=2), axis=2)
    
    x=np.dot(x,features)    
 
    x = np.delete(x, todelete,axis=0)
    df = df.drop(df.index[todelete]).reset_index(drop=True)
    
    print('inout_creator lasted ',round(time.time()-start,2),' s')    
    return x, fracsum, df

def compute_quotients(X = np.zeros((1,1,2))):
    """
    Returns the atomic radii pair quotients and the atomic radii 
    pair sum - quotients as a numpy array. This is the first part of
    all the features used to train the ANNs. The output of 
    this function is saved as X.npy
    
    Parameters:
        X: A numpy array, which is created with the function raw_features_extractor
    Returns:
        X: A numpy array of dimension [samples,1,features]
    """
    
    start=time.time()
    rad = X[:,:,1]

    drad = np.asarray([[item[0]/item[1] if item[1] != 0 else 0 for item in list(itertools.combinations(rad[sample],2))] \
                        for sample in range(X.shape[0])])

    dradsum = np.asarray([[item[0]/item[1] if item[1] != 0 else 0 for item in itertools.combinations([ \
                       item[0]+item[1] for item in list(itertools.combinations(rad[sample],2))], 2)] \
                       for sample in range(drad.shape[0])])
    
    drad = np.reshape(drad,(drad.shape[0],1,drad.shape[-1]))
    drads = np.reshape(dradsum,(dradsum.shape[0],1,dradsum.shape[-1]))
    
    Xq = np.concatenate((drad,drads), axis=2)
    print('Geometric and packing factors computed in', round(time.time()-start,2),' s')
    np.save('X', Xq)
    
    return Xq

def compute_diffelec(X = np.zeros((1,1,1))):
    """
    This function gets the electronegativity differences between the sites.
    
    Parameters:
        X: A numpy array, which is created with the function raw_features_extractor
    Returns:
        X: A numpy array of dimension [samples,1,features]
    """
    
    start=time.time()
    elec = X[:,:,0]

    delec = np.asarray([[item[0]-item[1] if item[1] != 0 else 0 for item in list(itertools.combinations(elec[sample],2))] \
                        for sample in range(X.shape[0])])

    delec = np.reshape(delec,(delec.shape[0],1,delec.shape[-1]))
    print('Electronegativity differences computed in', round(time.time()-start,2),' s')

    return delec

def append_local_functions(X = np.zeros((1,1,1)), df = pd.DataFrame(), 
                           local_function='./support/fij_2.0_25_diccio'):
    """
    Returns the features with the local functions. In case the local function
    does not exist for a sample in the collection, this is deleted and the
    collection is updated. The X.npy is updated.
    
    Parameters:
        X: The numpy array created with compute_quotients
        df: The pandas DataFrame created with raw_features_extractor
        local_function: The numpy dictionary having the local function to use.
    Returns:
        X: The numpy array with all neccesary features for the ANNs.
        df: The pandas DataFrame updated.
    """
    start = time.time()
    print('The dictionary ' + local_function + ' will be used for local functions')
    fij = np.load(local_function + '.npy', allow_pickle=True).item()
    
    delrow = list()
    n = np.max(df['sitios'])
    
    f = np.zeros((df.shape[0],n,n))
    
    for row in range(df.shape[0]):
        if df['cif'][row] not in fij.keys():
            delrow += [row]
        else:
            loc = fij[df['cif'][row]]
            s = loc.shape[1]
            f[row,-s:,-s:] = loc
    
    if len(delrow) != 0:
        
        print('The compounds with the next cifs will be deleted since ',
              'their local functions are not currently available')
        print([df['cif'][i] for i in delrow])
        print('The compound collection will be updated')
        
        totake = [i for i in range(df.shape[0]) if i not in delrow]
        df = df.take(totake).reset_index(drop=True)
        X = X[totake]
        f = f[totake]
        df.to_csv('compounds_collection.csv', index=None)

    fn = np.zeros((f.shape[0], f.shape[1], f.shape[2] - 1))        
    for item in range(f.shape[0]):
        delec = f[item]
        delec = delec[~np.eye(delec.shape[0], dtype=bool)].reshape(delec.shape[0],-1)
        fn[item] = delec
    
    f = fn

    f = f.reshape((f.shape[0], 1, f.shape[1]*f.shape[2]))
    
    X = np.concatenate((X,f), axis = 2)
    print('Local functions appended to features in ', round(time.time()-start,2),' s')
    return X, df

def append_density(df=pd.DataFrame(), density_diccio='./support/density_diccio.npy'):
    """
	This function appends to the input vector the density of the compound. 
	The density of the compound is contained in a Python dictionary.
	Parameters:
		df: A pandas DataFrame, containing the compounds of the collection to train the ANNs.
		density_diccio: The Python dictionary to be loaded.
	Returns:
		An array with the density of the compounds of the collection.
    """
    diccio = np.load(density_diccio, allow_pickle=True).item()
    
    xdensity = np.zeros((df.shape[0], 1, 1))
    for row in range(df.shape[0]):
        xdensity[row,0] = diccio[df['cif'][row]] 
        
    return xdensity

def split_collection(X = np.zeros((1)), df = pd.DataFrame(), frac = 0.15):
    """
    Splits the complete compounds collection in two sets:
        one for training and cross - validation and another for testing.
    Parameters:
        X: A numpy array with the features of all compounds in the collection to split.
        df: A pandas DataFrame with all the compounds.
        frac: The fraction reserved to create the test - set. If frac equals zero,
        arguments are passed to returns without modifications.
    Returns:
        Xtraval: A numpy array with the features of the compounds in the 
        training and cross - validation sets. This is saved as Xtraval.
        Xtest: A numpy array with the features of the compounds in the 
        test set. This is saved as Xtest.
        dftraval: A panda DataFrame with the compounds in the training and
        cross validation sets. This is saved as dftraval.
        dftest: A panda DataFrame with the compounds in the test set. This is 
        saved as dftest.
    """
    if frac != 0:
        
        traval = list()
        test = list()
        for item in df['target'].unique():
            globals() ['idx_' + str(item)] = df[df['target'] == item].index
            globals () ['test_cst_' + str(item)] = np.random.choice(globals() ['idx_' + str(item)], 
                    size = int(frac*len(globals() ['idx_' + str(item)])), replace = False)
            globals() ['test_cst_' + str(item)] = [i for i in globals() ['test_cst_' + str(item)]]
            globals() ['traval_cst_' + str(item)] = [i for i in globals() ['idx_' + str(item)] if i not in globals() ['test_cst_' + str(item)]]
            
            traval += globals() ['traval_cst_'+str(item)]
            test += globals() ['test_cst_'+str(item)]
        
        Xtraval = X[traval]
        Xtest = X[test]

        dftraval = df.take(traval).reset_index(drop=True)
        dftest = df.take(test).reset_index(drop=True)

        np.save('Xtraval', Xtraval)
        np.save('Xtest', Xtest)

        dftraval.to_csv('dbtraval.csv', index=None)
        dftest.to_csv('dbtest.csv', index=None)
    
    else:
        Xtraval = X
        dftraval = df
        Xtest = None
        dftest = None
    
    return Xtraval, Xtest, dftraval, dftest

def plotgraph(readfile='archivo.csv', outfiles='',cost_function='Categorical Cross Entropy'):
    """
    An auxiliar function to plot the information generated after the training
    of a ANN.
    """
    plt.rcParams['figure.figsize']=(12,9)
    
    df=pd.read_csv(str(readfile), header=None)
    df=df.rename(columns={0:'cost_train', 1:'acc_train', 2:'cost_test', 3:'acc_test'})

    plt.figure(1)
    plt.title('Accuracy', fontsize=20, fontweight='bold')
    train=plt.scatter(np.arange(1,len(df)+1,1), df['acc_train'].values*100, color='red',
                      marker='x',s=100)
    test=plt.scatter(np.arange(1,len(df)+1,1), df['acc_test'].values*100, color='blue',
                     marker='o',s=100)
    plt.legend([train,test],['Training set', 'CV set'], fontsize=18)
    plt.xlabel('Epochs',fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Accuracy %', fontsize=20, fontweight='bold')
    plt.annotate(str(round(df['acc_train'].values[-1]*100,2))+' %', xy=(len(df),int(df['acc_train'].values[-1]*100)), 
                 xytext=(len(df),int(df['acc_train'].values[-1]*100-1)), fontsize=14, fontweight='bold')
    plt.annotate(str(round(df['acc_test'].values[-1]*100,2))+' %', xy=(len(df),int(df['acc_test'].values[-1]*100)),
                 xytext=(len(df),int(df['acc_test'].values[-1]*100+2)), fontsize=14, fontweight='bold')
    plt.savefig('Accuracy'+'_'+str(outfiles)+'.png')
    
    plt.figure(2)
    plt.title(str(cost_function), fontsize=20, fontweight='bold')
    train=plt.scatter(np.arange(1,len(df)+1,1), df['cost_train'].values, color='red',
                      marker='x',s=100)
    test=plt.scatter(np.arange(1,len(df)+1,1), df['cost_test'].values, color='blue',
                     marker='o',s=100)
    plt.legend([train,test],['Training set', 'CV set'], fontsize=18)
    plt.xlabel('Epochs',fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Cost function', fontsize=20, fontweight='bold')
    plt.annotate(str(round(df['cost_train'].values[-1],4)), xy=(len(df),df['cost_train'].values[-1]), 
                 xytext=(len(df),1.1*df['cost_train'].values[-1]), fontsize=14, fontweight='bold')
    plt.annotate(str(round(df['cost_test'].values[-1],4)), xy=(len(df),df['cost_test'].values[-1]),
                 xytext=(len(df),1.1*df['cost_test'].values[-1]+0.05), fontsize=14, fontweight='bold')
    plt.savefig('Cost_function'+'_'+str(outfiles)+'.png')
    plt.close('all')

    return 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function is taken from http://scikit-learn.org/stable/auto_examples/
    model_selection/plot_confusion_matrix.html#
    sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.rcParams['figure.figsize']=(12,9)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=18, fontweight='bold')

    plt.ylabel('True label', fontsize=18, fontweight='bold')
    plt.xlabel('Predicted label', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    return


def modelo(hidden_layers=[1], activation='tanh',features=1, 
           beta_1=0.9, beta_2=0.999,lr=0.001, decay=1e-6, dropout=0,
           output_nodes=1):
    """
    With this function, the architecture of an ANN is given.
    Some hyperparameters must be provided as well. The optimization
    is accomplished with Adam.
    Parameters:
        hidden_layers: A list of floats. The length of the list
                    represents the number of hidden layers in the ANN.
                    Each element of the list 
                    is a multiple (or submultiple) that multiplies
                    the number of features of the model. For example,
                    if the number of features is 30 and the list
                    is [0.5, 2], then the ANN has two hidden layers,
                    with 30 and 60 nodes, respectively.
                    By default, the list is [1]
        activation: string, the name of the activation function to
                    use in the hidden layers. By default, it is tanh.
                    Check the keras documentation to know all the 
                    variety of activation functions.
        features: int, the number of features. By default, it is 1.
        beta_1: float, An hyperparameter of the Adam optimizer. 
                By default it is 0.9
        beta_2: float, An hyperparameter of the Adam optimizer.
                By default, it is 0.999
        lr: float, the learning rate. By default, it is 0.001
        decay: float, decay of the learning rate. By default, it is 1e-6.
        dropout: float, the fraction of the nodes to drop out during a batch.
                By default, it is 0.
        output_nodes: The number of ouputs in the last layer.
    """
        
    input_layer = layers.Input(shape=(features,))

    for item in range(len(hidden_layers)):

        if item == 0:
            vmiddle = layers.Dense(hidden_layers[item], 
                                   kernel_initializer='random_uniform')(input_layer)
        else:
            vmiddle = layers.Dense(hidden_layers[item], 
                                   kernel_initializer='random_uniform')(vmiddle)
    
        vmiddle = layers.LayerNormalization()(vmiddle)

        if item == len(hidden_layers) - 1:
            skipConnection = layers.Dense(hidden_layers[item], 
                                   kernel_initializer='random_uniform')(input_layer)
            skipConnection = layers.LayerNormalization()(skipConnection)
            vmiddle = layers.Add()([vmiddle, skipConnection])

        if activation == 'leaky_relu':
            vmiddle = layers.LeakyReLU(0.15)(vmiddle)
        else:
            vmiddle = layers.Activation(activation)(vmiddle)
        
        if dropout: vmiddle = layers.Dropout(dropout)(vmiddle)
            
    if output_nodes == 1:
        vexit =layers.Dense(1, kernel_initializer='random_uniform')(vmiddle)
        vexit =layers.Activation('sigmoid')(vexit)
    else:
        vexit =layers.Dense(output_nodes, kernel_initializer='random_uniform')(vmiddle)
        vexit =layers.Activation('softmax')(vexit)
    
    model = models.Model(inputs=input_layer, outputs=vexit)
    
    if output_nodes == 1:
        model.compile(loss='binary_crossentropy', 
                      optimizer=optimizer.Adam(beta_1=beta_1, beta_2=beta_2, lr=lr, decay=decay,), 
                      metrics=['acc'])
    else:
        model.compile(loss='categorical_crossentropy', 
                      optimizer=optimizer.Adam(beta_1=beta_1, beta_2=beta_2, lr=lr, decay=decay,), 
                      metrics=['acc'])
    
    return model

def training(model, X, Y,epochs=300,batch_size=16,test_frac=0.30, saveas='modelo_nn', verbose=1):
    
    """
    With this function, the models defined with the function modelo are trained
    Parameters:
        model: the model defined with the function modelo
        X: The input data (it should be X.npy)
        Y: The output values (it should be Y.npy)
        epochs: The number of epochs (Default: 300)
        batch_size: The number of samples in a batch (Default: 16)
        test_frac: The fraction reserved to cross-validate (Default: 0.30)
        saveas: The name coming with files created after the training (Default: modelo_nn)
        verbose: verbosity (Default 1. Zero verbosity with 0 and abundant with 2)
    Regresa:
        Files having Accuracy vs. epochs and Cost function vs. epochs.
    """
    modelCheckpoint=callbacks.ModelCheckpoint(str(saveas)+'.h5', monitor='val_loss', 
                                                    verbose=0, save_best_only=True, mode='auto')
    history = callbacks.History()
    data = model.fit(X,Y,validation_split=test_frac, epochs=epochs,batch_size=batch_size,
                     callbacks=[modelCheckpoint,history],shuffle=True, verbose=verbose)
    try:
        kutils.plot_model(model,to_file=str(saveas)+'.png', show_shapes=True, show_layer_names=True)
    except:
        print('GraphViz is not installed and therefore keras.utils cannot call it to plot the models')
    
        
    """ Creacion del archivo csv """
    acc_log = data.history['acc']
    val_acc_log = data.history['val_acc']
    loss_log = data.history['loss']
    val_loss_log = data.history['val_loss']
    acc_log = np.array(acc_log)
    val_acc_log = np.array(val_acc_log)
    loss_log = np.array(loss_log)
    val_loss_log = np.array(val_loss_log)
    mat = np.vstack((loss_log, acc_log, val_loss_log, val_acc_log))
    mat = np.transpose(mat)
    dataframe1 = pd.DataFrame(data=mat)
    dataframe1.to_csv(str(saveas)+'.csv', 
                      header= None,
                      sep=',',  float_format='%.7f', index=False)
    
    return data, dataframe1, model

def test_models(directorio='', output_values=1):
    """
    This function tests all models once the training finished.
    Parameters:
        directorio: A string with the name of the directory where the models are.
        output_values: The number of nodes in the last layer of the ANN.
    Returns:
        a txt - file with the name test_results, which is inside the given directory.
    """
    
    print('The trained models will be tested now')
    start = time.time()
    
    busqueda = "ls " + directorio + "/*.h5 > model_names.txt"

    os.system(busqueda)

    X = np.load(directorio + '/Xtest.npy')
    diccio = np.load(directorio + '/feature_standarisation.npy').item()
    y = pd.read_csv(directorio + '/dbtest.csv')['target'].values

    X = (X - diccio['mean'])/diccio['std']
    x = np.reshape(X,(X.shape[0],X.shape[2]))
    
    with open('model_names.txt','r') as f:
        for line in f:
            modelo = models.load_model(line[:len(line)-1])
            nombre = line.split('/')[1]
            outpred = modelo.predict(x)
            prediction = outpred >= 0.5
            
            if output_values != 1:
                prediction = np.matmul(prediction, 
                               np.arange(output_values).reshape(output_values,1))
                prediction = prediction.ravel()

            precision,recall,fscore,support = PRFS(y, prediction)
            
            with open(directorio + '/test_results.txt','a') as tr:
                tr.write(nombre + '\n')
                tr.write('samples: '+str(support)+'\n')
                tr.write('precision: '+str(np.round(precision*100,2))+'\n')
                tr.write('recall: '+str(np.round(recall*100,2))+'\n')
                tr.write('f1-score: '+str(np.round(fscore*100,2))+'\n')
                tr.write('\n')
                tr.close()
    
    print('The test of all trained models lasted ', round(time.time()-start,2),' s')
    os.system('rm model_names.txt')
    
    return

def test_all_false(directorio = str(), database = './support/red_cod-db.pkl', 
                   local_function = './support/fij_2.0_25_diccio',
                   use_rad_elec=True, use_elecdiff=False, use_density=True):
    """
    This function tests all models once the training finished with the remaining
    false samples of red_cod-db.pkl.
    Parameters:
        directorio: A string with the name of the directory where the models are.
        database: a pickle file containing the entire database
        local_function: a numpy dictionary with the local functions to append.
        use_rad_elec: boolean, if features containing the atomic radii and electronegativities per site
        		are used in the input data.
		use_elecdiff: boolean, if the electronegativity differences are used in the input data.
		use_density: boolean, if the density of the compound is used in the input data.
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
    x, df = append_local_functions(X = x,df=df, local_function=local_function)
    
    if use_rad_elec:
        xraw = inout_creator(df=df)
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
            prediction = outpred >= 0.5
            df['y_pred'] = np.ravel(prediction)
                
            with open(directorio+'/test_with_all_false.txt','a') as tr:
                tr.write(nombre + '\n')
                
                for sitios in range(1, max(df['sitios']) + 1):
                
                    acc = df[df['sitios'] == sitios][df['y_pred'] == False].shape[0]
                    miniset = df[df['sitios'] == sitios].shape[0]
                    percent = round(100*acc/miniset,2)
                
                    
                    tr.write('With '+ str(sitios) + ' sites:' + str(percent) +\
                             '(' + str(miniset) + ' samples)' + '\n')
                tr.close()
    return

def prfs_and_cnfmat(model, class_names, saveas = '',
                     name = '', df = pd.DataFrame(), x = np.asarray(1)):
    
    output_nodes = model.output_shape[1]
    predictions = model.predict(x)
    y_true = df['target']
    
    if output_nodes != 1:
        y_pred = np.argmax(predictions,axis=1)
    else:
        y_pred = (predictions >= 0.5)
        
    precision, recall, fscore, support = PRFS(y_true, y_pred)
    cnf_matrix=confusion_matrix(df['target'],y_pred)
    np.save(name+ '_' + saveas + '_cnfmat.npy',cnf_matrix)
    precision = np.round(100*precision,2)
    recall = np.round(100*recall,2)
    fscore = np.round(100*fscore,2)
    
    with open('prfs_' + saveas + '.txt', 'a') as prfs:
        prfs.write(name+'\n')
        prfs.write('classes: '+str(class_names)+'\n')
        prfs.write('samples: '+str(support)+'\n')
        prfs.write('precision: '+str(precision)+'\n')
        prfs.write('recall: '+str(recall)+'\n')
        prfs.write('f1-score: '+str(fscore)+'\n')
        prfs.write('\n')
        prfs.close()
     
    plt.figure(1)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.savefig('cnfmat_' + name + '_' + saveas + '.png')
    plt.close('all')
    
    return
    
def create_patolli(database='./support/red_cod-db.pkl', sites = -1, elements=-1, maxatoms=-1,
               dictionary='structure_dictionary', features='./support/datosrahm.csv',
               control_file='model_control_file', not_identified=True,
               verbose=1, test_frac = 0.15, local_function='./support/fij_2.0_25_diccio',
               use_density=True, use_rad_elec = True, use_elecdiff = False,
               test_with_all_false = False):
    
    """
    This is the function to develop the artificial neural networks.
    It calls all the mentioned before to create the models from the very beginning.
    
    Parameters: 
        database: A pickle file which summarizes the crytal information of
                the Crystallography Open Database. The pickle file
                must contain the occupation of the Wyckoff sites 
                as a Python dictionary in order that patolly can normally
                executes. 
        sites: int, it constrains the creation of the compounds collection
                to samples with a maximum number of Wyckoff sites.
        elements: int, it constrains the true samples of the compounds collection
                to have at least a number of different elements in the formula.
        maxatoms: int, it constrains the false samples of the compounds collection
                to have at most a maxatoms-number of atoms within the unit cell.
        dictionary: A txt - file which contains the spacegroups and the symmetry
                    sites which define a given structure.
        features: A csv-file containing the atomic radius (published by Ashcroft, Rahm
                    and Hoffmann) and the Pauling electronegativity for each element.
        control_file: A txt - file which has all the characteristics for the training of
                    each neural network.
        not_identified: boolean. If True, a set of compound not matching the crystal
        		structure types in the dictionary will be added as example of 
        		not identified compounds.
        local_function: a numpy dictionary with the local functions to append.
        use_rad_elec: boolean, if features containing the atomic radii and electronegativities per site
        		are used in the input data.
		use_elecdiff: boolean, if the electronegativity differences are used in the input data.
		use_density: boolean, if the density of the compound is used in the input data.
        test_frac: The fraction reserved to cross-validate (Default: 0.30)
        verbose: verbosity (Default 1. Zero verbosity with 0 and abundant with 2)
        test_with_all_samples: if the function test_with_all_samples must be executed
                                after the training of all the models was accomplished
    Returns:
        A h5 - model with the trained neural network, plots of accuracy and cost function,
        as well as crude and normalized confusion matrices, a csv - file with data about
        accuracy and cost function for training and test sets for all epochs and a txt - file
        with information about precision - recall - F1 score for each model.
        
    """
    
    start_main=time.time()
    
    length_dict = len(dictionary.split(','))
    
    X, _, _, _, df = raw_features_extractor(database=database, sites = sites, 
                                            elements = elements, maxatoms = maxatoms, 
                                            dictionary=dictionary, features=features,
                                            not_identified=not_identified)

    X = compute_quotients(X=X)
    X, df = append_local_functions(X = X, df = df, local_function = local_function)
    
    if use_rad_elec:
        xraw, _, _ = inout_creator(df=df)
        xre = add_rad_elec(X=xraw)
        
        if use_elecdiff:
            xde = compute_diffelec(X=xraw)
            xre = np.concatenate((xre, xde), axis=2)
            
        X = np.concatenate((xre, X), axis=2)
    
    if use_density:
        xdensity = append_density(df=df)
        X = np.concatenate((X, xdensity), axis=2)
    
    Xtraval, Xtest , dftraval, dftest = split_collection(X = X, df = df, frac = test_frac)

    if length_dict != 1:
        if not_identified:
            length_dict += 1
            
        Y = np.zeros((Xtraval.shape[0], length_dict))
        for row in range(dftraval.shape[0]):
            Y[row, dftraval['target'][row]] = 1
    else:
        Y = dftraval['target'].values
    
    class_names=list(set(dftraval['target']))
    
    subnets=Xtraval.shape[1]
    features=Xtraval.shape[2]
    
        
    average = np.mean(X, axis=0)    
    stdev = np.std(X, axis=0)
        
    X = (Xtraval - average)/stdev
    xtest = (Xtest -average)/stdev
    print('There are', X.shape[-1], 'features' )    
    dicfeatstand = {'mean':average,'std':stdev}
    np.save('feature_standarisation',dicfeatstand)
        
    with open('feature_standarisation.txt','w') as f:
        f.write('X matrix has dimensions '+str(X.shape[0])+' samples x ' + \
                    str(X.shape[1]) + ' sites x ' + str(X.shape[2]) + \
                    ' features'+'\n'+'\n')
        f.write('Features - mean:'+'\n'+'\n')
        f.write(str(average)+'\n'+'\n')
        f.write('Features - std:'+'\n'+'\n')
        f.write(str(stdev))
        f.close()
                
    Xor=copy.deepcopy(X)
    X,y = shuffle(X,Y,random_state=0)
    
    x={}
    xor={}
    
    for subnet in range(subnets):
        x[subnet] = X[:,subnet,:]
        xor[subnet] = Xor[:,subnet,:]
    
    directorio = time.ctime().replace(' ', '_').replace(':','_')
    os.system('mkdir ' + directorio)
    os.system('move compounds_collection.csv ' + directorio +'/')
    os.system('move multiplicities.npy ' + directorio +'/')
    os.system('move occupation_fractions.npy ' + directorio +'/')
    os.system('move output_values.npy ' + directorio +'/')
    os.system('move raw_features.npy ' + directorio +'/')
    os.system('move X*.npy ' + directorio +'/')
    os.system('move db*.csv ' + directorio +'/')
    os.system('move feature_standarisation* ' + directorio +'/')
    try:
        os.system('move output_str.txt ' + directorio +'/')    
    except:
        print('The models to train correspond to binary classification')
        
    ctrl_diccio = ctrl_dictionary(archivo=control_file)
    print('\n')
    print('*************************************************************'+
          '*************************************************************'+
          '*************************************************************'+
          '*************************************************************')
    print('ANNs TRAINING WILL START NOW.')
    print('\n')
    print('There are ',len(ctrl_diccio.keys()),' ANNs to train')
    
    for item in list(ctrl_diccio):
        print('Training ', item+1,'/',len(ctrl_diccio.keys()))
        diccionary = ctrl_diccio[item]
    
        hidden_layers=[float(x) for x in diccionary['HIDDEN_LAYERS'].split(",")]
        epochs=int(diccionary['EPOCHS'])
        batch_size=int(diccionary['BATCH_SIZE'])
        test_frac=float(diccionary['TEST_VAL'])
        cost_function=diccionary['COST_FUNCTION']
        learning_rate=float(diccionary['LEARNING_RATE'])
        beta_1=float(diccionary['BETA_1'])
        beta_2=float(diccionary['BETA_2'])
        decay=float(diccionary['DECAY'])
        dropout=float(diccionary['DROPOUT'])
        activation=diccionary['ACTIVATION']
        name=diccionary['NAME']
        
        hidden_layers = np.asarray(hidden_layers)*features
        hidden_layers = [int(x) for x in hidden_layers]
        
        model = modelo(hidden_layers=hidden_layers, activation=activation,
                       features=features, beta_1=beta_1, beta_2=beta_2, lr=learning_rate, decay=decay, 
                       dropout=dropout, output_nodes=length_dict)
        
        start=time.time()
        data, dataframe, model = training(model, X=[x[i] for i in range(subnets)], Y = y, epochs=epochs, 
                                               batch_size=batch_size, test_frac=test_frac, saveas=name,
                                               verbose=verbose)
        
        print('NN training lasted ',np.round(time.time() - start,2),'s')
        print('\n')
        plotgraph(readfile=name+'.csv', outfiles=name, cost_function=cost_function)
        
        try:
            prfs_and_cnfmat(model = model, class_names = class_names,
                            name = name, saveas = 'traval',
                            df = dftraval, x = [xor[i] for i in range(subnets)])
            
            if test_frac != 0:
                prfs_and_cnfmat(model = model, class_names = class_names,
                            name = name, saveas = 'test',
                            df = dftest, 
                            x = [xtest[:,i,:] for i in range(xtest.shape[1])])
        except:
            print('The function prfs_and_cnfmat cannot be executed.')
            print('Metrics are not created.')

        os.system('move *' + name + '* ' + directorio)
        
    os.system('move PRFS_traval.txt ' + directorio)
    os.system('move PRFS_test.txt ' + directorio)
    os.system('copy ' + control_file + '.txt ' + directorio)
    
    if length_dict == 1:
        os.system('copy ' + dictionary + '.txt ' + directorio)
    else:
        dictionaries = dictionary.split(',')
        dictionaries = [i.strip() for i in dictionaries]
        for diccionario in dictionaries:
            os.system('copy ' + diccionario + '.txt ' + directorio)
       
    if test_with_all_false:
        test_all_false(directorio=directorio, database=database, 
                       local_function=local_function,
                       use_rad_elec=use_rad_elec, use_elecdiff=use_elecdiff,
                       use_density=use_density)

    print('Whole process lasted ', np.round(-start_main+time.time(),2),'s')                
    return 


import sys

if sys.argv[1]:
    
    f=list(filter(None,open(sys.argv[1],'r').read().split('\n')))
    f = [i for i in f if i[0] != '#']
        
    diccio={}
    for row in f:
        diccio[row.split(':')[0]] = row.split(':')[1].strip()
        
    if diccio['DATABASE'] == 'default':
        database = './support/red_cod-db.pkl'
    else:
        database = diccio['DATABASE']
    
    if diccio['FEATURES'] == 'default':
        features = './support/datosrahm.csv'
    else: 
        features = diccio['FEATURES']
        
    if diccio['LOCAL_FUNCTION'] == 'default':
        local_function = './support/fij_2.0_25_diccio'
    else:
        local_function = diccio['LOCAL_FUNCTION']
        
create_patolli(database=database, sites = int(diccio['MAX_SITIOS']), 
               elements=int(diccio['MIN_ELEMENTS']), maxatoms=int(diccio['MAX_ATOMS']),
               dictionary=diccio['DICTIONARY'], features=features,
               control_file=diccio['CONTROL_FILE'], 
               not_identified=bool(int(diccio['NOT_IDENTIFIED'])),
               verbose=int(diccio['VERBOSITY']), test_frac = float(diccio['TEST_FRAC']), 
               local_function=local_function,
               use_density=bool(int(diccio['USE_DENSITY'])), 
               use_rad_elec = bool(int(diccio['USE_RAD_ELEC'])), 
               use_elecdiff = bool(int(diccio['USE_ELECDIFF'])),
               test_with_all_false = bool(int(diccio['TEST_WITH_ALL_FALSE'])))
