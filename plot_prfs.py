# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:14:43 2021

@author: User
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as PRFS

import numpy as np
import pandas as pd
import keras.models as models
import matplotlib.pyplot as plt
import itertools
import os

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


def prfs_and_cnfmat(directorio=''):
    
    modelos = "ls " + directorio + "/*.h5 > model_names.txt"
    os.system(modelos)
    
    dbtraval = pd.read_csv(directorio + '/dbtraval.csv')
    dbtest = pd.read_csv(directorio + '/dbtest.csv')
    
    xtraval = np.load(directorio + '/Xtraval.npy')
    xtest = np.load(directorio + '/Xtest.npy')
    
    diccio = np.load(directorio + '/feature_standarisation.npy').item()
    
    xtraval = (xtraval - diccio['mean'])/diccio['std']
    xtest = (xtest - diccio['mean'])/diccio['std']
    
    with open('model_names.txt','r') as f:
        for line in f:
            modelo = models.load_model(line[:len(line)-1])
            output_nodes = modelo.output_shape[1]
            name = line.split('/')[1].split('.')[0]
            class_names = [i for i in range(output_nodes)]
            
            for x, df, saveas in zip((xtraval,xtest),(dbtraval,dbtest),('traval','test')):
                predictions = modelo.predict(x[:,0,:])
                y_true = df['target']
                
                if output_nodes != 1:
                    y_pred = np.argmax(predictions,axis=1)
                else:
                    y_pred = (predictions >= 0.5)
                print(PRFS(y_true, y_pred))   
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