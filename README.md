# patolli_2021
This is an updated version of patolli, that enables to make multiclass classification of crystal compounds.

patolli was a typical game in ancient Mexico. 
Due to its predictive attributes on the fate of the people, patolli had a ritual character.
With patolli.py you can create your own patollis, which will help you to infer new materials with a crystal structure.

The way the patollis are created is with Artificial Neuronal Networks (ANNs). 

patolli.py needs the next Python modules:
<ul>
  <li>keras</li>
  <li>scikit-learn</li>
  <li>numpy</li>
  <li>pandas</li>
  <li>matplotlib</li>  
  <li>itertools</li>
  <li>copy</li>
  <li>time</li>
  <li>os</li>
</ul>

In your conda environment, pydot or graphviz should be installed to diplay some items related to the library Keras. 
Patolli was developed in Python 3.6. The library Keras 2.2.4 is required to satisfactorily execute patolli.py. 
The Keras backed which have been proven to work are Theano (1.0.3), and Tensorflow (1.14.0).

To run patolli, you need to enter the next command in the shell:

$python patolli.py patolli.inp

The text file patolli.inp calls other text files, which contain the characteristics of the ANNs to train (model_control_file), as well as the dictionaries of the crystal structures to use in the development of the ANNs. These crystal structure dictionaries are in the directory structures_dictionaries. It is recommended to place all txt-files of that directory here.

You should not remove the files within the directory 'support'. Otherwise, patolli.py crashes.

If you benefit of this code, I appreciate that you cite the next article:
<ul>
  <li> J. I. Gómez-Peralta, X. Bokhimi. Journal of Solid State Chemistry, Vol. 285 (2020) 121253. </li>
</ul>
