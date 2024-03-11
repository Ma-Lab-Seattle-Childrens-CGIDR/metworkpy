# Welcome to Metworkpy

Python library for working with metabolic networks in python, focused on generating and working with
graphs based on Genome Scale Metabolic Models. These graphs include ones representing the reactions
found within the model, as well as those based on information metrics. Additionally, in order to
integrate gene expression data into the above methods, several gene expression integration 
algorithms are included, such as IMAT, GIMME, MOMA, and Metchange. 

# Licensing
This project makes use of the following external libraries:
 - [COBRApy](https://github.com/opencobra/cobrapy/tree/devel) licensed 
    under the [LGPL-2.1](https://github.com/opencobra/cobrapy/blob/devel/LICENSE)
 - [NetworkX](https://networkx.org/) licensed under the [BSD-3-Clause](https://github.com/networkx/networkx/blob/main/LICENSE.txt)
 - [NumPy](https://numpy.org/) licensed under the
    [BSD-3-Clause](https://numpy.org/doc/stable/license.html)
 - [optlang](https://github.com/opencobra/optlang) licensed under 
    [Apace-2.0](https://github.com/opencobra/optlang/blob/master/LICENSE)
 - [Pandas](https://pandas.pydata.org/) licensed under the [BSD-3-Clause](https://github.com/pandas-dev/pandas/?tab=BSD-3-Clause-1-ov-file#readme)
 - [SciPy](https://github.com/scipy/scipy) licensed under the 
    [BSD-3-Clause](https://github.com/opencobra/cobrapy/blob/devel/LICENSE)
 - [SymPy](https://www.sympy.org/en/index.html) licensed under the [BSD-3-Clause](https://github.com/sympy/sympy/blob/master/LICENSE)

The mutual information implementation where partially inspired by those found in the 
`feature_selection` module of [scikit-learn](https://github.com/scikit-learn/scikit-learn?tab=readme-ov-file), and the tests for those methods 
were adapted from those in scikit-learn, which is licensed under the [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn?tab=BSD-3-Clause-1-ov-file)

# References:
## Mutual Information:

1. [Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. 
   Physical Review E, 69(6), 066138.](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138)
2. [Ross, B. C. (2014). Mutual Information between Discrete and Continuous 
   Data Sets. PLoS ONE, 9(2), e87357](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357)

## IMAT References:
1. [Shlomi T, et al. Network-based prediction of human tissue-specific 
        metabolism, Nat. Biotechnol., 2008, vol. 26 (pg. 1003-1010)](https://www.nature.com/articles/nbt.1487)