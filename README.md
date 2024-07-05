# Welcome to Metworkpy
![Metworkpy Logo](metworkpy_logo.png "Metworkpy Logo")

Metworkpy is a Python library containing tools for working with and analyzing metabolic networks.
This functionality includes:
 - Generating network representations of Genome Scale Metabolic Networks (GSMMs)
 - Integrating gene expression data with GSMMs
 - Evaluating where the metabolism is most perturbed using divergence metrics

# Issues and Pull Requests
If you experience any problems while using Metworkpy (including the documentation), please
create a GitHub issue in this repository. When creating an issue, a minimal reproducible example of the issue will make getting you help much easier.
You can also create issues for any enhancements you would like to see in Metworkpy.
Contributions are welcome! Feel free to open a pull request. Currently, the contribution
guidelines are still being worked out, but for enhanced functionality, please include an
explanation of the functionality, any needed citations, and test cases (tests are run using
pytest during continuous integration).

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

## IMAT References:
1. [Shlomi T, et al. Network-based prediction of human tissue-specific
        metabolism, Nat. Biotechnol., 2008, vol. 26 (pg. 1003-1010)](https://www.nature.com/articles/nbt.1487)

## Kulback-Leibler Divergence:
1. [Q. Wang, S. R. Kulkarni and S. Verdu, "Divergence Estimation for Multidimensional
   Densities Via k-Nearest-Neighbor Distances," in IEEE Transactions on Information Theory,
   vol. 55, no. 5, pp. 2392-2405, May 2009, doi: 10.1109/TIT.2009.2016060.](https://ieeexplore.ieee.org/document/4839047)

## Mutual Information:

1. [Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information.
   Physical Review E, 69(6), 066138.](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138)
2. [Ross, B. C. (2014). Mutual Information between Discrete and Continuous
   Data Sets. PLoS ONE, 9(2), e87357](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357)
