# Welcome to MetworkPy

![Metworkpy Logo](metworkpy_logo.png "MetworkPy Logo")

<!--toc:start-->

- [Welcome to MetworkPy](#welcome-to-metworkpy)
  - [Usage](#usage)
  - [Issues and Pull Requests](#issues-and-pull-requests)
  - [Licensing](#licensing)
  - [References](#references)
    - [iMAT References](#imat-references)
    - [Kullback-Leibler Divergence](#kullback-leibler-divergence)
    - [Mutual Information](#mutual-information)
    - [Permutation Testing](#permutation-testing)

<!--toc:end-->

MetworkPy is a Python library containing tools for working with and analyzing
metabolic networks. This functionality includes:

- Generating network representations of Genome Scale Metabolic Networks (GSMMs)
- Integrating gene expression data with GSMMs
- Evaluating where the metabolism is most perturbed using divergence metrics

## Documentation

Documentation can be found at
[https://metworkpy.readthedocs.io](https://metworkpy.readthedocs.io)

## Usage

MetworkPy can be installed with pip:

```{bash}
pip install metworkpy

```

The [documentation](https://metworkpy.readthedocs.io) includes some instructions
for getting started. For an example of the usage of MetworkPy see the
[examples directory](https://github.com/Ma-Lab-Seattle-Childrens-CGIDR/metworkpy/tree/main/examples)
. For a more advanced example associated associated with the application note
for MetworkPy (including some test data) see
[https://github.com/Ma-Lab-Seattle-Childrens-CGIDR/metworkpy_application_note](https://github.com/Ma-Lab-Seattle-Childrens-CGIDR/metworkpy_application_note).

## Issues and Pull Requests

If you experience any problems while using MetworkPy (including the
documentation), please create a GitHub issue in this repository. When creating
an issue, a minimal reproducible example of the issue will make getting you help
much easier. You can also create issues for any enhancements you would like to
see in MetworkPy. Contributions are welcome! Please see the CONTRIBUTING.md for
more information.

## Licensing

This project makes use of the following external libraries:

- [COBRApy](https://github.com/opencobra/cobrapy/tree/devel) licensed under the
  [LGPL-2.1](https://github.com/opencobra/cobrapy/blob/devel/LICENSE)
- [NetworkX](https://networkx.org/) licensed under the
  [BSD-3-Clause](https://github.com/networkx/networkx/blob/main/LICENSE.txt)
- [NumPy](https://numpy.org/) licensed under the
  [BSD-3-Clause](https://numpy.org/doc/stable/license.html)
- [Optlang](https://github.com/opencobra/optlang) licensed under
  [Apace-2.0](https://github.com/opencobra/optlang/blob/master/LICENSE)
- [Pandas](https://pandas.pydata.org/) licensed under the
  [BSD-3-Clause](https://github.com/pandas-dev/pandas/?tab=BSD-3-Clause-1-ov-file#readme)
- [SciPy](https://github.com/scipy/scipy) licensed under the
  [BSD-3-Clause](https://github.com/opencobra/cobrapy/blob/devel/LICENSE)
- [SymPy](https://www.sympy.org/en/index.html) licensed under the
  [BSD-3-Clause](https://github.com/sympy/sympy/blob/master/LICENSE)

The mutual information implementation where partially inspired by those found in
the `feature_selection` module of
[scikit-learn](https://github.com/scikit-learn/scikit-learn?tab=readme-ov-file),
and the tests for those methods were adapted from those in scikit-learn, which
is licensed under the
[BSD-3-Clause](https://github.com/scikit-learn/scikit-learn?tab=BSD-3-Clause-1-ov-file).
Additionally, the implementation of the iMAT functionality was inspired by
[gembox](https://github.com/ruppinlab/gembox) (which uses a
[GPL-3.0-only](https://github.com/ruppinlab/gembox?tab=GPL-3.0-1-ov-file)
license), and [dexom-python](https://github.com/MetExplore/dexom-python) (which
uses the
[GPL-3.0-only](https://github.com/MetExplore/dexom-python?tab=GPL-3.0-1-ov-file)
license).

The permutation test implementation uses modified code from SciPy's stats module
(licensed under the BSD-3-Clause, see above) for estimating the empirical
p-value in line with
[Phipson, B., & Smyth, G. K. (2010)](https://arxiv.org/abs/1603.05766).

## References

### iMAT References

1. [Shlomi T, et al. Network-based prediction of human tissue-specific metabolism, Nat. Biotechnol., 2008, vol. 26 (pg. 1003-1010)](https://www.nature.com/articles/nbt.1487)
1. Hadas Zur, Eytan Ruppin, Tomer Shlomi, iMAT: an integrative metabolic
   analysis tool, Bioinformatics, Volume 26, Issue 24, December 2010, Pages
   3140–3142,
   [https://doi.org/10.1093/bioinformatics/btq602](https://academic.oup.com/bioinformatics/article/26/24/3140/290045?login=true)

### Kullback-Leibler Divergence

1. [Q. Wang, S. R. Kulkarni and S. Verdu, "Divergence Estimation for Multidimensional Densities Via k-Nearest-Neighbor Distances," in IEEE Transactions on Information Theory, vol. 55, no. 5, pp. 2392-2405, May 2009, doi: 10.1109/TIT.2009.2016060.](https://ieeexplore.ieee.org/document/4839047)

### Mutual Information

1. [Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. Physical Review E, 69(6), 066138.](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138)
1. [Ross, B. C. (2014). Mutual Information between Discrete and Continuous Data Sets. PLoS ONE, 9(2), e87357](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357)

### Permutation Testing

1. [Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero: Calculating exact p-values when permutations are randomly drawn. Statistical Applications in Genetics and Molecular Biology, 9(1). https://doi.org/10.2202/1544-6115.1585](https://arxiv.org/abs/1603.05766)
