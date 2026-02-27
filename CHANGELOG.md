# Changelog

<!--toc:start-->

- [Changelog](#changelog)
  - [Version DEV](#version-dev)
  - [Version 0.5.0](#version-050)
  - [Version 0.4.1](#version-041)
  - [Version 0.4.0](#version-040)
  - [Version 0.3.0](#version-030)
  - [Version 0.2.9](#version-029)
  - [Version 0.2.8](#version-028)
  - [Version 0.2.7](#version-027)

<!--toc:end-->

## Version DEV

- Updated GPR parsing to directly use the ASTs rather than reparsing the strings
  (which should bring a significant performance improvement)
- Adding essentiality parameters to several of the density functions, allowing
  users to select whether translating from reactions to genes should require
  that the genes are required for the reaction or not
- Added fuzzy module for network with methods for converting sets of genes into
  fuzzy reaction sets with configurable membership functions.
- Added method for fuzzy-intersection to fuzzy sub-module. Added
  robustrankaggregpy as a dependency so that RRA can be used as an intersection
  method.
- Added methods to network construction for creating reaction and metabolic
  networks more directly

## Version 0.5.0

- Updated empirical estimation method in the permutation test function to
  account for underestimation of p-values as discussed in
  [Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn](https://arxiv.org/abs/1603.05766)

## Version 0.4.1

- Fixing issues with scipy permutation test's handling of multi-dimensional
  observations
  - Created new permutation test function which handles mutli-dimensional
    observations in a way that is compatable for the mutual information and
    divergence functions to be able to calculate p-values
  - Updated mutual information and divergence functions to use this new
    permutation test function

## Version 0.4.0

- Refactored divergence array code to use joblib, and enable calculating
  p-value. Additionally, the divergence array functions in KL and JS modules can
  now take an axis argument specifying which axis to slice along.
- Refactored grouped divergence and KO divergence to allow for calculating
  significance using permutation testing. Additionally, refactored ko_divergence
  to use the grouped divergence function instead of repeating its functionality,
  with the added benefit of allowing the divergence of the different groups to
  be calculated in parallel. Reworked the API to only accept kl/js literals for
  'divergence_metric', also renamed "divergence_metric" to "divergence_type" to
  match grouped divergence function and clarify the purpose. Also, now pass
  kwargs to the underlying divergence function. Moving the keyword args passed
  to the sampler function into their own dict and a seperate parameter (though
  the common ones should still be handled in a way to make this relatively
  transparent to the user, but this is breaking).

## Version 0.3.0

- Removed deprecated path function of importlib from example submodule
- Changed network density functions to use joblib for parallel processing
- Made gene list parameter optional in KO divergence function
- Added cutoff parameter for mutual information network calculations
- Added option to calculate p-value using a permutation test for the mutual
  information function (uses scipy.permutation test)
- Added option for calculating the p-values to the mi_pairwise function as well,
  and to threshold the mi_pairwise return value using a set cutoff, a quantile
  cutoff, or a significance cutoff
- Divergence functions can now return p-values using permutations tests
- Significant refactor of the network creation API, done to simplify the code
  and try to fix some issues. Only user facing API change should be that now the
  adjacency matrix is always a pandas dataframe.

## Version 0.2.9

- Reverting breaking change in previous version, adding flag to modify behaviour
  of the KO divergence function instead.

## Version 0.2.8

- Changed Handling of Perturbed vs Unperturbed samples in KO-divergence
  functions. Now the perturbed is treated as p (the true distribution), and the
  unperturbed is treated as q (the approximating distribution).
- Added included example model (ability to easily import an example model
  without worrying about filepaths). This will be mainly used for documentation
  examples in the future.
- Changes to developement tooling (moved to tox for running tests across
  multiple versions, uv lock updates, etc.).

## Version 0.2.7

- Added method for computing enrichment in target genes in neighborhoods around
  each reaction in a reaction network (expanding on the previous density finding
  functions)
- Allowed for threshold in IMAT functinality to be exactly 0 (prevously was
  required to be greater than 0)
