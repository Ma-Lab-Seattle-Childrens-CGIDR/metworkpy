# Changelog

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
