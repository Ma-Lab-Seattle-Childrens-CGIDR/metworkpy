"""
Submodule implementing methods for performing Corner Based Sampling
"""

# Standard Library Imports
from typing import Any, Optional, Union

# External Imports
import cobra
from joblib import Parallel, delayed
import numpy as np
import pandas as pd


def corner_sampling(
    model: cobra.Model,
    n_samples: int = 1_000,
    reaction_list: Optional[list[str]] = None,
    processes: Optional[int] = None,
    fva_scale: bool = False,
    seed: Optional[Union[int, np.random.Generator]] = None,
    fva_kwargs: Optional[dict[str, Any]] = None,
):
    """
    Perform Corner Based sampling of a Metabolic Model

    Parameters
    ----------
    model : cobra.Model
        The model to sample from
    n_samples : int, default=1000
        The number of samples to generate
    reaction_list : list[str], optional
        The set of reactions which could be selected
        to be a part of the objective during corner
        sampling (so, for example, you could remove pseudo reactions).
        Must be a list of reaction ids.
    processes : int, optional
        The number of processes to use (note uses joblib,
        so can be managed via a joblib context)
    fva_scale : bool, default=False
        Whether to scale the weights assigned to each objective
        value by the maximum flux value it could achieve
    seed : int or np.random.Generator, optional
        Optional seed to use for selection of reactions/weights.
        Note that this doesn't garuntee the generated solutions will
        be the same, only that the objectives selected to generate
        each will be (so it will depend on solver
        consistancy if the samples are identical).
    fva_kwargs : dict of str to Any
        Key word arguments passed to
        `cobra.flux_analysis.flux_variability_analysis <https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/variability/index.html#cobra.flux_analysis.variability.flux_variability_analysis>`_



    Notes
    -----
    Corner Based sampling iteratively creates a random objective
    function, and then optimizes it, storing the resulting flux
    distribution. It creates a random objective function by first selecting
    a value $\tau$, which is the proportion of reactions that will be involved.
    Then, for it chooses a subset of reaction based on this proportion,
    and for each assigns a random weight between -1 and 1. These weights
    can be optionally scaled using flux variability analysis.
    The objective of the FBA problem is them set to be the weighted sum of the
    fluxes of the selected reactions (weighted by the randomly generated weights).

    This method is based on the method discussed in "Adjusting for false discoveries
    in constraint-based differential metabolic flux analysis", by Bruno G. Galuzzi,
    Luca Milazzo, and Chiara Damiani.

    References
    ----------
    #. Galuzzi, B. G., Milazzo, L., & Damiani, C. (2024). Adjusting for false
       discoveries in constraint-based differential metabolic flux analysis.
       Journal of Biomedical Informatics, 150, 104597. https://doi.org/10.1016/j.jbi.2024.104597
    """
    # Seed needs to not be none, so that it can be
    # passed with the id to generate reproducible results
    if seed is None:
        rng = np.random.default_rng()
        seed = rng.integers(0, np.iinfo(np.int_).max)
    if isinstance(seed, np.random.Generator):
        seed = seed.integers(0, np.iinfo(np.int_).max)
    model_ = (
        model.copy()
    )  # Copy model to ensure we aren't going to change original model
    if reaction_list is None:
        reaction_list = model_.reactions.list_attr("id")
    if fva_scale:
        if fva_kwargs is None:
            fva_kwargs = {}
        fva_res = cobra.flux_analysis.flux_variability_analysis(
            model_,
            reaction_list=reaction_list,  # type:ignore
            **fva_kwargs,
        )
        fva_max = fva_res.abs().max(axis=1)
    else:
        fva_max = None
    sample_df = pd.DataFrame(
        np.nan,
        index=pd.Index(range(n_samples)),
        columns=model_.reactions.list_attr("id"),
    )
    for idx, sample in enumerate(
        Parallel(n_jobs=processes, return_as="generator")(
            delayed(_corner_sampling_worker)(
                model=model_,
                reaction_list=reaction_list,
                fva_max=fva_max,
                fva_scale=fva_scale,
                seed=[i, seed],
            )
            for i in range(n_samples)
        )
    ):
        sample_df.loc[idx] = sample
    return sample_df


def _corner_sampling_worker(
    model: cobra.Model,  # Model to sample from
    reaction_list: list[str],  # List of reactions to consider for objective
    fva_max: Optional[
        pd.Series
    ],  # Maximum fva to divide weights by (should be max absolute?)
    fva_scale: bool,
    seed: list[int],  # List so that the seed can include the worker ID
) -> Optional[pd.Series]:
    # Create an RNG from the seed
    rng = np.random.default_rng(seed)
    # Decide how many reactions from the reaction list to consider
    num_reactions = int(
        rng.random() * len(reaction_list)
    )  # Draw proportion from [0,1), get number of reactions instead of proportion, truncate
    # Select reactions
    objective_dict = {}
    for rxn in rng.choice(reaction_list, num_reactions, replace=False):
        weight = rng.random() * 2 - 1
        if fva_scale:
            assert fva_max is not None
            weight /= fva_max[rxn]
        objective_dict[model.reactions.get_by_id(rxn)] = weight
    with model as m:
        m.objective = objective_dict
        m.objective_direction = rng.choice(["max", "min"], 1, replace=False)[0]
        try:
            fluxes = m.optimize().fluxes
        except Exception:
            return None
    return fluxes
