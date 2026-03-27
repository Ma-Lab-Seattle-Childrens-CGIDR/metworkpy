# MetworkPy Examples

<!--toc:start-->

- [MetworkPy Examples](#metworkpy-examples)
  - [Dependencies](#dependencies)
    - [UV (Recommended)](#uv-recommended)
    - [Venv](#venv)
    - [Conda](#conda)
  - [References](#references)
    - [Data](#data)
    - [Genome Scale Metabolic Model](#genome-scale-metabolic-model)

<!--toc:end-->

## Dependencies

Required dependencies for the examples can be installed in several ways:

### UV (Recommended)

First install [uv](https://docs.astral.sh/uv/getting-started/installation/),
then a virtual environment can be set up using

```{bash}
uv sync
```

then the example notebooks can be opened with

```{bash}
uv run jupyter notebook path-to-notebook
```

### Venv

Ensure that python is installed, then create a virtual environment

```{bash}
python -m venv .venv
```

activate the virtual environment

```{bash}
# On MacOS and Linux
source .venv/scripts/activate
# On windows
.venv/Scripts/activate
```

and install the required dependencies

```bash
pip install -r requirements.txt
```

finally the jupyter notebook can be opened

```bash
jupyter notebook path-to-notebook
# Or
jupyter lab path-to-notebook
```

### Conda

Conda can be installed using [Anaconda](https://www.anaconda.com/download),
[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main), or
[Miniforge](https://github.com/conda-forge/miniforge). We recommend using
Miniforge as that is what this repository was tested with, but any method should
work.

After installing conda using one of the above methods, then the environment can
be created using the environment.yml file:

```{bash}
conda create --file environment.yml
```

and then activated with

```{bash}
conda activate metworkpy
```

finally the jupyter notebook can be opened

```bash
jupyter notebook path-to-notebook
# Or
jupyter lab path-to-notebook
```

## References

### Data

- 13059_2014_502_MOESM1_ESM.xlsx: Rustad, T. R., Minch, K. J., Ma, S., Winkler,
  J. K., Hobbs, S., Hickey, M., Brabant, W., Turkarslan, S., Price, N. D.,
  Baliga, N. S., & Sherman, D. R. (2014). Mapping and manipulating the
  Mycobacterium tuberculosis transcriptome using a transcription factor
  overexpression-derived regulatory network. Genome Biology, 15(11), 502.
  [https://doi.org/10.1186/s13059-014-0502-3](https://doi.org/10.1186/s13059-014-0502-3)

### Genome Scale Metabolic Model

- iEK1011_v2_7H9_ADC_glycerol.json: López-Agudelo, V. A., Mendum, T. A., Laing,
  E., Wu, H., Baena, A., Barrera, L. F., Beste, D. J. V., & Rios-Estepa, R.
  (2020). A systematic evaluation of Mycobacterium tuberculosis Genome-Scale
  Metabolic Networks. PLoS Computational Biology, 16(6), e1007533.
  [https://doi.org/10.1371/journal.pcbi.1007533](https://doi.org/10.1371/journal.pcbi.1007533)
