"""
Module to check if we are running in a notebook
"""


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # ty: ignore[unresolved-reference]
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
