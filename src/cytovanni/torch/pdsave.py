"""Store pandas objects as plain tuples of numpy arrays (for ``torch.save`` of exported models) and restore them.

`pdsave_expand` turns a DataFrame, Series or Index into a tagged tuple, `pdsave_collect` rebuilds it. Only the values
and the index and column labels are kept: index/column names are lost, and a DataFrame with mixed column dtypes
comes back with dtype object.
"""
import pandas as pd
import numpy as np

def pdsave_expand_DataFrame(df):
    """Expand a DataFrame to ``("_pdsave_expanded_DataFrame", values, index, columns)``."""
    return ("_pdsave_expanded_DataFrame", df.to_numpy(), df.index.to_numpy(), df.columns.to_numpy())
def pdsave_expand_Series(se):
    """Expand a Series to ``("_pdsave_expanded_Series", values, index)``."""
    return ("_pdsave_expanded_Series", se.to_numpy(), se.index.to_numpy())
def pdsave_expand_Index(idx):
    """Expand an Index to ``("_pdsave_expanded_Index", values)``."""
    return ("_pdsave_expanded_Index", idx.to_numpy())
def pdsave_expand(el):
    """Expand a pandas object into a tagged tuple of numpy arrays for storage.

    Parameters
    ----------
    el : object
        A pandas.DataFrame, pandas.Series or pandas.Index; anything else is returned unchanged.

    Returns
    -------
    tuple or object
        ``(tag, values, index[, columns])`` with numpy arrays, or `el` itself if it is not a pandas type.

    See Also
    --------
    pdsave_collect : the inverse.
    """
    if isinstance(el, pd.DataFrame): return pdsave_expand_DataFrame(el)
    elif isinstance(el, pd.Series):  return pdsave_expand_Series(el)
    elif isinstance(el, pd.Index):   return pdsave_expand_Index(el)
    else:                            return el
def pdsave_expand_dict(dct):
    """Apply `pdsave_expand` to every value of a dict.

    Parameters
    ----------
    dct : dict
        Dict whose values may be pandas objects (not expanded recursively).

    Returns
    -------
    dict
        New dict with the same keys and expanded values.
    """
    return {k:pdsave_expand(v) for k,v in dct.items()}


def pdsave_collect(el):
    """Rebuild a pandas object from a tuple produced by `pdsave_expand`.

    Parameters
    ----------
    el : object
        A tuple whose first element is a ``"_pdsave_expanded_..."`` tag; anything else (including other tuples) is
        returned unchanged.

    Returns
    -------
    pandas.DataFrame, pandas.Series, pandas.Index or object
        The rebuilt pandas object, or `el` itself.

    Raises
    ------
    ValueError
        If the tag starts with ``"_pdsave_expanded_"`` but names no known type. Errors while rebuilding (e.g. an
        index of the wrong length) are raised by pandas.
    """
    if isinstance(el, tuple) and len(el) > 0 and isinstance(el[0], str) and el[0].startswith("_pdsave_expanded_"):
        if el[0]=="_pdsave_expanded_DataFrame":
            return pd.DataFrame(el[1], index=el[2], columns=el[3])
        elif el[0]=="_pdsave_expanded_Series":
            return pd.Series(el[1], index=el[2])
        elif el[0]=="_pdsave_expanded_Index":
            return pd.Index(el[1])
        else:
            raise ValueError(f"Expansion {el[0]} unknown!")
    return el
def pdsave_collect_dict(dct):
    """Apply `pdsave_collect` to every value of a dict.

    Parameters
    ----------
    dct : dict
        Dict as produced by `pdsave_expand_dict`.

    Returns
    -------
    dict
        New dict with the same keys and rebuilt pandas objects.
    """
    return {k:pdsave_collect(v) for k,v in dct.items()}
