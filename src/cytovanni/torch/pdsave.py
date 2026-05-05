import pandas as pd
import numpy as np

def pdsave_expand_DataFrame(df):
    return ("_pdsave_expanded_DataFrame", df.to_numpy(), df.index.to_numpy(), df.columns.to_numpy())
def pdsave_expand_Series(se):
    return ("_pdsave_expanded_Series", se.to_numpy(), se.index.to_numpy())
def pdsave_expand_Index(idx):
    return ("_pdsave_expanded_Index", idx.to_numpy())
def pdsave_expand(el):
    """ Expand pandas types into tuple of numpy arrays for storage.
        Passes through all other types.
    """
    if isinstance(el, pd.DataFrame): return pdsave_expand_DataFrame(el)
    elif isinstance(el, pd.Series):  return pdsave_expand_Series(el)
    elif isinstance(el, pd.Index):   return pdsave_expand_Index(el)
    else:                            return el
def pdsave_expand_dict(dct):
    """ Apply pdsave_expand to all elements of dct.
    """
    return {k:pdsave_expand(v) for k,v in dct.items()}


def pdsave_collect(el):
    """ Collects pandas types from tuple of numpy arrays as produced by pdsave_expand.
        Passes through all other types.
    """
    try:
        if isinstance(el, tuple):
            if el[0]=="_pdsave_expanded_DataFrame":
                return pd.DataFrame(el[1], index=el[2], columns=el[3])
            elif el[0]=="_pdsave_expanded_Series":
                return pd.Series(el[1], index=el[2])
            elif el[0]=="_pdsave_expanded_Index":
                return pd.Index(el[1])
            else:
                raise ValueError(f"Expansion {el[0]} unknown!")
        return el
    except Exception:
        return el
def pdsave_collect_dict(dct):
    """ Apply pdsave_collect to all elements of dct.
    """
    return {k:pdsave_collect(v) for k,v in dct.items()}
