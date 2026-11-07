"""Ordinal encoding of labels (e.g. batch names) to integers, with stable, extendable codes."""
import numpy as np
import pandas
from sklearn.preprocessing import OrdinalEncoder

class CustomOrdinalEncoder():
    """Map labels to consecutive integers 0..N-1 and back.

    Wraps ``sklearn.preprocessing.OrdinalEncoder`` for 1-D input (and single values). All labels are converted to
    ``str`` internally, so ``1`` and ``"1"`` are the same label, and `inverse_transform` always returns strings.
    Create it with `from_labels` (codes in sorted label order) or `from_old` (codes in a given order); `fit_extend`
    adds labels without changing existing codes.

    Attributes
    ----------
    labels : numpy.ndarray of str
        Known labels; the code of a label is its position in this array.
    num_labels : int
        Number of known labels (also ``len(encoder)``).
    allow_unknown : bool
        Whether unknown labels are encoded as -1 instead of raising.
    oe : sklearn.preprocessing.OrdinalEncoder
        The wrapped encoder.

    Examples
    --------
    >>> oe = CustomOrdinalEncoder.from_labels(["b", "a", "b"])
    >>> oe.transform(["a", "b"])
    array([0, 1])
    >>> oe.fit_extend(["c"])
    [2]
    """
    def __init__(self):
        pass

    @classmethod
    def from_labels(cls, labels, allow_unknown=False):
        """Create an encoder for the unique labels in `labels`, coded in sorted (string) order.

        Parameters
        ----------
        labels : array-like
            Labels to encode, may contain repetitions; converted to str before sorting (so ``10`` sorts before
            ``3``).
        allow_unknown : bool, default False
            If True, `transform` encodes unknown labels as -1; if False, it raises ``ValueError`` for them.

        Returns
        -------
        CustomOrdinalEncoder
        """
        oe = cls()
        unique_labels = np.unique(np.asarray(labels).astype(str))
        oe.init_oe(unique_labels, allow_unknown)
        return oe

    @classmethod
    def from_old(cls, old_labels, allow_unknown=False):
        """Create an encoder that keeps a given label order, e.g. the `labels` of an existing encoder.

        Parameters
        ----------
        old_labels : array-like
            Unique labels in the order of their codes (label ``old_labels[i]`` gets code ``i``); converted to str.
        allow_unknown : bool, default False
            If True, `transform` encodes unknown labels as -1; if False, it raises ``ValueError`` for them.

        Returns
        -------
        CustomOrdinalEncoder
        """
        oe = cls()
        oe.init_oe(np.asarray(old_labels).astype(str), allow_unknown)
        return oe

    def fit_extend(self, labels):
        """Add all labels of `labels` that are not known yet, keeping the codes of the known labels.

        New labels get the next free codes, in sorted (string) order. Modifies the encoder in place.

        Parameters
        ----------
        labels : array-like
            Labels to add, may contain repetitions and already known labels.

        Returns
        -------
        list of int
            Codes of the newly added labels (empty if all were known).
        """
        unique_labels = list(np.unique(np.asarray(labels).astype(str)))
        old_labels = list(self.labels)
        new_labels = sorted(list((set(unique_labels) | set(old_labels)) - set(old_labels)))
        self.init_oe(np.asarray(old_labels+new_labels).astype(str), allow_unknown=self.allow_unknown)

        return list(range(len(old_labels), len(old_labels+new_labels)))

    def init_oe(self, unique_labels, allow_unknown=False):
        """(Re)initialise the wrapped sklearn encoder with `unique_labels` in the given order.

        Used by the constructors and `fit_extend`; sets `oe`, `labels`, `num_labels` and `allow_unknown`.

        Parameters
        ----------
        unique_labels : array-like of str
            Unique labels in code order.
        allow_unknown : bool, default False
            Encode unknown labels as -1 instead of raising.
        """
        self.oe = OrdinalEncoder(categories=list(np.asarray(unique_labels).reshape((1,-1))),
                                 handle_unknown= 'use_encoded_value' if allow_unknown else 'error',
                                 unknown_value= -1 if allow_unknown else None)
        self.oe.fit(np.asarray(unique_labels).reshape((-1,1)))
        self.num_labels = self.oe.categories_[0].size
        self.labels = self.oe.categories_[0]
        self.allow_unknown = allow_unknown

    def transform(self, labels):
        """Encode labels as integer codes.

        Parameters
        ----------
        labels : array-like or scalar
            Labels to encode (converted to str).

        Returns
        -------
        numpy.ndarray of int or int
            Codes, shape (labels,) for array-like input, a single integer for a scalar. Unknown labels give -1 if
            `allow_unknown`.

        Raises
        ------
        ValueError
            From sklearn, if a label is unknown and not `allow_unknown`.
        """
        labels = np.asarray(labels)
        transformed = self.oe.transform(labels.reshape((-1,1)).astype(str)).reshape(-1).astype(int)
        return transformed if len(labels.shape)>0 else transformed[0]

    def inverse_transform(self, ordinals):
        """Decode integer codes back into labels.

        Labels are always returned as strings (the encoder stores them as str), also if they were numbers originally.

        Parameters
        ----------
        ordinals : array-like of int or int
            Codes to decode. -1 (unknown) decodes to None if `allow_unknown`.

        Returns
        -------
        numpy.ndarray of str or str
            Labels, shape (ordinals,) for array-like input, a single label for a scalar.
        """
        ordinals = np.asarray(ordinals)
        transformed = self.oe.inverse_transform(ordinals.reshape((-1,1))).reshape(-1)
        return transformed if len(ordinals.shape)>0 else transformed[0]

    def __len__(self):
        return self.num_labels
