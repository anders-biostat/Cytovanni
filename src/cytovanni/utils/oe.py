import numpy as np
import pandas
from sklearn.preprocessing import OrdinalEncoder

class CustomOrdinalEncoder():
    """ sklearn OrdinalEncoder, but wrapped because it expects 2D input.
        Internally converts labels to str, does not invert this after inverse_transform!
    """
    def __init__(self):
        pass

    @classmethod
    def from_labels(cls, labels, allow_unknown=False):
        """ Fit from a list of labels.
            
            :param labels: iterable. Labels to encode, can contain repetitions.
            
            :param allow_unknown: bool. Whether unknown labels should get encoded as -1 or throw an error.
        """
        oe = cls()
        unique_labels = np.unique(np.asarray(labels).astype(str))
        oe.init_oe(unique_labels, allow_unknown)
        return oe

    @classmethod
    def from_old(cls, old_labels, allow_unknown=False):
        """ Fit from list of old oe labels, preserves order.
            
            :param old_labels: iterable. Old .labels from oe, preserves their order.
            
            :param allow_unknown: bool. Whether unknown labels should get encoded as -1 or throw an error.
        """
        oe = cls()
        oe.init_oe(np.asarray(old_labels).astype(str), allow_unknown)
        return oe

    def fit_extend(self, labels):
        """ Extends by all new labels in labels, preserving encoding of the old labels.
            Returns list of all newly added ordinals.
            
            :param labels: iterable. New labels to add, can contain repetitions.
        """
        unique_labels = list(np.unique(np.asarray(labels).astype(str)))
        old_labels = list(self.labels)
        new_labels = sorted(list((set(unique_labels) | set(old_labels)) - set(old_labels)))
        self.init_oe(np.asarray(old_labels+new_labels).astype(str), allow_unknown=self.allow_unknown)
        
        return list(range(len(old_labels), len(old_labels+new_labels)))

    def init_oe(self, unique_labels, allow_unknown=False):
        """ Internal function to initialise sklearn OrdinalEncoder
        """
        self.oe = OrdinalEncoder(categories=list(np.asarray(unique_labels).reshape((1,-1))),
                                 handle_unknown= 'use_encoded_value' if allow_unknown else 'error',
                                 unknown_value= -1 if allow_unknown else None)
        self.oe.fit(np.asarray(unique_labels).reshape((-1,1)))
        self.num_labels = self.oe.categories_[0].size
        self.labels = self.oe.categories_[0]
        self.allow_unknown = allow_unknown
    
    def transform(self, labels):
        """ Transform labels into ordinal encoding.
            
            :param labels: iterable or single value. Labels to encode.
        """
        labels = np.asarray(labels)
        transformed = self.oe.transform(labels.reshape((-1,1)).astype(str)).reshape(-1).astype(int)
        return transformed if len(labels.shape)>0 else transformed[0]
    
    def inverse_transform(self, ordinals):
        """ Transform ordinals back into labels.
            Internally, all labels are string, this does not reverse this!
            Output will always be strings.
            
            :param ordinals: iterable or single value. Ordinals to decode.
        """
        ordinals = np.asarray(ordinals)
        transformed = self.oe.inverse_transform(ordinals.reshape((-1,1))).reshape(-1)
        return transformed if len(ordinals.shape)>0 else transformed[0]

    def __len__(self):
        return self.num_labels
