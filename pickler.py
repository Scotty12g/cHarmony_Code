#!/usr/bin/env python
"""
A method to store (pickle) and load (unpickle) files
"""
import pickle

class Pickler:

    # pickle a file
    @staticmethod
    def save_pickle(obj, binhandle):
        pickle.dump(obj, binhandle)

    # load a pickeled file
    @staticmethod
    def load_pickle(binhandle):
        return pickle.load(binhandle)