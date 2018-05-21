'''
This module contains a second generation of 
experiment classes. Rather than classes that
have various feature selection functions, I will be
using Text-Fabric search templates for data selection.
These templates will be linked to tokenizers which will
convert the coordinated data into token strings for counting.

This method has many advantages over the first-gen, class-only
format. With the old format, I gained extensibility
at the expense of readability and clarity. In the new method,
we maintain extensibility while maximizing clarity. This includes:

√ selection parameters for bases and targets clearly described: 
    templates are very easy to read/edit compared with code.
√ individual clause elements chosen separately but can be 
    united in a dictionary keyed by the clause; 
    good for both frame spaces and individual feature spaces
√ all parameters coordinated at once in one readable space (template)
    to avoid having to reclarify multiple parameters in multiple
    places
'''

import collections
import numpy as np
import pandas as pd

class Experiment:
    
    def __init__(self, parameters, tf=None, min_observation=10):
        '''
        Parameters is a tuple of tuples.
        Tuples consist of:
        
        (template, template_kwargs, target_i, 
        (bases_i,), target_tokenizer, basis_tokenizer)

        template - a TF search template (string)
        template_kwargs - dict of keyword arguments for formatting the template (if any)
        search_filter - list comprehension to filter out certain search results
        target_i - the index of the target word
        bases_i - tuple of basis indexes
        target_tokenizer - a function to construct target tokens, requires target_i
        basis_tokenizer - a function to construct basis tokens, requires basis_i(s)
        '''
        
        self.min_obs = min_observation # minimum observation requirement
         
        # Text-Fabric method short forms
        F, E, T, L, S = tf.F, tf.E, tf.T, tf.L, tf.S
        self.tf_api = tf
        
        # raw experiment_data[target_token][clause][list_of_bases_tokens]
        experiment_data = collections.defaultdict(lambda: collections.defaultdict(list))
        
        # helper data, for SemSpace class
        self.target2gloss = dict()
        self.target2lex = dict()
        self.target2node = dict()

        for templ, filt, target_i, bases_i, target_tokener, basis_tokener in parameters:

            # run search query on template
            search_template = templ.format(**templ_kw)
            sample = sorted(S.search(search_template))
            sample = filt(sample) if filt else sample # filter results for not-exist type queries

            # make target token
            for specimen in sample:
                clause = specimen[0]
                target = specimen[target_i]
                target_token = target_tokener(target)

                # make bases tokens, map to clause
                for basis_i in bases_i:
                    basis = specimen[basis_i]
                    basis_token = basis_tokener(basis, target)
                    experiment_data[target_token][clause].append(basis_token)

                # add helper data
                self.target2gloss[target_token] = F.gloss.v(L.u(target, 'lex')[0])
                self.target2lex[target_token] = L.u(target, 'lex')[0]
                self.target2node[target_token] = target
                
        # finalize data
        self.count_experiment(experiment_data)
    
    
    def count_experiment(self, experiment_data):
        '''
        Counts experiment data into a dataframe from an
        experiment data dictionary structured as:
        experiment_data[target_token][clause][list_of_bases_tokens]
        
        --input--
        dict
        
        --output--
        pandas df
        '''
        
        ecounts = collections.defaultdict(lambda: collections.Counter())
        
        for target, clauses in experiment_data.items():
            for clause, bases in clauses.items():
                ecounts[target].update(bases)
                
        counts = dict((target, counts) for target, counts in ecounts.items()
                                if sum(counts.values()) >= self.min_obs)
        
        self.data = pd.DataFrame(counts).fillna(0)
        self.raw_data = experiment_data
        
        
class ExperimentFrame(Experiment):
    
    '''
    Identical to Experiment except
    it builds frames as basis elements.
    '''
    
    def __init__(self, parameters, tf=None, min_observation=10):
        super().__init__(parameters, tf=tf, min_observation=min_observation)
        
    def count_experiment(self, experiment_data):
        '''
        Counts experiment data into a dataframe from an
        experiment data dictionary structured as:
        experiment_data[target_token][clause][list_of_bases_tokens]
        
        *special*: Counts clauses as a single frame.
        
        --input--
        dict
        
        --output--
        pandas df
        '''
        
        ecounts = collections.defaultdict(lambda: collections.Counter())
        
        for target, clauses in experiment_data.items():
            for clause, bases in clauses.items():
                frame = '|'.join(sorted(bases))
                ecounts[target][frame] += 1
                
        counts = dict((target, counts) for target, counts in ecounts.items()
                                if sum(counts.values()) >= self.min_obs)
        
        self.data = pd.DataFrame(counts).fillna(0)
        self.raw_data = experiment_data