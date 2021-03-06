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
    
    def __init__(self, parameters, tf=None, min_observation=10, frame=False):
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
        count_instances - boolean on whether to collapse instances of a basis element at the result level
        '''
        
        self.min_obs = min_observation # minimum observation requirement
        
        # Text-Fabric method short forms
        F, E, T, L, S = tf.F, tf.E, tf.T, tf.L, tf.S
        self.tf_api = tf
        
        # raw experiment_data[target_token][clause][list_of_bases_tokens]
        experiment_data = collections.defaultdict(lambda: collections.defaultdict(list)) if not frame else\
                          collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
        self.collapse_instances = False # count presence/absence of features in a clause, i.e. don't add up multiple instances
        count_experiment = self.inventory_count if not frame else self.frame_count
        
        # helper data, for mappings between bases and TF data
        self.target2gloss = dict()
        self.target2lex = dict()
        self.target2node = dict()
        self.clause2basis2result = collections.defaultdict(lambda: collections.defaultdict(list))

        for search_templ, filt_param, target_i, bases_i, target_tokener, basis_tokener, count_inst in parameters:
            
            # run search query on template
            filt = filt_param if type(filt_param) != tuple else filt_param[0] # configure filter function
            search_sets = None if type(filt_param) != tuple else filt_param[1] # configure search sets
            sample = sorted(S.search(search_templ, sets=search_sets))
            sample = filt(sample) if filt else sample # filter results for not-exist type queries

            # make target token
            for specimen in sample:
                clause = specimen[0] if F.otype.v(specimen[0]) == 'clause'\
                             else next(r for r in specimen if F.otype.v(r) == 'clause')
                target = specimen[target_i]
                target_token = target_tokener(target)

                # make bases tokens, map to clause
                for basis_i in bases_i:
                    basis = specimen[basis_i]
                    basis_token = basis_tokener(basis, target)
                    basis_tokens = (basis_token,) if type(basis_token) == str else basis_token
                    if not frame:
                        experiment_data[target_token][clause].extend(basis_tokens)
                    else:
                        basis_unit = L.u(basis, 'phrase') if F.otype.v(basis) == 'word' else basis
                        experiment_data[target_token][clause][basis_unit].extend(basis_tokens)
                    
                    # add helper data 1, basis to results mapping
                    for bt in basis_tokens:
                        self.clause2basis2result[clause][bt].append(specimen)

                # add helper data 2
                self.target2gloss[target_token] = F.gloss.v(L.u(target, 'lex')[0])
                self.target2lex[target_token] = L.u(target, 'lex')[0]
                self.target2node[target_token] = target
                
            self.collapse_instances = count_inst
                
        # finalize data
        count_experiment(experiment_data)
    
    
    def inventory_count(self, experiment_data):
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
                bases = bases if not self.collapse_instances else set(bases)
                bases = bases if not set(bases) == {'ø'} else {'ø'} # count only one null value
                ecounts[target].update(bases)
                
        counts = dict((target, counts) for target, counts in ecounts.items()
                                if sum(counts.values()) >= self.min_obs)
        
        self.data = pd.DataFrame(counts).fillna(0)
        self.raw_data = experiment_data
        
        # assemble helper data: mapping from basis to list of search results
        self.basis2result = collections.defaultdict(list)
        for clause, bases in self.clause2basis2result.items():
            for basis, results in bases.items():
                self.basis2result[basis].extend(results)
        self.target2basis2result = collections.defaultdict(lambda: collections.defaultdict(list))
        for target, clauses in experiment_data.items():
            for clause, bases in clauses.items():
                for basis, results in self.clause2basis2result[clause].items():
                    self.target2basis2result[target][basis].extend(results)
        
    
    def frame_count(self, experiment_data):
        '''
        Counts frame experiment data into a dataframe from an
        experiment data dictionary structured as:
        experiment_data[target_token][clause][list_of_bases_tokens]
        
        Rather than counting individual instances of bases in a result,
        the frame_count sorts and assembles all the basis elements into
        a single string (i.e. "frame") which is then counted.
        
        --input--
        dict
        
        --output--
        pandas df
        '''
        
        ecounts = collections.defaultdict(lambda: collections.Counter())
        
        # helper data
        self.basis2result = collections.defaultdict(list)
        self.target2basis2result = collections.defaultdict(lambda: collections.defaultdict(list))
        
        for target, clauses in experiment_data.items():
            for clause, phrases in clauses.items():
                
                bases = tuple(tuple(bs) for bs in phrases.values())
                bases = bases if not set(bases) == {('ø',)} else (('ø',),)        
                
                # recursively combine all possible frame elements
                for frame_list in self.stitch_frames(bases):
                    frame = '|'.join(sorted(frame_list))
                    ecounts[target][frame] += 1
                
                    # helper data; combine all search results into a single result mapped to the frame
                    frame_results = set()
                    for basis, results in self.clause2basis2result[clause].items():
                        if basis not in frame_list:
                            continue
                        for result in results:
                            frame_results |= set(result)
                    self.basis2result[frame].append(tuple(frame_results))
                    self.target2basis2result[target][frame].append(tuple(frame_results))
                
        counts = dict((target, counts) for target, counts in ecounts.items()
                                if sum(counts.values()) >= self.min_obs)
        
        self.data = pd.DataFrame(counts).fillna(0)
        self.raw_data = experiment_data
        
        
    def stitch_frames(self, tokenlists):
        '''
        A recursive constructor for frames
        with multiple head elements.
        Returns all possible frames given
        the presence of multiple heads per
        phrase. Requires a list of token lists,
        where each token list corresponds to a phrase
        with multiple heads that have been tokenized.

        Credit: Thanks to Dirk Roorda for
        this code and helping me understand
        how it works.
        '''
        if len(tokenlists) == 1:
            for token in tokenlists[0]:
                yield (token,)

        elif len(tokenlists) > 1:
            for result in self.stitch_frames(tokenlists[1:]):
                for token in tokenlists[0]:
                    yield (token, *result)