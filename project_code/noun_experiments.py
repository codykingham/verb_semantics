'''
The following experiments are constructed for building
new noun spaces, which can be use to enhance the verb spaces.
The classes are based on the Experiment class, which is responsible for selecting and processing
BHSA data into target and basis word strings. The Experiment class is intended
to be easily modified into various subclasses, each of which represents a different
set of experiment parameters which I want to test. To accomplish this, Experiment's 
tasks are broken down into a bunch of small methods that can be overwritten in subclasses. 
'''

import collections, os, math
import numpy as np
import pandas as pd
from tf.fabric import Fabric
if not __package__:
    from lingo.heads.heads import find_quantified
    from experiments import Experiment
else:
    from .lingo.heads.heads import find_quantified
    from .experiments import Experiment
    
bhsa_data_paths=['~/github/etcbc/bhsa/tf/c',
                 '~/github/semantics/project_code/lingo/heads/tf/c']


class NounExperiment1(Experiment):
    
    '''
    In this experiment, a noun space 
    is constructed that includes as much
    informative information as possible. Thus,
    many kinds of relations are considered including
    apposition, construct, and coordinations.
    
    The parameters for this experiment are in many ways a reversed
    version of the verb experiment, with some modifications.
    
    Caution:
    This will be a big space.
    '''
    
    def __init__(self, tf_api=None):
        
        super().__init__(tf_api=tf_api)
        
    def config(self):
        '''
        Experiment Configurations
        '''
        self.min_target_freq = 1
        self.min_observation_freq = 1
        self.target2basis = { 
                                ('Subj', 'Objc'): # clause-internal relations
                                    {('Pred', 'PreO', 'PtcO', 'PreS'): self.make_predicate_basis, 
                                     ('Para', 'Appo'): self.make_phraseA_rela_basis},
                                
                                ('PrAd', 'Adju', 'Cmpl', 'Loca', 'Time'):
                                    {('Pred', 'PreO', 'PtcO', 'PreS'): self.make_pred_cmpl_basis, 
                                     ('Para', 'Appo'): self.make_phraseA_rela_basis},

                                ('subs', 'nmpr', 'advb', 'adjv'): 
                                    {('par', 'rec'): self.make_subphrase_rela_basis, # subphrase relations
                                    },
                            }
        
    '''
    <><><><>
    TARGET CONSTRUCTION
    <><><><>
    '''
        
    def target_word_parameters(self, word):
        '''
        Applies word-level parameters on phrase heads.
        This version checks for frequency,
        head words, and proper names.
        
        --input--
        phrase node
        
        --output--
        boolean on acceptable word
        '''
        
        # prepare for parameter checks
        language = self.F.language.v(word) # language
        freq = self.F.freq_lex.v(self.L.u(word, 'lex')[0]) # word frequencies
        
        # test all parameters
        return all([language == 'Hebrew', 
                    freq >= self.min_target_freq,])

    def make_target_token(self, target):
        '''
        Maps a target word to its
        string representation.
        
        --input--
        word node
        
        --output--
        lexeme string
        '''
        
        lex = self.F.lex.v(target)
        return lex

    '''
    <><><><>
    MAP CONTEXT
    <><><><>
    '''
    
    def map_context(self, target):
        '''
        Maps context elements in relation
        to a supplied target word to strings.
        The strings serve as the basis elements
        which are counted as cooccurrences.
        
        --input--
        word node
        
        --output--
        tuple of strings
        '''
        
        self.target = target
        
        # define shortform TF api methods
        F, E, L = self.F, self.E, self.L
        
        # prepare context
        clause_phrases = L.d(L.u(target, 'clause')[0], 'phrase')
        target_funct = F.function.v(L.u(target, 'phrase')[0])
        target_pos = F.pdp.v(target)
        phrase_tgroup = next((k for k in self.target2basis.keys() if target_funct in k), 0) 
        subphrase_tgroup = next((k for k in self.target2basis.keys() if target_pos in k), 0)

        phrase_bases = [phrase for phrase in clause_phrases # inner-clausal relations
                            if self.basis_phrase_parameters(phrase, phrase_tgroup)]
        
        phraseA_bases = [related_pa for phrase_atom in L.u(target, 'phrase_atom') # phrase atom relations
                            for related_pa in E.mother.t(phrase_atom)
                            if self.basis_phraseA_parameters(related_pa, phrase_tgroup)]
        
        subphrase_bases = [related_sp for subphrase in L.u(target, 'subphrase') # subphrase relations
                              for related_sp in E.mother.t(subphrase)
                              if self.basis_subphrase_parameters(related_sp, subphrase_tgroup)]

        subphrase_bases.extend([sp for sp in E.mother.t(target)  # handle nomen rectum relations
                                    if self.basis_subphrase_parameters(sp, subphrase_tgroup)])
        
        bases = []
        
        # make the phrase-level basis elements
        for phrase in phrase_bases:
            basis_function = F.function.v(phrase)
            phrase_bgroup = next((k for k in self.target2basis[phrase_tgroup].keys() if basis_function in k), 0)     
            basis_constructor = self.target2basis[phrase_tgroup][phrase_bgroup]
            basis = basis_constructor(phrase, target)
            bases.extend(basis)

        # make the phrase atom level basis elements
        for phraseA in phraseA_bases:
            basis_rela = F.rela.v(phraseA)
            phraseA_bgroup = next((k for k in self.target2basis[phrase_tgroup].keys() if basis_rela in k), 0)
            basis_constructor = self.target2basis[phrase_tgroup][phraseA_bgroup]
            basis = basis_constructor(phraseA, target)
            bases.extend(basis)
            
        # make the subphrase-level basis elements
        for subphrase in subphrase_bases:
            basis_rela = F.rela.v(subphrase)
            subphrase_bgroup = next((k for k in self.target2basis[subphrase_tgroup].keys() if basis_rela in k), 0)     
            basis_constructor = self.target2basis[subphrase_tgroup][subphrase_bgroup]
            basis = basis_constructor(subphrase, target)
            bases.extend(basis)
        
        return tuple(bases)    
    
    '''
    <><><><>
    BASIS CONSTRUCTION
    <><><><>
    '''
    
    def basis_phraseA_parameters(self, phrase_atom, target_group):
        '''
        Defines and applies the parameters 
        for the selection of phrase atom basis elements.
        
        --input--
        phrase atom node
        
        --output--
        boolean on good basis candidate
        '''
        
        good_relas = set(k for group in self.target2basis.get(target_group, {}) for k in group)
        phraseA_relation = self.F.rela.v(phrase_atom)
        
        return all([phraseA_relation in good_relas,
                    self.basis_lexical_restrictions(phrase_atom)])
    
    def make_subphrase_rela_basis(self, basis_subphrase, target):
        
        '''
        Maps subphrase relations to a basis element.
        Valid relations include nomen rectum and parallel.
            
        --input--
        subphrase node, target function string
        
        --output--
        basis element string
        '''
        target_pdp = self.F.pdp.v(target)
        good_sp_heads = {'subs': {'subs', 'nmpr'},
                         'advb': {'advb'},
                         'adjv': {'adjv'}}
        
        head = next((find_quantified(w, self.tf_api) or w
                        for w in self.L.d(basis_subphrase, 'word')
                        if self.F.pdp.v(w) in good_sp_heads.get(target_pdp, {})), 0)
        if head:
            basis_rela = self.F.rela.v(basis_subphrase)
            return (f'.{basis_rela}.{self.F.lex.v(head)}',)
        
        else:
            return tuple()
        
    def make_phraseA_rela_basis(self, basis_phraseA, target):
        '''
        Maps coordinates to a basis string
        based on the Para phrase atom relation.
        Depends on the Heads feature to pull
        the phrase atom's head.
        
        --input--
        basis phrase node and target word node
        
        --output--
        tuple with basis string
        '''
        
        head = next((h for h in self.E.heads.f(basis_phraseA)), 0)
        
        
        if head:
            basis_rela = self.F.rela.v(basis_phraseA)
            basis_rela = basis_rela if basis_rela != 'Para' else 'par' # make equal with subphrase parallel code
            return (f'.{basis_rela}.{self.F.lex.v(head)}',)
            
        else:
            return tuple()
        
    def make_pred_cmpl_basis(self, basis_phrase, target):
        '''
        Maps a predicate element to an associated
        complement basis tag. Includes a special
        parameter to include prepositional data.
        '''
        
        verb = self.E.heads.f(basis_phrase)[0]
        lex = self.F.lex.v(verb)
        stem = self.F.vs.v(verb)
        target_phrase = self.L.u(target, 'phrase')[0]
        target_function = self.F.function.v(target_phrase)
        
        if self.E.prep_obj.t(target):
            prep = self.E.prep_obj.t(target)[0]
            prep_lex = self.F.lex.v(prep)
            return (f'{target_function}_{prep_lex}.Pred.{lex}.{stem}',)
            
        else:
            return (f'{target_function}.Pred.{lex}.{stem}',) 
            