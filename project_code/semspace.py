'''
This module contains code used for constructing and evaluating a semantic space.
The space is built as a Pandas dataframe with target words as columns 
and co-occurrence (basis elements) as rows.

The module also contains an algorithm for selecting data from the BHSA,
the ETCBC's Hebrew Bible syntax data.
Several selection restrictions are applied in the preparation of that data.

A single class, SemSpace, contains all of the methods combined and can be used for
evaluations and analyses.

SemSpace requires an instance of another class, Experiment, which is responsible
for selecting and formatting the BHSA Hebrew data. The Experiment class is intended
to be easily modified into various subclasses, each of which represents a different
set of experiment parameters which I want to test. To accomplish this, Experiment's 
tasks are broken down into a bunch of methods that can be overwritten in subclasses. 

*NB*
Paths here for the BHSA data must be modified to run correctly on your system.
See first function, load_tf_bhsa.
'''

import collections, os, math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from tf.fabric import Fabric
from tf.extra.bhsa import Bhsa
from kmedoids.kmedoids import kMedoids
from lingo.heads.heads import find_quantified

class SemSpace:
    '''
    This class brings together all of the methods defined in 
    notebook 4 of my semantics repository. This class has a
    number of attributes which contain cooccurrence counts,
    adjusted counts, similarity scores, and more.
    The class loads and processes the data in one go.
    '''
    
    def __init__(self, experiment, info=0):
        '''
        Requires an experiment class (defined below).
        This allows various experiment parameters
        to be feed to the semantic space builder.
        
        "info" is an optionally divisable number
        if status updates are desired for lengthy
        calculations.
        '''
        
        # adjust raw counts with log-likelihood & pointwise mutual information
        self.loglikelihood = self.apply_loglikelihood(experiment)
        self.pmi = self.apply_pmi(experiment)
        self.raw = experiment
        
        # apply pca or other data maneuvers
        self.pca_ll = self.apply_pca(self.loglikelihood)
        self.pca_ll_3d = self.apply_pca(self.loglikelihood, n=3)
        self.pca_pmi = self.apply_pca(self.pmi)
        self.pca_raw = self.apply_pca(experiment)

    '''
    -----
    Association Measures:
        i.e. applying adjustments to raw frequency counts
        The measures are based on various sources including Padó & Lapita (2007)
        and Levshina (2015).
    '''
    
    def safe_log(self, number):
        '''
        Evaluate for zero before applying log function.
        '''
        if number == 0:
            return 0
        else:
            return math.log(number)

    def loglikelihood(self, k, l, m, n, log):
        '''
        Returns the log-likelihood when the supplied elements are given.
        via Padó & Lapita 2007
        '''
    
        llikelihood = 2*(k*log(k) + l*log(l) + m*log(m) + n * log(n)        
                            - (k+l)*log(k+l) - (k+m)*log(k+m)
                            - (l+n)*log(l+n) - (m+n)*log(m+n)
                            + (k+l+m+n)*log(k+l+m+n))

        return llikelihood

    def apply_loglikelihood(self, comatrix, info=0):

        '''
        Adjusts values in a cooccurrence matrix using log-likelihood. 
        Requires a cooccurrence matrix.
        
        An option for progress updates is commented out.
        This option can be useful for very large datasets
        that take more than a few seconds to execute.
        i.e. sets that contain target words > ~500
        
        via Padó & Lapita 2007
        '''
        safe_log = self.safe_log
        log_likelihood = self.loglikelihood
        
        new_matrix = comatrix.copy()
        #optional: information for large datasets
        i = 0 
        self.tf_api.indent(reset=True)
        self.tf_api.info('beginning calculations...')
        self.tf_api.indent(1, reset=True)
        
        for target in comatrix.columns:
            for basis in comatrix.index:
                k = comatrix[target][basis]

                if not k:
                    i += 1
                    if info and i % info == 0:
                        self.tf_api.indent(1)
                        self.tf_api.info(f'at iteration {i}')
                    continue

                l = comatrix.loc[basis].sum() - k
                m = comatrix[target].sum() - k
                n = comatrix.values.sum() - (k+l+m)
                ll = self.loglikelihood(k, l, m, n, safe_log)
                new_matrix[target][basis] = ll
                
                # optional: information for large datasets
                i += 1
                if info and i % info == 0:
                    self.tf_api.indent(1)
                    self.tf_api.info(f'at iteration {i}')
        self.tf_api.indent(0)
        self.tf_api.info(f'FINISHED at iteration {i}')
        
        return new_matrix
    
    def apply_pmi_column(self, col, datamatrix):

        '''
        Apply PMI to a given column.
        Method derived from Levshina 2015.
        '''
        expected = col * datamatrix.sum(axis=1) / datamatrix.values.sum()
        pmi = np.log(col / expected).fillna(0)
        return pmi
    
    def apply_pmi(self, datamatrix):
        '''
        Apply pmi to a data matrix.
        Method derived from Levshina 2015.
        '''
        return datamatrix.apply(lambda k: self.apply_pmi_column(k, datamatrix))
    
    '''
    // Data Transformations and Cluster Analyses //
    '''

    def apply_pca(self, comatrix, n=2):
        '''
        Apply principle component analysis to a supplied cooccurrence matrix.
        '''
        pca = PCA(n_components=n)
        return pca.fit_transform(comatrix.T.values)
        
class Experiment:
    
    '''    
    Retrieves BHSA cooccurrence data based on an experiment's parameters.
    Returns a Pandas dataframe with cooccurrence data.
    Requires TF api loaded with BHSA data and the appropriate features.    
    '''
    
    def __init__(self, bhsa_data_paths=['~/github/etcbc/bhsa/tf/c',
                                        'lingo/heads/tf/c']):

        self.config()
    
        # load BHSA Hebrew data
        TF = Fabric(bhsa_data_paths, silent=True)
        tf_api = TF.load('''
                        function lex vs language
                        pdp freq_lex gloss domain ls
                        heads prep_obj mother rela
                      ''', silent=True)
        
        F, E, L, T = tf_api.F, tf_api.E, tf_api.L, tf_api.T  # shorten TF methods
        self.TF, self.F, self.E, self.L, self.T = TF, F, E, L, T # make available to class
        
        # raw cooccurrence data goes here
        target_counts = collections.defaultdict(lambda: collections.Counter())
        
        # Begin gathering data by clause:
        for clause in F.otype.s('clause'):
            
            # filter for targets
            targets = [head for phrase in L.d(clause, 'phrase') 
                          for head in E.heads.f(phrase)
                          if self.target_parameters(phrase)]            
            if not targets: 
                continue

            # process and count context
            for target in targets:
                target_token = self.make_target_token(target)
                bases = self.map_context(target)
                target_counts[target_token].update(bases)
    
        # filter and arrange data
        target_counts = dict((word, counts) for word, counts in target_counts.items()
                                if sum(counts.values()) >= self.min_observation_freq
                            )
        self.data = pd.DataFrame(target_counts).fillna(0)

    
    def config(self):
        '''
        / Experiment Configurations /
        '''
        self.min_target_freq = 8
        self.min_observation_freq = 8
        self.target2basis = {
                                ('Subj', 'Objc'):
                                    {'Pred': self.predicate_basis},  

                                ('subs',): 
                                    {'par': self.coordinate_noun_basis},
                            }
    
    '''
    / Target Parameters & Methods /
    ''' 
        
    def target_parameters(self, phrase):
        
        '''
        Evaluates whether a phrase matches the supplied
        parameters. Parameters are broken down into smaller
        methods, so that they can be easily modified or exchanged
        for other experiment parameters.
        
        --input--
        phrase node
        
        --output--
        boolean
        '''

        return all([self.target_phrase_parameters(phrase),
                    self.target_clause_parameters(phrase),
                    self.target_word_parameters(phrase)])
        
        
    def target_phrase_parameters(self, phrase):
        '''
        Applies phrase parameters.
        This is a strict version that validates 
        only subject or object function phrases.
        Requires a phrase node number.
        '''
        # phrase features
        function = self.F.function.v(phrase)
        good_functions = set(target for target_group in self.target2basis
                                for target in target_group)
        
        return bool(function in good_functions)
        
        
    def target_clause_parameters(self, phrase):
        '''
        Applies clause parameters.
        This version validates only narrative clauses.
        Requires a phrase node number.
        '''
        clause = self.L.u(phrase, 'clause')[0]
        domain = self.F.domain.v(clause) # check for narrative or discourse
        
        return bool(domain == 'N')
        
        
    def target_word_parameters(self, phrase):
        '''
        Applies word-level parameters on phrase heads.
        This version checks for frequency,
        head words, and proper names.
        
        --input--
        phrase node
        
        --output--
        boolean on acceptable word
        '''
        
        # check for heads
        heads = self.E.heads.f(phrase)
        if not heads:
            return False # i.e. there is no available word to test
        
        # prepare for parameter checks
        language = self.F.language.v(heads[0]) # language
        freq = [self.F.freq_lex.v(self.L.u(w, 'lex')[0]) for w in heads] # word frequencies
        pdps = set(self.F.pdp.v(w) for w in heads) # proper nouns
        ls = set(self.F.ls.v(w) for w in heads) # gentilics
        
        # test all parameters
        test = all([language == 'Hebrew', 
                    min(freq) >= self.min_target_freq,
                    'nmpr' not in pdps,
                    'gntl' not in ls,
                   ])
        
        return test

    
    def make_target_token(self, target):
        '''
        Maps a target word to its
        string representation.
        
        --input--
        word node
        
        --output--
        lexeme string
        '''
        return self.F.lex.v(target)
    
    
    '''
    / Basis & Context Mapping /
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
        
        # define shortform TF api methods
        F, E, L = self.F, self.E, self.L
        
        # prepare context
        clause_phrases = L.d(L.u(target, 'clause')[0], 'phrase')
        target_funct = F.function.v(L.u(target, 'phrase')[0])
        target_group = next(k for k in self.target2basis.keys() if target_funct in k)
        target_pos = F.pdp.v(target)
        
        phrase_bases = [phrase for phrase in clause_phrases
                            if self.basis_phrase_parameters(phrase, target_group)]
        
        subphrase_bases = [related_sp for subphrase in L.u(target, 'subphrase')
                               for related_sp in E.mother.t(subphrase)
                               if self.basis_subphrase_parameters(related_sp, target_group)]
        
        subphrase_bases.extend([sp for sp in E.mother.t(target) 
                                    if self.basis_subphrase_parameters(sp, target_group)])
        
        bases = []
        
        # make the phrase-level basis elements
        for phrase in phrase_bases:
            basis_function = F.function.v(phrase)
            basis_constructor = self.target2basis[target_group][basis_function]
            basis = basis_constructor(phrase, target_group)
            bases.extend(basis)

        # make the subphrase-level basis elements
        for subphrase in subphrase_bases:
            basis_rela = F.rela.v(subphrase)
            basis_constructor = self.target2basis[target_group][basis_rela]
            basis = basis_constructor(subphrase, target_group)
            bases.extend(basis)
        
        return tuple(bases)
    
    
    '''
    // Basis Parameters //
    '''
    
    
    def basis_lexical_restrictions(self, phrase):
        '''
        Tests for restricted lexemes in a 
        basis candidate. E.g. the verb היה
        which is less informative for semantic
        meaning.
        
        --input--
        phrase node
        
        --output--
        boolean on acceptable lexeme
        '''
        
        restricted_lexemes = {'HJH['}
        lexemes = set(self.F.lex.v(w) for w in self.E.heads.f(phrase) or self.L.d(phrase, 'word'))
        return not restricted_lexemes & lexemes
    
    
    def basis_phrase_parameters(self, phrase, target_group):
        '''
        Defines and applies the parameters 
        for the selection of phrase basis elements.
        
        --input--
        phrase node
        
        --output--
        boolean on good basis candidate
        '''
        
        good_functions = self.target2basis[target_group]
        basis_function = self.F.function.v(phrase)
        
        return all([basis_function in good_functions,
                    self.basis_lexical_restrictions(phrase)])
        
        
    def basis_subphrase_parameters(self, subphrase, target_group):
        '''
        Defines and applies the parameters 
        for the selection of subphrase basis elements.
        
        --input--
        subphrase node
        
        --output--
        boolean on good basis candidate
        '''
        
        good_relas = self.target2basis[target_group]
        subphrase_relation = self.F.rela.v(subphrase)
        
        return all([subphrase_relation in good_relas,
                    self.basis_lexical_restrictions(subphrase)])
       
        
    '''
    // Basis Constructor Methods //
    '''
       
        
    def predicate_basis(self, basis_phrase, target_function):
        '''
        Maps a verb to a string basis element, 
        where:
            verb_basis = target_function + verb_lex + verb_stem
            
        --input--
        phrase node, target function string
        
        --output--
        basis element string
        '''
        
        verb = self.E.heads.f(basis_phrase)[0]
        lex = self.F.lex.v(verb)
        stem = self.F.vs.v(verb)
        return (f'{target_function}.{lex}.stem',)       
    
    
    def noun_basis(self, basis_phrase, target_function):  
        '''
        Maps a noun to a string basis element, 
        where:
            noun_basis = target_function + noun_token
            
        --input--
        phrase node, target function string
        
        --output--
        basis element string(s)
        '''
        
        if self.F.typ.v(basis_phrase) == 'NP':
            bases_tokens = [make_basis_token(h) for h in self.E.heads.f(phrase)]
        
        elif self.F.typ.v(basis_phrase) == 'PP':
            bases_tokens = [make_basis_token(obj) for prep in self.E.heads.f(phrase)
                               for obj in self.E.prep_obj.f(prep)]
            
        return tuple(f'{target_function}.{token}' for token in bases_tokens)
    
    
    def make_basis_token(self, basis):
        '''
        Makes a basis token out of a word node.
        
        --input--
        word node
        
        --output--
        basis token string
        '''
        
        return self.F.lex.v(basis)
        
        
    def coordinate_noun_basis(self, basis_subphrase, target_function):
        '''
        Maps coordinate nouns to bases, 
        i.e. nouns connected to a target with a conjunction
        where:
            noun_basis = target_function + noun_token
            
        --input--
        phrase node, target function string
        
        --output--
        basis element string
        '''
        
        head = next(find_quantified(w) or w
                        for w in self.L.d(subphrase, 'word')
                        if self.F.pdp.v(w) == 'subs')
        
        return f'coor.{self.F.lex.v(head)}'