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
from .kmedoids import kmedoids

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
    [!UNDER CONSTRUCTION & INCOMPLETE!]
    
    Retrieves BHSA cooccurrence data based on an experiment's parameters.
    Returns a Pandas dataframe with cooccurrence data.
    Requires TF api loaded with BHSA data and the appropriate features.    
    '''
    
    # easily configurable parameters:
    
    min_freq = 8    # min word occurrence frequency
    phrase_bases = {
                    ('Subj', 'Objc'):
                        {'Pred': self.predicate_basis},  
                    }
    subphrase_bases = {
                        ('subs',): 
                               {'rec': 1,
                                'par': 1,
                               },
                      }
    
    def __init__(self, bhsa_data_paths=['~/github/etcbc/bhsa/tf/c',
                                        '~/github/etcbc/lingo/heads/tf/c']):
    
        # load BHSA Hebrew data
        TF = Fabric(bhsa_data_paths, silent=True)
        tf_api = TF.load('''
                        function lex vs language
                        pdp freq_lex gloss domain ls
                        heads prep_obj
                      ''', silent=True)
        F, E, L, T = tf_api.F, tf_api.E, tf_api.L, tf_api.T  # shorten TF methods
        self.F, self.E, self.L, self.T = F, E, L, T # make available to class
        
        # raw cooccurrence data goes here
        target_counts = collections.defaultdict(lambda: collections.Counter())
        
        # Begin gathering data by clause:
        for clause in F.otype.s('clause'):
            
            # filter for targets
            targets = [E.heads.f(phrase) for phrase in L.d(clause, 'phrase') 
                          if self.target_parameters(phrase)]            
            if not targets: 
                continue
                
            # process and count context
            for target in targets:
                target_token = self.make_target()
                bases = self.map_context(target)
                target_counts[target_token].update(bases)
    
    '''
    // BHSA Helper Methods //
    '''
    
    def get_lex(self, lex_string):
        '''
        Return ETCBC lex node number from a lexeme string.
        Requires a text fabric feature class with otype/lex features loaded.
        '''
        F = self.tf_api.F
        lex = next(lex for lex in F.otype.s('lex') if F.lex.v(lex) == lex_string)
        return lex
    
    '''
    // Target Parameters & Methods //
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

        # apply various parameters at various linguistic levels
        # these parameters can be easily overwritten in subclasses
        test_parameters = all([self.target_phrase_parameters(phrase),
                               self.target_clause_parameters(phrase),
                               self.target_words_parameters(phrase)])
        
        return test_parameters # a boolean
        
    def target_phrase_parameters(self, phrase):
        '''
        Applies phrase parameters.
        This is a strict version that validates 
        only subject or object function phrases.
        Requires a phrase node number.
        '''
        # phrase features
        function = self.F.function.v(phrase)
        good_functions = {'Subj', 'Objc'} # check whether subj/obj phrase
        
        return bool(function in good_functions)
        
    def target_clause_parameters(self, phrase):
        '''
        Applies clause parameters.
        This version validates only narrative clauses.
        Requires a phrase node number.
        '''
        clause = self.L.u(phrase, 'clause')
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
        heads = E.heads.f(phrase)
        if not heads:
            return False # i.e. there is no available word to test
        
        # prepare for parameter checks
        language = self.F.language.v(head_nouns[0]) # language
        freq = [self.F.freq_lex.v(self.L.u(w, 'lex')[0]) for w in head_nouns] # word frequencies
        pdps = set(self.F.pdp.v(w) for w in head_nouns) # proper nouns
        ls = set(self.F.ls.v(w) for w in head_nouns) # gentilics
        
        # test all parameters
        test = all([language == 'Hebrew', 
                    min(freq) >= self.min_freq, # defined in __init__
                    'nmpr' not in pdps,
                    'gntl' not in ls,
                   ])
        
        return test

    def make_target(self, target):
        '''
        Maps a target word to its
        string representation.
        
        --input--
        word node
        
        --output--
        lexeme string
        '''
        return F.lex.v(target)
    
    '''
    // Basis & Context Mapping //
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
        F, E, L = self.tf_api.F, self.tf_api.E, self.tf_api.L
        
        # prepare context
        clause_phrases = L.d(L.u(target, 'clause')[0], 'phrase')
        target_funct = F.function.v(L.u(target, 'phrase')[0])
        target_pos = F.pdp.v(target)
        
        phrase_bases = [phrase for phrase in clause_phrases
                            if basis_phrase_parameters(phrase)]
        
        subphrase_bases = [related_sp for subphrase in L.u(target, 'subphrase')
                               for related_sp in E.mother.t(subphrase)
                               if basis_subphrase_parameters(related_sp)]
        
        subphrase_bases.extend([sp for sp in E.mother.t(target) 
                                    if basis_subphrase_parameters(sp)])
        
        bases = []
        
        # make the phrase-level basis elements
        for phrase in phrase_bases:
            basis_function = F.function.v(phrase)
            basis_constructor = phrase_paths[target_funct][basis_function]
            basis = basis_constructor(phrase, target_funct)
            bases.append(basis)

        # make the subphrase-level basis elements
        for subphrase in subphrase_bases:
            basis_rela = F.rela.v(subphrase)
            basis_constructor = subphrase_paths[target_pos][basis_rela]
            basis = basis_constructor(subphrase, target_funct)
            bases.append(basis)
        
        return tuple(bases)
    
    '''
    / Basis Parameters /
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
        return not restricted_lexemes & phrase_lexemes
    
    def basis_phrase_parameters(self, phrase, target_function):
        '''
        Defines and applies the parameters 
        for the selection of phrase basis elements.
        
        --input--
        phrase node
        
        --output--
        boolean on good basis candidate
        '''
        
        good_functions = self.phrase_bases[next(k for k in self.phrase_bases.keys() 
                                                if target_function in k)]
        basis_function = self.F.function.v(phrase)
        
        return all([basis_function in good_functions,
                    basis_lexical_restrictions(phrase)])
        
    def basis_subphrase_parameters(self, subphrase, target_pos):
        '''
        Defines and applies the parameters 
        for the selection of subphrase basis elements.
        
        --input--
        subphrase node
        
        --output--
        boolean on good basis candidate
        '''
        
        good_relas = self.subphrase_bases[next(k for k in self.subphrase_bases.keys() 
                                               if target_pos in k)]
        subphrase_relation = self.F.rela.v(subphrase)
        
        return all([subphrase_relation in good_relas,
                    basis_lexical_restrictions(subphrase)])
        
    '''
    / Basis Constructor Methods / 
    '''
        
    def predicate_basis(self, basis_phrase, target_function):
        '''
        Maps a verb to a string basis element, 
        where:
            verb_basis = target_function + verb_lex + verb_stem
            
        --input--
        phrase node, target function string
        
        --output--
        string basis element
        '''
        
        verb = [w for w in L.d(path, 'word') if F.pdp.v(w) == 'verb'][0]
        verb_lex = F.lex.v(verb)
        verb_stem = F.vs.v(verb)
        verb_basis = function + '.' + verb_lex + '.' + verb_stem # with function name added
        
        if verb and F.lex.v(verb) not in {'HJH['}: # omit "to be" verbs, others?
            for noun in nouns:
                cooccurrences[noun][verb_basis] += 1

    def noun_basis(self, basis_phrase, target_function):  
                    
        # count for subj/obj
        else:
            conouns = E.heads.f(path)
            cnoun_bases = set(function + '.' + F.lex.v(w) + f'.{pfunct}' for w in conouns) # with function name added
            counts = dict((basis, weight) for basis in cnoun_bases)
            if counts:
                for noun in nouns: # TO-DO: CHANGE THIS SO THAT A NOUN IS ONLY COUNTED ONCE, "NOUNS" is now in target parameters function
                    cooccurrences[noun].update(counts)

    def coord_subphrase_basis(self, basis_subphrase, target_function):
        
    # count coordinates
    for noun in nouns:
        for cnoun in nouns:
            if cnoun != noun:
                cnoun_basis = 'coor.'+cnoun # with coordinate function name
                cooccurrences[noun][cnoun_basis] += path_weights['coor']

        # weed out results with little data
        cooccurrences = dict((word, counts) for word, counts in cooccurrences.items()
                                if sum(counts.values()) >= 8
                            )

        self.data = pd.DataFrame(cooccurrences).fillna(0)
        
    