'''
This module contains an Experiment class which is responsible for selecting and processing
BHSA data into target and basis word strings. The Experiment class is intended
to be easily modified into various subclasses, each of which represents a different
set of experiment parameters which I want to test. To accomplish this, Experiment's 
tasks are broken down into a bunch of small methods that can be overwritten in subclasses. 

Beneath the default Experiment class, which closely resembles the parameters of phase 1 of this 
project, are a set of derivative experiments and their corresponding parameters.
'''

import collections, os, math
import numpy as np
import pandas as pd
from tf.fabric import Fabric
if not __package__:
    from lingo.heads.heads import find_quantified
else:
    from .lingo.heads.heads import find_quantified
    
bhsa_data_paths=['~/github/etcbc/bhsa/tf/c',
                 '~/github/semantics/project_code/lingo/heads/tf/c']

class Experiment:
    
    '''    
    Retrieves BHSA cooccurrence data based on an experiment's parameters.
    Returns a Pandas dataframe with cooccurrence data.
    Requires TF api loaded with BHSA data and the appropriate features.    
    '''
    
    def __init__(self, tf_api=None):

        self.config()
    
        # configure Text-Fabric
        if tf_api:
            self.tf_api = tf_api
        else:
            self.load_tf()      
            
        # configure shortform TF methods
        self.F, self.E, self.L, self.T = self.tf_api.F, self.tf_api.E, self.tf_api.L, self.tf_api.T
        F, E, L, T = self.F, self.E, self.L, self.T
            
        # raw cooccurrence data goes here
        target_counts = collections.defaultdict(lambda: collections.Counter())
        self.target2gloss = dict()
        self.target2lex = dict()
        self.target2node = dict()
        
        # Begin gathering data by clause:
        for clause in F.otype.s('clause'):
            
            # filter for targets
            targets = [target for phrase in L.d(clause, 'phrase') 
                          for target in self.get_targets(phrase)
                          if self.target_parameters(target)]            
            if not targets: 
                continue

            # process and count context
            for target in targets:
                target_token = self.make_target_token(target)
                self.target2gloss[target_token] = F.gloss.v(L.u(target, 'lex')[0])
                self.target2lex[target_token] = L.u(target, 'lex')[0]
                self.target2node[target_token] = target
                bases = self.map_context(target)
                if bases:
                    target_counts[target_token].update(bases)
    
        # filter and arrange data
        target_counts = dict((word, counts) for word, counts in target_counts.items()
                                if sum(counts.values()) >= self.min_observation_freq
                            )
        self.data = pd.DataFrame(target_counts).fillna(0)
        self.raw_data = target_counts

    def config(self):
        '''
        Experiment Configurations
        '''
        self.min_target_freq = 8
        self.min_observation_freq = 8
        self.target2basis = {
                                ('Subj', 'Objc'):
                                    {('Pred',): self.make_predicate_basis},  

                                ('subs',): 
                                    {('par',): self.make_coordinate_noun_basis},
                            }
    
    def load_tf(self):
        
        '''
        Loads an instance of TF if necessary.
        '''
        
        # load BHSA Hebrew data
        TF = Fabric(bhsa_data_paths, silent=True)
        tf_api = TF.load('''
                        function lex vs language
                        pdp freq_lex gloss domain ls
                        heads prep_obj mother rela
                        typ sp
                      ''', silent=True)
        
        self.tf_api = tf_api
            
    '''
    / Target Parameters & Methods /
    ''' 
        
    def get_targets(self, phrase):
        '''
        Extracts a target word based on the
        phrase type using the heads.tf and
        prep_obj.tf features.
        
        --input--
        phrase node
        
        --output--
        tuple of target word nodes
        '''
        
        if self.F.typ.v(phrase) == 'PP':
            return [obj for prep in self.E.heads.f(phrase)
                       for obj in self.E.prep_obj.f(prep)]
        else:
            return self.E.heads.f(phrase)
        
    def target_parameters(self, target):
        
        '''
        Evaluates whether a word matches the supplied
        parameters. Parameters are broken down into smaller
        methods, so that they can be easily modified or exchanged
        for other experiment parameters.
        
        --input--
        word node
        
        --output--
        boolean
        '''

        phrase = self.L.u(target, 'phrase')[0]
        clause = self.L.u(target, 'clause')[0]
        
        return all([self.target_phrase_parameters(phrase),
                    self.target_clause_parameters(clause),
                    self.target_word_parameters(target)])
        
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
        
    def target_clause_parameters(self, clause):
        '''
        Applies clause parameters.
        This version validates only narrative clauses.
        Requires a clause node number.
        '''
        domain = self.F.domain.v(clause) # check for narrative or discourse
        return bool(domain == 'N')
           
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
        pdp = self.F.pdp.v(word) # proper nouns
        ls = self.F.ls.v(word) # gentilics
        
        # test all parameters
        return all([language == 'Hebrew', 
                    freq >= self.min_target_freq,
                    pdp == 'subs',
                    ls != 'gntl'])

    
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
        target_pos = F.pdp.v(target)
        phrase_tgroup = next((k for k in self.target2basis.keys() if target_funct in k), 0)        
        subphrase_tgroup = next((k for k in self.target2basis.keys() if target_pos in k), 0)

        phrase_bases = [phrase for phrase in clause_phrases
                            if self.basis_phrase_parameters(phrase, phrase_tgroup)]
        
        subphrase_bases = [related_sp for subphrase in L.u(target, 'subphrase')
                              for related_sp in E.mother.t(subphrase)
                              if self.basis_subphrase_parameters(related_sp, subphrase_tgroup)]

        subphrase_bases.extend([sp for sp in E.mother.t(target) 
                                    if self.basis_subphrase_parameters(sp, subphrase_tgroup)])
        
        bases = []
        
        # make the phrase-level basis elements
        for phrase in phrase_bases:
            basis_function = F.function.v(phrase)
            phrase_bgroup = next((k for k in self.target2basis[phrase_tgroup].keys() if basis_function in k), 0)     
            basis_constructor = self.target2basis[phrase_tgroup][phrase_bgroup]
            basis = basis_constructor(phrase, target_funct)
            bases.extend(basis)

        # make the subphrase-level basis elements
        for subphrase in subphrase_bases:
            basis_rela = F.rela.v(subphrase)
            subphrase_bgroup = next((k for k in self.target2basis[subphrase_tgroup].keys() if basis_rela in k), 0)     
            basis_constructor = self.target2basis[subphrase_tgroup][subphrase_bgroup]
            basis = basis_constructor(subphrase, target_funct)
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
        
        good_functions = set(k for group in self.target2basis.get(target_group, {}) for k in group)
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
        
        good_relas = set(k for group in self.target2basis.get(target_group, {}) for k in group)
        subphrase_relation = self.F.rela.v(subphrase)
        
        return all([subphrase_relation in good_relas,
                    self.basis_lexical_restrictions(subphrase)])
       
    '''
    // Basis Constructor Methods //
    '''
        
    def make_predicate_basis(self, basis_phrase, target_function):
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
        return (f'{target_function}.Pred.{lex}.{stem}',)       
     
    def make_noun_basis(self, basis_phrase, target_function):  
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
        
    def make_coordinate_noun_basis(self, basis_subphrase, target_function):
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
        
        # TO-DO: Fix this workaround properly. Should I have a 
        # method with parameters on coordinate noun selection?
        
        try:
            head = next(find_quantified(w, self.tf_api) or w
                        for w in self.L.d(basis_subphrase, 'word')
                        if self.F.pdp.v(w) == 'subs')
            return (f'{target_function}.coor.{self.F.lex.v(head)}',)
        
        except StopIteration:
            return tuple()
      
'''
Experiments with verb vector spaces:
Composed during phase2 initial experiments notebook.
'''
    
class VerbExperiment1(Experiment):
    
    def __init__(self, tf_api=None):
        super().__init__(tf_api=tf_api)
        
    def config(self):
        '''
        Experiment Configurations
        '''
        self.min_target_freq = 0
        self.min_observation_freq = 0
        self.target2basis = {
                                ('Pred', 'PreO', 'PreS', 'PtcO'):
                                    {('PrAd', 'Adju', 'Cmpl', 'Loca', 'Time', 'Objc', 'Subj'): self.make_adverbial_bases},  
                            }
        
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
        
        # parameter checks
        language = self.F.language.v(word) # language
        freq = self.F.freq_lex.v(self.L.u(word, 'lex')[0]) # word frequency
        pdp = self.F.pdp.v(word) # part of speech
        lex = self.F.lex.v(word) # lexeme
        
        return all([language == 'Hebrew', 
                    freq >= self.min_target_freq,
                    pdp == 'verb',
                    lex not in {'HJH['}])
    
    def make_adverbial_bases(self, phrase, target_funct):
        '''
        Builds a basis string from a supplied
        adverbial phrase. Treats prepositional
        phrases different from other phrase types.
        
        --input--
        basis phrase node
        
        --output--
        basis string
        '''
        
        function = self.F.function.v(phrase)
        heads = self.E.heads.f(phrase)
        
        if self.F.typ.v(phrase) == 'PP':
            preps = [self.F.lex.v(h) for h in heads]
            objs = [self.F.lex.v(obj) for prep in heads for obj in self.E.prep_obj.f(prep)]
            basis_tokens = '|'.join(f'{prep}_{obj}' for prep, obj in zip(preps, objs))
            if basis_tokens:
                return (f'{target_funct}.{function}.{basis_tokens}',)
            else:
                return tuple()
            
        else:
            return tuple(f'{target_funct}.{function}.{self.F.lex.v(w)}' for w in heads)
    
class VerbMinLexBasis(VerbExperiment1):
    
    '''
    This experiment supresses lexical data
    for complement bases.
    '''
    
    def __init__(self, tf_api=None):
        super().__init__(tf_api=tf_api)
        
    def make_adverbial_bases(self, phrase, target_funct):
        '''
        Builds a basis string from a supplied
        adverbial phrase. Treats prepositional
        phrases different from other phrase types.
        
        --input--
        basis phrase node
        
        --output--
        basis string
        '''
        
        function = self.F.function.v(phrase)
        heads = self.E.heads.f(phrase)
        
        if self.F.typ.v(phrase) == 'PP':
            preps = [self.F.lex.v(h) for h in heads]
            objs = [self.F.pdp.v(obj) for prep in heads for obj in self.E.prep_obj.f(prep)]
            basis_tokens = '|'.join(f'{prep}_{obj}' for prep, obj in zip(preps, objs))
            basis_tokens = basis_tokens or '|'.join(self.F.lex.v(h) for h in heads)
            return (f'{target_funct}.{function}.{basis_tokens}',)

        else:
            return tuple(f'{target_funct}.{function}.{self.F.lex.v(w)}' for w in heads)
        
class VerbNoSubj(VerbExperiment1):
    
    '''
    This experiment excludes subjects from functioning
    as basis elements.
    '''
    
    def __init__(self, tf_api=None):
        super().__init__(tf_api=tf_api)
        
    def config(self):
        '''
        Experiment Configurations
        '''
        self.min_target_freq = 0
        self.min_observation_freq = 0
        self.target2basis = {
                                ('Pred', 'PreO', 'PreS', 'PtcO'):
                                    {('PrAd', 'Adju', 'Cmpl', 'Loca', 'Time', 'Objc'): self.make_adverbial_bases},  
                            }
        
class VerbAndStem(VerbNoSubj):
    
    '''
    This experiment adds the stem to the verb lexeme
    during the target token construction.
    The parameters of the VerbNoSubj are inherited
    due to the success of that experiment.
    '''
    
    def __init__(self, tf_api=None):
        super().__init__(tf_api=tf_api)
    
    def make_target_token(self, target):
        '''
        Maps a target word to its
        string representation.
        
        --input--
        word node
        
        --output--
        lexeme string
        '''
        
        stem = self.F.vs.v(target)
        lex = self.F.lex.v(target)
        
        return f'{lex}.{stem}'
    
'''
Experiments with single-basis verb spaces: namely
subject-only, object-only, and complement-only verb
spaces. Explored in the phase 2 clustering experiment 
notebook.
'''

class VerbSubjOnly(VerbExperiment1):
    
    '''
    In this experiment, only subjects
    are examined as basis elements.
    '''
    
    def __init__(self, tf_api=None):
        super().__init__(tf_api=tf_api)
        
    def config(self):
        '''
        Experiment Configurations
        '''
        self.min_target_freq = 0
        self.min_observation_freq = 0
        self.target2basis = {
                                ('Pred', 'PreO', 'PreS', 'PtcO'):
                                    {('Subj',): self.make_adverbial_bases},  
                            }
        
class VerbObjOnly(VerbExperiment1):
    
    '''
    In this experiment, only objects
    are examined as basis elements.
    '''
    
    def __init__(self, tf_api=None):
        super().__init__(tf_api=tf_api)
        
    def config(self):
        '''
        Experiment Configurations
        '''
        self.min_target_freq = 0
        self.min_observation_freq = 0
        self.target2basis = {
                                ('Pred', 'PreO', 'PreS', 'PtcO'):
                                    {('Objc',): self.make_adverbial_bases},  
                            }

class VerbCmplOnly(VerbExperiment1):
    
    '''
    In this experiment, only complements
    are examined as basis elements.
    
    In this version, Loca (location) and Time
    are excluded as complementizers to eliminate 
    looser connections based less on semantic class
    than on similarity of temporal/locative context.
    '''
    
    def __init__(self, tf_api=None):
        super().__init__(tf_api=tf_api)
        
    def config(self):
        '''
        Experiment Configurations
        '''
        self.min_target_freq = 0
        self.min_observation_freq = 0
        self.target2basis = {
                                ('Pred', 'PreO', 'PreS', 'PtcO'):
                                    {('PrAd', 'Adju', 'Cmpl'): self.make_adverbial_bases},  
                            }
        
class VerbMinLexBasis2(VerbExperiment1):
    
    '''
    This experiment supresses lexical data
    for complement bases.
    
    This version also suppresses lexical
    info into a part of speech tag for 
    non-prepositional objects as well.
    '''
    
    def __init__(self, tf_api=None):
        super().__init__(tf_api=tf_api)
        
    def make_adverbial_bases(self, phrase, target_funct):
        '''
        Builds a basis string from a supplied
        adverbial phrase. Treats prepositional
        phrases different from other phrase types.
        
        --input--
        basis phrase node
        
        --output--
        basis string
        '''
        
        function = self.F.function.v(phrase)
        heads = self.E.heads.f(phrase)
        
        if self.F.typ.v(phrase) == 'PP':
            preps = [self.F.lex.v(h) for h in heads]
            objs = [self.F.pdp.v(obj) for prep in heads for obj in self.E.prep_obj.f(prep)]
            basis_tokens = '|'.join(f'{prep}_{obj}' for prep, obj in zip(preps, objs))
            basis_tokens = basis_tokens or '|'.join(self.F.lex.v(h) for h in heads)
            return (f'{target_funct}.{function}.{basis_tokens}',)

        else:
            return tuple(f'{target_funct}.{function}.{self.F.pdp.v(w)}' for w in heads)
        
class VerbSubjOnlyMinLex(VerbMinLexBasis2):
    
    '''
    In this experiment, only subjects
    are examined as basis elements.
    '''
    
    def __init__(self, tf_api=None):
        super().__init__(tf_api=tf_api)
        
    def config(self):
        '''
        Experiment Configurations
        '''
        self.min_target_freq = 0
        self.min_observation_freq = 0
        self.target2basis = {
                                ('Pred', 'PreO', 'PreS', 'PtcO'):
                                    {('Subj',): self.make_adverbial_bases},  
                            }
        
class VerbObjOnlyMinLex(VerbMinLexBasis2):
    
    '''
    In this experiment, only objects
    are examined as basis elements.
    '''
    
    def __init__(self, tf_api=None):
        super().__init__(tf_api=tf_api)
        
    def config(self):
        '''
        Experiment Configurations
        '''
        self.min_target_freq = 0
        self.min_observation_freq = 0
        self.target2basis = {
                                ('Pred', 'PreO', 'PreS', 'PtcO'):
                                    {('Objc',): self.make_adverbial_bases},  
                            }

class VerbCmplOnlyMinLex(VerbMinLexBasis2):
    
    '''
    In this experiment, only complements
    are examined as basis elements.
    
    In this version, Loca (location) and Time
    are excluded as complementizers to eliminate 
    looser connections based less on semantic class
    than on similarity of temporal/locative context.
    '''
    
    def __init__(self, tf_api=None):
        super().__init__(tf_api=tf_api)
        
    def config(self):
        '''
        Experiment Configurations
        '''
        self.min_target_freq = 0
        self.min_observation_freq = 0
        self.target2basis = {
                                ('Pred', 'PreO', 'PreS', 'PtcO'):
                                    {('PrAd', 'Adju', 'Cmpl'): self.make_adverbial_bases},  
                            }
        
