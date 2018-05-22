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
                 '~/github/semantics/project_code/lingo/heads/tf/c',
                 '~/github/semantics/project_code/sdbh']

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
                          for target in self.get_heads(phrase)
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
                        typ sp sem_domain sem_domain_code
                      ''', silent=True)
        
        self.tf_api = tf_api
            
    '''
    / Target Parameters & Methods /
    ''' 
        
    def get_heads(self, phrase):
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
        
        self.target = target
        
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
        
        bases = collections.Counter()
        
        # make the phrase-level basis elements
        for phrase in phrase_bases:
            basis_function = F.function.v(phrase)
            phrase_bgroup = next((k for k in self.target2basis[phrase_tgroup].keys() if basis_function in k), 0)     
            basis_constructor = self.target2basis[phrase_tgroup][phrase_bgroup]
            these_bases = basis_constructor(phrase, target)
            bases.update(these_bases)

        # make the subphrase-level basis elements
        for subphrase in subphrase_bases:
            basis_rela = F.rela.v(subphrase)
            subphrase_bgroup = next((k for k in self.target2basis[subphrase_tgroup].keys() if basis_rela in k), 0)     
            basis_constructor = self.target2basis[subphrase_tgroup][subphrase_bgroup]
            these_bases = basis_constructor(subphrase, target)
            bases.update(these_bases)
        
        return bases
    
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
        
        restricted_lexemes = {'HJH[', self.F.lex.v(self.target)}
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
        
    def make_predicate_basis(self, basis_phrase, target):
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
        target_phrase = self.L.u(target, 'phrase')[0]
        target_function = self.F.function.v(target_phrase)
        
        return collections.Counter((f'{target_function}.Pred.{lex}.{stem}',))       
     
    def make_noun_basis(self, basis_phrase, target):  
        '''
        Maps a noun to a string basis element, 
        where:
            noun_basis = target_function + noun_token
            
        --input--
        phrase node, target function string
        
        --output--
        basis element string(s)
        '''
        
        target_phrase = self.L.u(target, 'phrase')[0]
        target_function = self.F.function.v(target_phrase)
        
        if self.F.typ.v(basis_phrase) == 'NP':
            bases_tokens = [make_basis_token(h) for h in self.E.heads.f(phrase)]
        
        elif self.F.typ.v(basis_phrase) == 'PP':
            bases_tokens = [make_basis_token(obj) for prep in self.E.heads.f(phrase)
                               for obj in self.E.prep_obj.f(prep)]
            
        return collections.Counter(f'{target_function}.{token}' for token in bases_tokens)
    
    def make_basis_token(self, basis):
        '''
        Makes a basis token out of a word node.
        
        --input--
        word node
        
        --output--
        basis token string
        '''
        
        return self.F.lex.v(basis)
        
    def make_coordinate_noun_basis(self, basis_subphrase, target):
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
        
        target_phrase = self.L.u(target, 'phrase')[0]
        target_function = self.F.function.v(target_phrase)
        
        # TO-DO: Fix this workaround properly. Should I have a 
        # method with parameters on coordinate noun selection?
        
        try:
            head = next(find_quantified(w, self.tf_api) or w
                        for w in self.L.d(basis_subphrase, 'word')
                        if self.F.pdp.v(w) == 'subs')
            return collections.Counter((f'{target_function}.coor.{self.F.lex.v(head)}',))
        
        except StopIteration:
            return {}
      
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
        self.min_target_freq = 7
        self.min_observation_freq = 7
        self.target2basis = {
                                ('Pred', 'PreO', 'PreS', 'PtcO'):
                                    {('PrAd', 'Adju', 'Cmpl', 'Loca', 'Time', 'Objc', 'Subj'): self.make_adverbial_bases},  
                            }
        
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
    
    def make_adverbial_bases(self, phrase, target):
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
        target_funct = self.F.function.v(self.L.u(target, 'phrase')[0])
        
        if self.F.typ.v(phrase) == 'PP':
            preps = [self.F.lex.v(h) for h in heads]
            objs = [self.F.lex.v(obj) for prep in heads for obj in self.E.prep_obj.f(prep)]
            basis_tokens = '|'.join(f'{prep}_{obj}' for prep, obj in zip(preps, objs))
            if basis_tokens:
                return collections.Counter((f'{target_funct}.{function}.{basis_tokens}',))
            else:
                return dict()
            
        else:
            return collections.Counter(f'{target_funct}.{function}.{self.F.lex.v(w)}' for w in heads)
    
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
            return collections.Counter((f'{target_funct}.{function}.{basis_tokens}',))

        else:
            return collections.Counter(f'{target_funct}.{function}.{self.F.lex.v(w)}' for w in heads)
        
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
            return collections.Counter((f'{target_funct}.{function}.{basis_tokens}',))

        else:
            return collections.Counter(f'{target_funct}.{function}.{self.F.pdp.v(w)}' for w in heads)
        
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
        
class CompositeVerb(VerbExperiment1):
    
    '''
    In this experiment, we use the 
    results of a noun vector space to enhance
    a verb space. This is done by adding
    the similarity value of all similar terms
    of a given basis word. The similarity values
    are normalized so that the total value is no 
    greater than 1. This distributes the meaning 
    of the cooccurring word across all of its similar
    terms. 
    '''
    
    def __init__(self, norm_sim_matrix, tf_api=None):
        
        '''
        norm_sim_matrix is a similarity
        matrix wherein the similarity ratios
        have been normalized per target word
        '''
        
        self.sim_words = norm_sim_matrix.to_dict()
        super().__init__(tf_api=tf_api)
        
        
    def make_adverbial_bases(self, phrase, target):
        '''
        Builds a basis string from a supplied
        adverbial phrase. Treats prepositional
        phrases different from other phrase types.
        
        **Special: Uses the results of a 
        normalized similarity matrix to pull
        counts for all words which are similar 
        to the basis element.
        
        --input--
        basis phrase node
        
        --output--
        dictionary of basis counts
        '''
        target_funct = self.F.function.v(self.L.u(target, 'phrase')[0])
        sim_words = self.sim_words
        function = self.F.function.v(phrase)
        heads = self.E.heads.f(phrase)
        bases = collections.Counter()

        if self.F.typ.v(phrase) == 'PP': # modification needed for prepositions, take first head only
            head_obj = self.E.prep_obj.f(heads[0])
            if not head_obj:
                return(bases)
            heado_lex = self.F.lex.v(head_obj[0])
            if heado_lex not in sim_words:
                return(bases)
            prep_lex = self.F.lex.v(heads[0])
            bases_values = dict((f'{target_funct}.{function}.{prep_lex}_{sw}', value) for sw, value in sim_words[heado_lex].items())
            bases.update(bases_values)
            
        else:
            for head in heads:
                head_lex = self.F.lex.v(head)
                if head_lex not in sim_words:
                    continue
                bases_values = dict((f'{target_funct}.{function}.{sw}', value) for sw, value in sim_words[head_lex].items())
                bases.update(bases_values)
                
        return bases
    
class CompositeVerbSubObj(CompositeVerb):
    
    '''
    In this version of the composite
    experiment, only subjects and
    objects are selected as basis elements.
    '''
    
    def __init__(self, sim_matrix, tf_api=None):
        super().__init__(sim_matrix, tf_api=tf_api)
        
    def config(self):
        '''
        Experiment Configurations
        '''
        self.min_target_freq = 7
        self.min_observation_freq = 7
        self.target2basis = {
                                ('Pred', 'PreO', 'PreS', 'PtcO'):
                                    {('Subj', 'Objc'): self.make_adverbial_bases},  
                            }
        
class CompositeVerbSubj(CompositeVerb):
    
    '''
    In this version of the composite
    experiment, only subjects and
    objects are selected as basis elements.
    '''
    
    def __init__(self, sim_matrix, tf_api=None):
        super().__init__(sim_matrix, tf_api=tf_api)
        
    def config(self):
        '''
        Experiment Configurations
        '''
        self.min_target_freq = 7
        self.min_observation_freq = 7
        self.target2basis = {
                                ('Pred', 'PreO', 'PreS', 'PtcO'):
                                    {('Subj',): self.make_adverbial_bases},  
                            }
        
class VerbFrame(VerbExperiment1):
    
    '''
    Count verbs' subcategorization frames.
    '''
    
    def __init__(self, 
                 tf_api=None, 
                 with_lex=False, 
                 bases=('PrAd', 'Adju', 'Cmpl', 'Loca', 'Time', 'Objc',)):
        
        self.with_lex = with_lex
        self.bases = bases
        super().__init__(tf_api=tf_api)
        
    def config(self):
        '''
        Experiment Configurations
        '''
        self.min_target_freq = 7
        self.min_observation_freq = 7
        self.target2basis = {
                                ('Pred', 'PreO', 'PreS', 'PtcO'):
                                    {self.bases: self.make_adverbial_bases}
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
        
        return all([language == 'Hebrew', 
                    freq >= self.min_target_freq,
                    pdp == 'verb'])
    
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
        phrase_tgroup = next((k for k in self.target2basis.keys() if target_funct in k), 0)        
        phrase_bases = [phrase for phrase in clause_phrases
                            if self.basis_phrase_parameters(phrase, phrase_tgroup)]

        # return the (sorted) counts
        bases = self.detach_suffix(target_funct) +\
                sorted(self.make_adverbial_bases(phrase) for phrase in phrase_bases) 
        frame = '|'.join(b for b in bases if b)
        return {frame : 1}
    
    def detach_suffix(self, target_funct):
        '''
        Converts a PreO or PreS verb tag to Pred + subs.
        The conversion is made to "subs" even though the
        object is suffixed. This is so that these forms
        are considered together with word-level subjects
        and objects.
        
        --input--
        target function
        
        --output--
        list of frame (phrase function) strings
        '''
        
        convert = {'PreS': ['Pred', 'Subj.subs'],
                   'PreO': ['Pred', 'Objc.subs'],
                   'PtcO': ['Pred', 'Objc.subs']
                  }
        
        return convert.get(target_funct, ['Pred'])
        
    
    def make_adverbial_bases(self, phrase):
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
        token = self.F.pdp.v if not self.with_lex else self.F.lex.v
        
        if self.F.typ.v(phrase) == 'PP' and function != 'Objc' and heads:
            prep = self.F.lex.v(heads[0])
            #prep_obj = [token(obj) for obj in self.E.prep_obj.f(heads[0])][0]\
            #                if self.E.prep_obj.f(heads[0]) else ''
            #prep_obj = prep_obj if prep_obj != 'nmpr' else 'subs'
            #basis_token = f'{prep}_{prep_obj}' if prep_obj else f'{prep}'
            return f'{function}.{prep}'

        elif self.F.typ.v(phrase) == 'PP' and function == 'Objc' and heads:
            prep = heads[0]
            prep_obj = self.E.prep_obj.f(prep)
            if prep_obj:
                this_token = token(prep_obj[0]) if token(prep_obj[0]) != 'nmpr' else 'subs'
                return f'{function}.{this_token}'
            else:
                return ''
        
        elif heads:
            this_token = token(heads[0]) if token(heads[0]) != 'nmpr' else 'subs'
            return f'{function}.{this_token}'
        
        else:
            return ''
        
        
class VerbFrameSemantic(VerbFrame):
    
    '''
    Count verbs' subcategorization frames with
    enhanced semantic data from the Semantic
    Dictionary of Biblical Hebrew (De Blois, UBS)
    
    Excludes verbs with subject/object suffixes 
    since these require participant tracking data.
    
    Minimum word/observation occurrences are set to 10.
    '''
    
    
    
    def __init__(self, 
                 tf_api=None, 
                 with_lex=False, 
                 bases=('Subj', 'PrAd', 'Adju', 'Cmpl', 'Loca', 'Time', 'Objc',)):
        
        self.with_lex = with_lex
        self.bases = bases
        super().__init__(tf_api=tf_api, bases=bases)
        
    def config(self):
        '''
        Experiment Configurations
        '''
        self.min_target_freq = 10
        self.min_observation_freq = 10
        self.target2basis = {
                                ('Pred',):
                                    {self.bases: self.make_adverbial_bases}
                            }
        
    def filter_semcodes(self, wordnode):
        '''
        Retrieves and separates semantic
        codes from SDBH.
        
        --input--
        code string
        
        --output--
        selected code or None
        '''
        
        if not self.F.sem_domain_code.v(wordnode):
            return None
        
        # code filter
        codes = [code for code in self.F.sem_domain_code.v(wordnode).split('|')
                    if code in set(f'1.00100{i}' for i in range(1, 7)) | {'1.002004'}
                ]
        
        if codes:
            return codes[0]
        
        else:
            return None
        
    def code2tag(self, sem_domain_code):
        '''
        Maps a semantic domain code
        to a generalized category.
        
        --input--
        domain code string
        
        --output--
        generalized category string or None
        '''
        
        if not sem_domain_code:
            return None
        
        code = sem_domain_code[:8]
        
        if code == '1.001001':
            return 'animate'
        
        elif code in set(f'1.00100{i}' for i in range(2, 7)):
            return 'inanimate'
        
        elif code == '1.002004':
            return 'event'
        
        else:
            return None
        
    def make_adverbial_bases(self, phrase):
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
        token = self.filter_semcodes
        
        # for prepositional phrases
        prep_objs = [token(obj) for head in heads
                        for obj in self.E.prep_obj.f(head)]
        prep_objs = [self.code2tag(code) for code in prep_objs] or [None]

        # format non-object prepositional phrases 
        if all([self.F.typ.v(phrase) == 'PP',
                function != 'Objc', 
                heads,
                all(prep_objs)]):
            
            preps = [self.F.lex.v(head) for head in heads]
            basis_tokens = '|'.join(f'{prep}_{obj}' for prep, obj in zip(preps, prep_objs))
            basis_token = f'{function}.{basis_tokens}'
            
            return basis_token
        
        # format object prep. phrases
        elif all([self.F.typ.v(phrase) == 'PP',
                  function == 'Objc',
                  heads,
                  all(prep_objs)]):
            
            return f'{function}.{prep_objs[0]}'
        
        # format everything else
        elif heads and token(heads[0]):
            this_token = self.code2tag(token(heads[0]))
            return f'{function}.{this_token}'
        
        else:
            return ''
        