'''
This module contains code used for constructing and evaluating a semantic space.
The space should consist of a Pandas dataframe with target words as columns;
the co-occurrences are stored as rows.

The module also contains an algorithm for selecting data from the BHSA,
the ETCBC's Hebrew Bible syntax data.
Several selection restrictions are applied in the preparation of that data.

A single class, SemSpace, contains all of the methods combined and can be used for
evaluations and analyses.

Most of the code in this module is derived from notebook 4 in my semantics repo.
'''

import collections, os, math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from tf.fabric import Fabric
os.sys.path.append('~/github/kmedoids')

class SemSpace:
    '''
    This class brings together all of the methods defined in 
    notebook 4 of my semantics repository. This class has a
    number of attributes which contain cooccurrence counts,
    adjusted counts, similarity scores, and more.
    The class loads and processes the data in one go.
    '''
    
    def __init__(self):
        pass
    
    # BHSA related functions
        
    def load_tf_bhsa(self):
        '''
        Loads the TF instance of the BHSA dataset.
        '''
        TF = Fabric(locations='~/github', modules=['etcbc/bhsa/tf/c', 'semantics/tf/c'],
                    silent=True)
        api = TF.load('''
                        book chapter verse
                        function lex vs language
                        pdp freq_lex gloss domain ls
                        heads
                      ''', silent=True)
        return api
        
    def get_lex(self, lex_string):
        '''
        Return ETCBC lex node number from a lexeme string.
        '''
        lex = next(lex for lex in F.otype.s('lex') if F.lex.v(lex) == lex_string)
        return lex
        
    def get_bhsa(self, tf_api):
        '''
        Retrieves BHSA cooccurrence data.
        The function contains numerous selection restrictions
        which can be edited as needed.
        For example: only narrative clauses are selected,
        but discourse can be included as well.

        Returns a Pandas dataframe with cooccurrence data.
        Requires TF api loaded with BHSA data and the appropriate features.
        '''

        # define shortform tf api method names
        F, E, L = tf_api.F, tf_api.E, tf_api.L

        # put raw cooccurrence data here
        cooccurrences = collections.defaultdict(lambda: collections.Counter()) # noun counts here

        # Subj/Objc Counts
        for phrase in F.otype.s('phrase'):

            # skip non-Hebrew sections
            language = F.language.v(L.d(phrase, 'word')[0]) 
            if language != 'Hebrew':
                continue

            # skip non subject/object phrases
            function = F.function.v(phrase)
            if function not in {'Subj', 'Objc'}:
                continue

            # get head nouns
            nouns = set(F.lex.v(w) for w in E.heads.f(phrase)) # count lexemes only once
            if not nouns:
                continue

            # restrict on frequency
            freq = [F.freq_lex.v(L.u(w, 'lex')[0]) for w in E.heads.f(phrase)]
            if min(freq) < 8:
                continue

            # restrict on proper names
            pdps = set(F.pdp.v(w) for w in E.heads.f(phrase))
            ls = set(F.ls.v(w) for w in E.heads.f(phrase))
            if {'nmpr', 'gntl'} & set(pdps|ls):
                continue

            # restrict on domain
            if F.domain.v(L.u(phrase, 'clause')[0]) != 'N':
                continue

            # gather contextual data
            clause = L.u(phrase, 'clause')[0]
            good_paths = path_weights[function]
            paths = [phrase for phrase in L.d(clause, 'phrase')
                        if F.function.v(phrase) in good_paths.keys()
                    ]

            # make the counts
            for path in paths:

                pfunct = F.function.v(path)
                weight = good_paths[pfunct]

                # count for verb
                if pfunct == 'Pred':
                    verb = [w for w in L.d(path, 'word') if F.pdp.v(w) == 'verb'][0]
                    verb_lex = F.lex.v(verb)
                    verb_stem = F.vs.v(verb)
                    verb_basis = function + '.' + verb_lex + '.' + verb_stem # with function name added
                    if verb and F.lex.v(verb) not in {'HJH['}: # omit "to be" verbs, others?
                        for noun in nouns:
                            cooccurrences[noun][verb_basis] += 1

                # count for subj/obj
                else:
                    conouns = E.heads.f(path)
                    cnoun_bases = set(function + '.' + F.lex.v(w) + f'.{pfunct}' for w in conouns) # with function name added
                    counts = dict((basis, weight) for basis in cnoun_bases)
                    if counts:
                        for noun in nouns:
                            cooccurrences[noun].update(counts)

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

        cooccurrences = pd.DataFrame(cooccurrences).fillna(0)

        return cooccurrences
    
    def apply_pca(self, comatrix):
        '''
        Apply principle component analysis to a supplied cooccurrence matrix.
        '''
        pca = PCA(n_components=2)
        return pca.fit_transform(cooccurrences.T.values)
        
    def k_medoid_clusters(self, comatrix):
        '''
        Apply k-medoid clustering to a comatrix.
        '''
    
    
class AssociationMeasures:
    '''
    Contains code for applying association measures 
    to supplied cooccurrence data. The measures are 
    written based on various sources including Padó & Lapita (2007)
    and Levshina (2015).
    '''
    
    def __init__(self, raw_cooccurrence_matrix):
        self.loglikelihood = self.apply_loglikelihood(raw_cooccurrence_matrix)
        self.pmi = self.apply_pmi(raw_cooccurrence_matrix)
    
    # functions for loglikelihood (Padó & Lapita 2007)
    
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
        '''
        p1 = (k*log(k)) + (l*log(l)) + (m*log(m)) + (n*log(n))        
        p2 = ((k+l)*log(k+l)) - ((k+m)*log(k+m))
        p3 = ((l+n)*log(l+n)) - ((m+n)*log(m+n))
        p4 = ((k+l+m+n))*log(k+l+m+n)
        llikelihood = 2*(p1-p2-p3+p4)
        return llikelihood

    def apply_loglikelihood(self, comatrix):

        '''
        Adjusts values in a cooccurrence matrix using log-likelihood. 
        Requires a cooccurrence matrix.
        
        An option for progress updates is commented out.
        This option can be useful for very large datasets
        that take more than a few seconds to execute.
        i.e. sets that contain target words > ~500
        '''
        safe_log = self.safe_log
        log_likelihood = self.loglikelihood
        
        new_matrix = comatrix.copy()
        #optional: information for large datasets
        #i = 0 
        #indent(reset=True)
        #info('beginning calculations...')
        #indent(1, reset=True)
        for target in comatrix.columns:
            for basis in comatrix.index:
                k = comatrix[target][basis]

                if not k:
                    #i += 1
                    #if i % 500000 == 0:
                        #indent(1)
                        #info(f'at iteration {i}')
                    continue

                l = comatrix.loc[basis].sum() - k
                m = comatrix[target].sum() - k
                n = comatrix.values.sum() - (k+l+m)
                ll = loglikelihood(k, l, m, n, safe_log)
                new_matrix[target][basis] = ll
                
                # optional: information for large datasets
                #i += 1
                #if i % 500000 == 0:
                    #indent(1)
                    #info(f'at iteration {i}')
        #indent(0)
        #info(f'FINISHED at iteration {i}')
        return new_matrix
    
    # PMI method (Levshina 2015)
    
    def apply_pmi_col(self, col):

        '''
        Apply PMI to a given column.
        '''
        expected = col * cooccurrences.sum(axis=1) / cooccurrences.values.sum()
        pmi = np.log(col / expected).fillna(0)
        return pmi
    
    def apply_pmi(self, datamatrix):
        '''
        apply pmi to a data matrix
        '''
        return datamatrix.apply(lambda k: self.apply_pmi(k))