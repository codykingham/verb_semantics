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
for selecting and formatting the BHSA Hebrew data.

*NB*
To properly run this module on your own system, you must set the BHSA data paths 
upon initializing Experiment. say:
    Experiment(bhsa_data_paths = [PATH_TO_BHSA_CORE, PATH_TO_ETCBC_LINGO_HEADS])
'''

import collections, os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.metrics.pairwise import pairwise_distances

if not __package__:
    from kmedoids.kmedoids import kMedoids
    from helpers import get_lex
else:
    from .kmedoids.kmedoids import kMedoids
    from .helpers import get_lex


class SemSpace:
    '''
    This class brings together all of the methods defined in 
    notebook 4 of my semantics repository. This class has as
    number of attributes which contain cooccurrence counts,
    adjusted counts, similarity scores, and more.
    The class loads and processes the data in one go.
    '''
    
    def __init__(self, experiment, info=True, test=False, run_ll=False, verb_space=True):
        '''
        Requires an experiment class (defined below).
        This allows various experiment parameters
        to be feed to the semantic space builder.
        
        "info" is an optionally divisable number
        if status updates are desired for lengthy
        calculations.
        
        "test" builds a semantic space on the first 100
        elements of the experiment.
        
        "run_ll" toggles whether to run the log-likelihood
        adjustment, which is more time consuming for large
        data, and which is not being used much in this project anymore.
        '''
        
        self.tf_api = experiment.tf_api
        self.info = experiment.tf_api.info
        self.indent = experiment.tf_api.indent
        self.report = info
        self.experiment = experiment
        F = experiment.tf_api.F
        data = experiment.data if not test else experiment.data.head(100)[experiment.data.columns[:100]]
        self.raw = data
        row_col = [f'{w} ({experiment.target2gloss[w]})' for w in self.raw.columns] # for similarity matrix indices/cols


        if self.report:
            self.indent(0, reset=True)
            self.info('Applying association measure(s)...')
            self.indent(1, reset=True)
        
        # adjust raw counts with log-likelihood & pointwise mutual information
        if run_ll:
            if self.report:
                self.indent(0)
                self.info('Beginning Loglikelihood calculations...')
                self.indent(1)
            self.loglikelihood = precomputed['loglikelihood'] if 'loglikelihood' in precomputed\
                                     else self.apply_loglikelihood(data)
            if self.report:
                self.indent(1, reset=True)
                self.info('Building all LL matrices...')
            self.pca_ll = self.apply_pca(self.loglikelihood)
            self.pairwise_ll = pairwise_distances(self.loglikelihood.T.values, metric='cosine')
            self.pairwise_ll_pca = pairwise_distances(self.pca_ll, metric='euclidean')
            self.distance_ll = pd.DataFrame(self.pairwise_ll, columns=row_col, index=row_col)
            self.distance_ll_pca = pd.DataFrame(self.pairwise_ll_pca, columns=row_col, index=row_col)
            self.similarity_ll = self.distance_ll.apply(lambda x: 1-x)
            self.ll_plot = PlotSpace(self.pca_ll, self.loglikelihood, self.tf_api, experiment)
        
        if self.report:
            self.indent(1)
            self.info('Applying PPMI...')
            
        self.pmi = self.get_pmi(data)
        
        if self.report:
            self.indent(1)
            self.info('Finished PPMI...')
            self.indent(0)
            self.info('Building pairwise matrices...')
        
        
        # apply pca or other data maneuvers
        self.pca_pmi = self.apply_pca(self.pmi)
        self.pca_raw = self.apply_pca(data)
        # TODO: self.raw_norm = ....
        
        # pairwise distances
        if self.report:
            self.indent(1, reset=True)
            self.info('Building pairwise distances...')
        self.pairwise_pmi = pairwise_distances(self.pmi.T.values, metric='cosine')
        self.pairwise_raw = pairwise_distances(self.raw.T.values, metric='cosine')
        self.pairwise_pmi_pca = pairwise_distances(self.pca_pmi, metric='euclidean')
        self.pairwise_raw_pca = pairwise_distances(self.pca_raw, metric='euclidean')
        self.pairwise_jaccard = pairwise_distances((self.raw > 0).T.values, metric='jaccard')
        
        # distance matrices
        self.distance_pmi = pd.DataFrame(self.pairwise_pmi, columns=row_col, index=row_col)
        self.dist_pmi_nogloss = pd.DataFrame(self.pairwise_pmi, columns=self.pmi.columns, index=self.pmi.columns)
        self.dist_raw_nogloss = pd.DataFrame(self.pairwise_raw, columns=self.raw.columns, index=self.raw.columns)
        self.distance_raw = pd.DataFrame(self.pairwise_raw, columns=row_col, index=row_col)
        self.distance_pmi_pca = pd.DataFrame(self.pairwise_pmi_pca, columns=row_col, index=row_col)
        self.distance_raw_pca = pd.DataFrame(self.pairwise_raw_pca, columns=row_col, index=row_col)
        self.distance_jaccard = pd.DataFrame(self.pairwise_jaccard, columns=row_col, index=row_col)
        
        # similarity matrices
        if self.report:
            self.info('Building pairwise similarities...')
        self.similarity_pmi = self.distance_pmi.apply(lambda x: 1-x)
        self.sim_pmi_nogloss = self.dist_pmi_nogloss.apply(lambda x: 1-x)
        self.sim_pmi_normalized = self.sim_pmi_nogloss / self.sim_pmi_nogloss.sum()
        self.similarity_raw = self.distance_raw.apply(lambda x: 1-x)
        self.sim_raw_nogloss = self.dist_raw_nogloss.apply(lambda x: 1-x)
        self.similarity_jaccard = self.distance_jaccard.apply(lambda x: 1-x)
        
        # space plots
        verb_functs = {'Pred', 'PreO', 'PreS', 'PtcO'} # format plots for verbs if space is verb space (add stem to gloss)
        self.pmi_plot = PlotSpace(self.pca_pmi, self.pmi, self.tf_api, experiment, verb_space=verb_space)
        self.raw_plot = PlotSpace(self.pca_raw, self.raw, self.tf_api, experiment, verb_space=verb_space)
        
        if self.report:
            self.indent(0)
            self.info('space is ready!')
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

    def apply_loglikelihood(self, comatrix):

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
        
        for target in comatrix.columns:
            for basis in comatrix.index:
                k = comatrix[target][basis]
                if not k:
                    continue
                l = comatrix.loc[basis].sum() - k
                m = comatrix[target].sum() - k
                n = comatrix.values.sum() - (k+l+m)
                ll = self.loglikelihood(k, l, m, n, safe_log)
                new_matrix[target][basis] = ll
                
        if self.report:
            self.indent(0)
            self.info(f'FINISHED loglikelihood at iteration {i}')
        
        return new_matrix
        
    def get_pmi(self, datamatrix):
        '''
        Apply PMI to a given column.
        Algorithm derived from 
        Levshina 2015, Linguistics with R, 327.
        Credit: Etienne van de Bijl, 02.05.18
        '''
        n = len(datamatrix.columns)
        sum_r = datamatrix.sum(1)
        expected = sum_r/n
        expected = expected.replace(0, np.nan)
        datamatrix = datamatrix.replace(0, np.nan)
        pmi = np.log2(datamatrix.div(expected, axis=0))
        pmi[pmi<0] = 0
        return pmi.fillna(0)
    
    '''
    // Data Transformations and Cluster Analyses //
    '''

    def apply_pca(self, comatrix, n=100):
        '''
        Apply principle component analysis to a supplied cooccurrence matrix.
        '''
        n_components = n if comatrix.shape[0] > n else comatrix.shape[0] - 1
        pca = PCA(n_components=n_components)
        return pca.fit_transform(comatrix.T.values)
    
    def apply_sparse_pca(self, comatrix, n=10):
        '''
        Apply principle component analysis to a supplied cooccurrence matrix.
        '''
        pca = SparsePCA(n_components=n)
        return pca.fit_transform(comatrix.T.values)
    
    '''
    // Plotting and Visualizations //
    '''

class PlotSpace:
    '''
    A simple visualization class that visualizes
    a semantic space with PCA and with input data.
    '''
    def __init__(self, pca_arrays, matrix, tf_api, experiment, verb_space=False):
        self.pca_arrays = pca_arrays
        self.matrix = matrix
        self.api = tf_api
        self.F = tf_api.F
        self.target2gloss = experiment.target2gloss
        self.target2node = experiment.target2node
        self.verb_space = verb_space
            
    def show(self, size=(10, 6), annotate=True, title='', axis=[], principal_components=(0, 1)):
        0
        '''
        Shows the requested plot.
        '''
        pc1, pc2 = principal_components
        plt.figure(1, figsize=size)
        plt.scatter(self.pca_arrays[:, pc1], self.pca_arrays[:, pc2])
        plt.title(title)
        if axis:
            plt.axis(axis)
        if annotate:
            annotator = self.annotate_space if not self.verb_space else self.annotate_verb_space
            annotator(principal_components)
            
    def annotate_space(self, principal_components):
        '''
        Annotates PCA scatter plots with word lexemes.
        '''
        pc1, pc2 = principal_components
        words = [f'{self.target2gloss[l]}' for l in self.matrix.columns]
        for i, word in enumerate(words):
            plt.annotate(word, xy=(self.pca_arrays[i, pc1], self.pca_arrays[i, pc2]))
    
    def annotate_verb_space(self, principal_components):
        '''
        Annotates PCA scatter plots with verb lexemes + stem.
        '''
        pc1, pc2 = principal_components
        words = [f'{self.target2gloss[l]}.{self.F.vs.v(self.target2node[l])}' for l in self.matrix.columns]
        for i, word in enumerate(words):
            plt.annotate(word, xy=(self.pca_arrays[i, pc1], self.pca_arrays[i, pc2]))