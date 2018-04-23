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
    
    def __init__(self, experiment, info=0, test=False):
        '''
        Requires an experiment class (defined below).
        This allows various experiment parameters
        to be feed to the semantic space builder.
        
        "info" is an optionally divisable number
        if status updates are desired for lengthy
        calculations.
        '''
        
        self.tf_api = experiment.tf_api
        self.info = experiment.tf_api.info
        self.indent = experiment.tf_api.indent
        self.report = info
        self.experiment = experiment
        F = experiment.tf_api.F
        data = experiment.data if not test else experiment.data.head(10)
        
        if self.report:
            self.indent(reset=True)
            self.info('Beginning all calculations...')
            self.indent(1, reset=True)
        
        # adjust raw counts with log-likelihood & pointwise mutual information
        self.loglikelihood = self.apply_loglikelihood(data)
        self.pmi = self.apply_pmi(data)
        self.raw = data
        
        if self.report:
            self.indent(0)
            self.info('FINISHED PMI...')
        
        # apply pca or other data maneuvers
        self.pca_ll = self.apply_pca(self.loglikelihood)
        self.pca_ll_3d = self.apply_pca(self.loglikelihood, n=3)
        self.pca_pmi = self.apply_pca(self.pmi)
        self.pca_raw = self.apply_pca(data)
        
        # pairwise distances
        self.pairwise_ll = pairwise_distances(self.loglikelihood.T.values, metric='cosine')
        self.pairwise_pmi = pairwise_distances(self.pmi.T.values, metric='cosine')
        self.pairwise_raw = pairwise_distances(self.raw.T.values, metric='cosine')
        self.pairwise_ll_pca = pairwise_distances(self.pca_ll, metric='euclidean')
        self.pairwise_pmi_pca = pairwise_distances(self.pca_pmi, metric='euclidean')
        self.pairwise_raw_pca = pairwise_distances(self.pca_raw, metric='euclidean')
        self.pairwise_jaccard = pairwise_distances((self.raw > 0).T.values, metric='jaccard')
        
        # distance matrices
        row_col = [f'{w} ({experiment.target2gloss[w]})' for w in self.raw.columns]
        self.distance_ll = pd.DataFrame(self.pairwise_ll, columns=row_col, index=row_col)
        self.distance_pmi = pd.DataFrame(self.pairwise_pmi, columns=row_col, index=row_col)
        self.distance_raw = pd.DataFrame(self.pairwise_raw, columns=row_col, index=row_col)
        self.distance_ll_pca = pd.DataFrame(self.pairwise_ll_pca, columns=row_col, index=row_col)
        self.distance_pmi_pca = pd.DataFrame(self.pairwise_pmi_pca, columns=row_col, index=row_col)
        self.distance_raw_pca = pd.DataFrame(self.pairwise_raw_pca, columns=row_col, index=row_col)
        self.distance_jaccard = pd.DataFrame(self.pairwise_jaccard, columns=row_col, index=row_col)
        
        # similarity matrices
        self.similarity_ll = self.distance_ll.apply(lambda x: 1-x)
        self.similarity_pmi = self.distance_pmi.apply(lambda x: 1-x)
        self.similarity_raw = self.distance_raw.apply(lambda x: 1-x)
        self.similarity_jaccard = self.distance_jaccard.apply(lambda x: 1-x)
        
        # space plots
        self.ll_plot = PlotSpace(self.pca_ll, self.loglikelihood, self.tf_api, experiment.target2gloss)
        self.pmi_plot = PlotSpace(self.pca_pmi, self.pmi, self.tf_api, experiment.target2gloss)
        self.raw_plot = PlotSpace(self.pca_raw, self.raw, self.tf_api, experiment.target2gloss)
        
        if self.report:
            self.indent(0)
            self.info('data gathering complete!')
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
        
        i = 0 
        if self.report:
            self.indent(0)
            self.info('beginning Loglikelihood calculations...')
            self.indent(1, reset=True)
        
        for target in comatrix.columns:
            for basis in comatrix.index:
                k = comatrix[target][basis]

                if not k:
                    i += 1
                    if self.report and i % self.report == 0:
                        self.indent(1)
                        self.info(f'at iteration {i}')
                    continue

                l = comatrix.loc[basis].sum() - k
                m = comatrix[target].sum() - k
                n = comatrix.values.sum() - (k+l+m)
                ll = self.loglikelihood(k, l, m, n, safe_log)
                new_matrix[target][basis] = ll
                
                # optional: information for large datasets
                i += 1
                if self.report and i % self.report == 0:
                    self.indent(1)
                    self.info(f'at iteration {i}')
        if self.report:
            self.indent(0)
            self.info(f'FINISHED loglikelihood at iteration {i}')
        
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
        if self.report:
            self.indent(0)
            self.info('beginning PMI calculations...')
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
    
    '''
    // Plotting and Visualizations //
    '''

class PlotSpace:
    '''
    A simple visualization class that visualizes
    a semantic space with PCA and with input data.
    '''
    def __init__(self, pca_arrays, matrix, tf_api, target2gloss):
        self.pca_arrays = pca_arrays
        self.matrix = matrix
        self.api = tf_api
        self.target2gloss = target2gloss
            
    def show(self, size=(10, 6), annotate=True, title='', axis=[]):
        
        '''
        Shows the requested plot.
        '''
    
        plt.figure(1, figsize=size)
        plt.scatter(self.pca_arrays[:, 0], self.pca_arrays[:, 1])
        plt.title(title)
        if axis:
            plt.axis(axis)
        if annotate:
            self.annotate_space()
            
    def annotate_space(self):
        '''
        Annotates PCA scatter plots with word lexemes.
        '''
        
        words = [self.target2gloss[l] for l in self.matrix.columns]
        for i, word in enumerate(words):
            plt.annotate(word, xy=(self.pca_arrays[i, 0], self.pca_arrays[i, 1]))