'''
This module contains the experiment parameters
and data generation used during phase 1 of my 
semantic space project. The code has since been modified
and improved in the new module, semspace.py.
'''

import collections, os, math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from tf.fabric import Fabric
from tf.extra.bhsa import Bhsa

if not __package__:
    from lingo.heads.heads import find_quantified
else:
    from .lingo.heads.heads import find_quantified

class SemSpace:
    '''
    Phase 1 Semantic Space:
    This class brings together all of the methods defined in 
    notebook 4 of my semantics repository (phase 1). This class has a
    number of attributes which contain cooccurrence counts,
    adjusted counts, similarity scores, and more.
    The class loads and processes the data in one go.
    '''
    
    def __init__(self, info=0):
        '''
        Requires an experiment class (defined below).
        This allows various experiment parameters
        to be feed to the semantic space builder.
        
        Change "info" to divisable number
        if status updates are desired for the
        log likelihood calculations.
        '''
        
        # load BHSA experiment data
        tf_api, B = self.load_tf_bhsa()
        cooccurrences = p1_experiment_data(tf_api)
        
        # make TF api available
        self.tf_api = tf_api
        self.B = B
        
        # adjust raw counts with log-likelihood & pointwise mutual information
        self.loglikelihood = self.apply_loglikelihood(cooccurrences)
        self.pmi = self.apply_pmi(cooccurrences)
        self.raw = cooccurrences
        
        # apply pca or other data maneuvers
        self.pca_ll = self.apply_pca(self.loglikelihood)
        self.pca_ll_3d = self.apply_pca(self.loglikelihood, n=3)
        self.pca_pmi = self.apply_pca(self.pmi)
        self.pca_raw = self.apply_pca(cooccurrences)
        
    '''
    -----
    BHSA Methods:
        Methods used to prepare and process the BHSA
        Hebrew Bible data.
    '''
        
    def load_tf_bhsa(self):
        '''
        Loads a TF instance of the BHSA dataset.
        '''
        TF = Fabric(locations='~/github', modules=['etcbc/bhsa/tf/c', 'semantics/phase1/tf/c'], # modify paths here for your system
                    silent=True)
        api = TF.load('''
                        book chapter verse
                        function lex vs language
                        pdp freq_lex gloss domain ls
                        heads
                      ''', silent=True)
        
        B = Bhsa(api, '4. Semantic Space Construction', version='c')
        
        return api, B
        
    def get_lex(self, lex_string):
        '''
        Return ETCBC lex node number from a lexeme string.
        Requires a text fabric feature class with otype/lex features loaded.
        '''
        F = self.tf_api.F
        lex = next(lex for lex in F.otype.s('lex') if F.lex.v(lex) == lex_string)
        return lex
    

    '''
    -----
    Association Measure Methods:
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
        if info:
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
                    
        if info:
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
    Data Transformations and Cluster Analyses:
    '''

    def apply_pca(self, comatrix, n=2):
        '''
        Apply principle component analysis to a supplied cooccurrence matrix.
        '''
        pca = PCA(n_components=n)
        return pca.fit_transform(comatrix.T.values)
    
def p1_experiment_data(tf_api):
    '''
    Retrieves BHSA cooccurrence data based on my first experiment's parameters.
    Returns a Pandas dataframe with cooccurrence data.
    Requires TF api loaded with BHSA data and the appropriate features.

    --experiment 1 parameters--
    • phrase must be a subject/object function phrase
    • language must be Hebrew
    • head nouns extracted from subj/obj phrase w/ (old) E.heads feature
    • minimum noun occurrence frequency is 8
    • proper names and gentilics excluded
    • only nouns from narrative is included
    • weigh 1 for subject -> predicate relations
    • weigh 1 for object -> predicate relations
    • weigh 1 for noun -> coordinate noun relations
    • exclude HJH[ (היה) predicates
    '''

    # shortform text fabric methods
    F, E, L = tf_api.F, tf_api.E, tf_api.L

    # configure weights for path counts
    path_weights = {'Subj': {'Pred': 1,
                            },
                    'Objc': {
                             'Pred': 1,
                            },
                    'coor': 1
                   }

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

    # return final results
    return pd.DataFrame(cooccurrences).fillna(0)

def plot_silhouettes(data_vectors, range_n_clusters, scatter=False, randomstate=10):
    
    '''    
    Plot silhouette plots based on a supplied range of K using a supplied method.
    Can also plot an optional scatter plot if PCA transformed vectors are supplied.
    Code modified from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    '''
    
    pairwise_dists = pairwise_distances(data_vectors, metric='cosine')
    
    for n_clusters in range_n_clusters:
        
        # use method to make the clusters
        medoids, clusters = kmedoids.kMedoids(pairwise_dists, n_clusters, state=randomstate)
    
        # make cluster labels with index corresponding to target word cluster
        cluster_labels = sorted((i, group) for group in clusters # sort groups by index
                                    for i in clusters[group])
        cluster_labels = np.array(list(group[1] for group in cluster_labels)) # put the groups in indexed order in an array
        
        # Create plots
        if not scatter:
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(10, 8)
        else:
            fig, (ax, ax2) = plt.subplots(1,2)
            fig.set_size_inches(18, 7)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(pairwise_dists, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(pairwise_dists, cluster_labels)
                
        # Set x-axis limits
        # The silhouette coefficient can range from -1, 1
        # I set the lower limit a -0.2 since the min with one method is -0.13
        ax.set_xlim([-0.2, 0.06])
        
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax.set_ylim([0, len(pairwise_dists) + (n_clusters + 1) * 10])
        
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.ocean(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor='black', alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title("The silhouette plot for the various clusters.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # plot scatter if PCA data is given
        if scatter:
            colors = cm.ocean(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(data_vectors[:, 0], data_vectors[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = data_vectors[medoids]
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        

        plt.show()
        
def plot_silhouette_scores():
    '''
    OLD CODE
    '''
    sil_scoresX = []
    sil_scoresY = []
    limit = 41

    for n in range(2, limit):

        # make the clusters with n
        medoids, clusters = kmedoids.kMedoids(pca_dists_ll, n, state=random_seed)

        # make cluster labels with index corresponding to target word cluster
        cluster_labels = sorted((i, group) for group in clusters # sort groups by index
                                    for i in clusters[group])
        cluster_labels = np.array(list(group[1] for group in cluster_labels)) # put the groups in indexed order in an array

        # get sil average
        silhouette_avg = silhouette_score(pca_dists_ll, cluster_labels)

        sil_scoresX.append(n)
        sil_scoresY.append(silhouette_avg)

    plt.figure(1, figsize=(10, 5))
    ax = plt.axes()
    ax.set_xticks(range(0, limit))
    plt.plot(sil_scoresX, sil_scoresY)