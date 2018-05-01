# Semantic Space
## Harvesting Word Similarity Sets from BHSA

The goal of this project is to cluster similar nouns in the Hebrew Bible using only syntactic data. That is done with a semantic [vector space model](https://en.wikipedia.org/wiki/Vector_space_model), by which the lexical and syntactic context of words can be compared as [vectors](https://en.wikipedia.org/wiki/Vector_space). 
This project applies an adaptation of Padó and Lapata's ([2007](https://www.mitpressjournals.org/doi/pdf/10.1162/coli.2007.33.2.161)) approach by which syntax is used to restrict co-occurrence selections. This syntactically informed method is contrasted with, for instance, [word2vec](https://radimrehurek.com/gensim/models/word2vec.html), which only uses a selection window (i.e. words that occur within `N` slots). 

The data used for the analysis is the [ETCBC](http://www.etcbc.nl)'s [BHSA](https://github.com/ETCBC/bhsa) in [Text-Fabric representation](https://github.com/Dans-labs/text-fabric/wiki). 

## Phase 2

### Phase Updates

#### 01 May 2018
Based on the output of the verb experiments in this phase, I see that it is not possible to simply build a space for only verbs or only nouns. Rather, these two spaces really need each other if there is any hope to obtain more defined word clusters. A fine example of the problem can be seen in the [phase2_clustering](http://nbviewer.jupyter.org/github/codykingham/semantics/blob/master/phase2_clustering.ipynb) notebook, where the noun יהוה and אלהים occur disproportionately as the subject with verbs like אנף "be angry," מטר "cause rain," and others. In reality, יהוה and אלהים both possess attributes which are common to other nouns, such as the collocability with action verbs. But because the space does not see these similarities, the lexemes are treated as completely separate entities. This is where the noun space can enter in and offer observations about the similarity of nouns with each other. By adding similar words to the space, based on a pre-calculated noun space, it is hoped that the knowledge gap can be bridged and lexemes with similar attributes considered together. This will allow, for instance, the noun אלהים, which also takes verbs like אמר, to be identified with other nouns that take the verb אמר.

### Phase Description

This project is presently entering a phase 2, in which the experience and results from phase 1 will be used to improve the model. Phase 2 aims at four goals in particular:

1) √ &nbsp;&nbsp;&nbsp;&nbsp; Implement a group of Python classes which contain all experiment parameters, and which can be easily modified and saved for back-referencing. This will allow for experiment parameters to be carefully tracked and recorded, and also for multiple kinds of results to be analyzed alongside one another. The classes contain BHSA target word and context selection measures, data transformation, and visualization methods. The BHSA target word and context selection methods break their tasks down into many smaller methods that can easily be exchanged or supplemented with inheriting classes.
2) √ &nbsp;&nbsp;&nbsp;&nbsp; Apply the more robust method of head selection which was designed and saved to the ETCBC [lingo repository](https://github.com/ETCBC/lingo/tree/master/heads). This method can handle any phrase type rather than just subject or object phrases.
3) Experiment with compositional vector spaces as inspired by Schütze 1998. Semantic spaces will be added on top of each other to enhance their accuracy. Specifically, phase 2 will attempt to use verb and noun vectors as cooccurrence bases instead of individual basis elements. This requires in effect two runs: a rudimentary space for both verbs and nouns and a compositional space that uses the rudimentary spaces in a second run. The motivation here is that observations across different, but related, lexemes would be brought to bear in each case. For instance, if a rare verb is used in a given context, but that verb's properities are more related to a common verb, the common verb's vector can serve to disambiguate.
4) To experiment with other statistical methods, especially distance measures. A jaccardian distance measure, for instance, might be a better indicator of a word's combinability with certain other words.
