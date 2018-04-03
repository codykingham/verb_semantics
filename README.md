# Semantic Space
## Harvesting Noun Similarity Sets from BHSA

Read the draft text, "Modeling Noun Semantics with Vector Spaces and the ETCBC Hebrew Database" (above) of my 2018 ETEN conference presentation, describing the rationale of this project.

The goal of this project is to cluster similar nouns in the Hebrew Bible using only syntactic data. That is done with a semantic [vector space model](https://en.wikipedia.org/wiki/Vector_space_model), by which the lexical and syntactic context of words can be compared as [vectors](https://en.wikipedia.org/wiki/Vector_space). 

This project applies an adaptation of Padó and Lapata's ([2007](https://www.mitpressjournals.org/doi/pdf/10.1162/coli.2007.33.2.161)) approach by which syntax is used to restrict co-occurrence selections. This syntactically informed method is contrasted with, for instance, [word2vec](https://radimrehurek.com/gensim/models/word2vec.html), which only uses a selection window (i.e. words that occur within `N` slots). 

The data used for the analysis is the [ETCBC](http://www.etcbc.nl)'s [BHSA](https://github.com/ETCBC/bhsa) in [Text-Fabric representation](https://github.com/Dans-labs/text-fabric/wiki). 

The repository consists of these primary analysis notebooks:

1) word2vec Experiment - An experiment that applies word2vec yields a set similarity sets. These sets are interesting but contain mixed results, such as the combination of numbers with animals (due to a high frequency of co-occurrence). However, these sets are useful for notebook 2.
2) Context Selection Discovery - This NB explores which colexemes should be selected on the basis of the noun's status in the clause. E.g. if the noun is in a direct object function phrase, the verb of the predicate phrase is valuable and should be considered. The similar words generated in the word2vec experiment (1) are used to identify other desirable traits that should be considered in the selection process.
3) Context Selection Development - This NB develops the functions that can extract the needed relations identified in the context selection discovery NB.
4) Semantic Space Construction - The semantic space is constructed using the context selection functions. Words are clustered into similarity sets. The results are exported to a text-fabric data representation stored on lexeme nodes in the BHSA dataset.
5) Evaluation - The semantic space is evaluated by exploring what features are most determiniative for a word's location in the space or its similarity to other words. Are certain verbs or coordinate relations especially important? Can these verbs/co-nouns be considered formal attributes of a noun?
