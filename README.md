# Semantic Space
## Harvesting Noun Similarity Sets from BHSA

Read the draft text, "Modeling Noun Semantics with Vector Spaces and the ETCBC Hebrew Database" (above) of my 2018 ETEN conference presentation, describing the rationale of this project.

The goal of this project is to cluster similar nouns in the Hebrew Bible using only syntactic data. That is done with a semantic [vector space model](https://en.wikipedia.org/wiki/Vector_space_model), by which the lexical and syntactic context of words can be compared as [vectors](https://en.wikipedia.org/wiki/Vector_space). 

This project applies an adaptation of Padó and Lapata's ([2007](https://www.mitpressjournals.org/doi/pdf/10.1162/coli.2007.33.2.161)) approach by which syntax is used to restrict co-occurrence selections. This syntactically informed method is contrasted with, for instance, [word2vec](https://radimrehurek.com/gensim/models/word2vec.html), which only uses a selection window (i.e. words that occur within `N` slots). 

The data used for the analysis is the [ETCBC](http://www.etcbc.nl)'s [BHSA](https://github.com/ETCBC/bhsa) in [Text-Fabric representation](https://github.com/Dans-labs/text-fabric/wiki). 
