'''
This module contains a second generation of 
experiment classes. Rather than classes that
have various feature selection functions, I will be
using Text-Fabric search templates for data selection.
These templates will be linked to tokenizers which will
convert the coordinated data into token strings for counting.

This method has many advantages over the first-gen, class-only
format. With the old format, I gained extensibility
at the expense of readability and clarity. In the new method,
we maintain extensibility while maximizing clarity:

√ selection parameters for bases and targets clearly described: 
    templates are very easy to read/edit compared with code.
√ individual clause elements chosen separately but can be 
    united in a dictionary keyed by the clause; 
    good for both frame spaces and individual feature spaces
√ all parameters coordinated at once in one readable space (template)
    to avoid having to reclarify multiple parameters in multiple
    places
'''

class Experiment:
    
    def __init__(self, tf=None, parameters):
        '''
        Parameters is a tuple of tuples.
        Tuples consist of:
        
        (template, template_kwargs, target_i, 
        (bases_i,), target_tokenizer, basis_tokenizer)

        template - a TF search template (string)
        template_kwargs - dict of keyword arguments for formatting the template (if any)
        target_i - the index of the target word
        bases_i - tuple of basis indexes
        target_tokenizer - a function to construct target tokens, requires target_i
        basis_tokenizer - a function to construct basis tokens, requires basis_i(s)
        '''
        
        # Text-Fabric method short forms
        F, E, T, L, S = tf.F, tf.E, tf.T, tf.L, tf.S
        
        # experiment_data[target_token][clause][list_of_bases_tokens]
        experiment_data = collections.defaultdict(lambda: collections.defaultdict(list))
        
        for templ, templ_kw, target, bases, target_token, basis_token in parameters:
            
            search_template = templ.format(*templ_kw)
            sample = S.search(search_template)
            
            
    
    
    
    
    
    
    
    
    
    
# Semantic Verb Frame Templates
good_codes = set(f'1\.00100{i}.*' for i in range(1, 7)) | {'1\.002004\.*'}
bases=('Subj', 'PrAd', 'Adju', 'Cmpl', 'Loca', 'Time', 'Objc',)
verb_frames = '''

clause
    phrase function=Pred
        word lex=JY>[ vs=hif pdp=verb
        
    p1:phrase function=Objc typ=PP
        w1:word
        w2:word sem_domain~{good_codes}

p1 -head> w1
w1 -prep_obj> w2

'''.format(good_codes='|'.format(good_codes),
          )