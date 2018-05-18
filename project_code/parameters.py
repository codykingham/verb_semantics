'''
This module contains a group of 
experiment parameters that can be
fed to the experiments2 version Experiment
classes to construct target and basis elements.
'''
     
verb_frames = (

    (
'''
clause
    phrase function=Pred
        word \\Target word
        
    p1:phrase function=Objc typ=PP
        w1:word
        w2:word sem_domain~{good_codes}

p1 -head> w1
w1 -prep_obj> w2
''',
    {'good_codes': '|'.format(set(f'1\.00100{i}.*' for i in range(1, 7)) | {'1\.002004\.*'})
    },
    
    )









)

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
