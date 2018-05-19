'''
This module contains a group of 
experiment parameters that can be
fed to the experiments2 version Experiment
classes to construct target and basis elements.
'''

from __main__ import F, E, T, L, S
import re

# DATA SELECTION

# Verb Frames, Sem Domain, Subj/Obj Only, noun/prep phrases

good_sem_codes = {'good_codes': '|'.join(set(f'1\.00100{i}[0-9]*' for i in range(1, 7)) | {'1\.002004\[0-9]*'})}

# noun phrases, target=2, basis=4
VF_SD_SO_NP = '''

clause
    phrase function=Pred
        target:word pdp=verb
        
    phrase typ=NP|PrNP function=Subj|Objc
        -heads> word sem_domain_code~{good_codes}
    
lex freq_lex>9
   lexword:word 
   lexword = target

'''

# prepositional phrases, target=2, basis=5
VF_SD_SO_PP = '''

clause
    phrase function=Pred
        target:word pdp=verb

    p1:phrase typ=PP function=Subj|Objc
        -heads> word
        -prep_obj> word sem_domain_code~{good_codes}
    
lex freq_lex>9
    lexword:word
    lexword = target
'''

# verbs with unfulfilled subject/object slots
VF_SD_SO_null = '''

clause
    phrase function=Pred
        target:word pdp=verb

lex freq_lex>9
    lexword:word
    lexword = target
'''

def vfsd_so_null_filter(results):
    '''
    Removes matches with subject/object functions.
    '''
    results = [r for r in results 
               if not {'Subj', 'Objc', 'PreO', 'PreS', 'PtcO', 'Rela'} & set(F.function.v(ph) for ph in L.d(r[0], 'phrase'))]
    return results


# DATA ARRANGEMENT

def verb_token(target):
    # target tokenizer
    vs = F.vs.v(target)
    lex = F.lex.v(target)
    return f'{lex}.{vs}'

def code2tag(sem_domain_code):
    # map sem domain to manual codes
    good_codes = good_sem_codes['good_codes']
    code = re.findall(good_codes, sem_domain_code)[0][:8]
    if code == '1.001001':
        return 'animate'
    elif code in set(f'1.00100{i}' for i in range(2, 7)):
        return 'inanimate'
    elif code == '1.002004':
        return 'event'
    
def vfsd_so_base_token(basis, target):
    # basis tokenizer
    sem_category = code2tag(F.sem_domain_code.v(basis))
    function = F.function.v(L.u(basis, 'phrase')[0])
    return f'{function}.{sem_category}'
    
def vfsd_so_base_null_token(basis, target):
    # basis tokenizer for blank frames
    return 'ø'
    
# PACKAGED PARAMETERS

vf_sd_so = (
    (VF_SD_SO_NP, good_sem_codes, None, 2, (4,), verb_token, vfsd_so_base_token),
    (VF_SD_SO_PP, good_sem_codes, None, 2, (5,), verb_token, vfsd_so_base_token),
    (VF_SD_SO_null, {}, vfsd_so_null_filter, 2, (2,), verb_token, vfsd_so_base_null_token)
)