'''
This module contains a group of 
experiment parameters that can be
fed to the experiments2 version Experiment
classes to construct target and basis elements.
'''

from __main__ import F, E, T, L, S
import re

params = {}

# TODO: ADD REFERENCE CODES!
good_sem_codes = '|'.join(set(f'1\.00100{i}[0-9]*' for i in range(1, 7)) | {'1\.002004\[0-9]*'})

def verb_token(target):
    # standard verb target tokenizer
    vs = F.vs.v(target)
    lex = F.lex.v(target)
    return f'{lex}.{vs}'

def code2tag(sem_domain_code):
    # map sem domain to manual codes (see next funct) #TODO!!: FIX SEM CODES
    good_codes = good_sem_codes['good_codes']
    code = re.findall(good_codes, sem_domain_code)[0][:8]
    if code == '1.001001':
        return 'animate'
    elif code in set(f'1.00100{i}' for i in range(2, 7)):
        return 'inanimate'
    elif code == '1.002004':
        return 'event'
    
def sem_domain_tokens(basis, target):
    # basis tokenizer
    sem_category = code2tag(F.sem_domain_code.v(basis))
    function = F.function.v(L.u(basis, 'phrase')[0])
    return f'{function}.{sem_category}'

def lexer(basis, target):
    # basis tokenizer for simple lexemes
    return F.lex.v(basis)

# standard predicate target template
pred_target = '''
clause
    phrase function={pred_funct}
        target:word pdp=verb

{basis}

lex freq_lex>9
   lexword:word 
   lexword = target
'''




# \\ 1.1.1 Verb Inventory, Subject Only, no sem domains, lexemes

vi_s_nsd = pred_target.format(basis='''

    phrase typ=NP|PrNP function=Subj
        -heads> word

''', pred_funct='Pred|PreO|PtcO'
)

params['vbframe_s_nodomain_lex'] = (
                                    (vi_s_nsd, None, 2, (4,) verb_token, lexer),
                                )




# \\ 1.1.2 Verb Inventory, Subject Only, sem domains

vi_s_sd = pred_target.format(basis=f'''

    phrase typ=NP|PrNP function=Subj
        -heads> word sem_domain_code~{good_sem_codes}

''', pred_funct='Pred|PreO|PtcO'
)

params['vbframe_s_nodomain_lex'] = (
                                    (vi_s_nsd, None, 2, (4,) verb_token, lexer),
                                )





# \\ 1.2.1 Verb Inventory, Object Only, no sem domains, lexemes

vi_o_nsd = pred_target.format(basis='''

    phrase typ=NP|PrNP function=Objc
        -heads> word

''', pred_funct='Pred|PreS'
)

params['vbframe_o_nodomain_lex'] = (
                                        (vi_o_nsd, None, 2, (4,) verb_token, F.lex.v),
                                   )




# \\ 1.2.2 Verb Inventory, Object Only, no sem domains, no lex

vi_o_nsd_nl = pred_target.format(basis='''

    phrase typ=NP|PrNP function=Objc
        -heads> word

''', pred_funct='Pred|PreS'
)


vi_o_nsd_nl_suffix = pred_target.format(basis='', pred_funct='PreO|PtcO')

def simple_object(basis, target):
    return 'object'

params['vbframe_o_nodomain_nolex'] = (
                                        (vi_o_nsd_nl, None, 2, (4,) verb_token, simple_object),
                                        (vi_o_nsd_nl_suffix, None, 2, (2,) verb_token, simple_object)
                                    )





# \\ 1.2.3 Verb Inventory, Object Only, sem domains (basic)

vi_o_sd = pred_target.format(basis=f'''

    phrase typ=NP|PrNP function=Objc
        -heads> word sem_domain_code~{good_sem_codes}

''', pred_funct='Pred|PreS'
)

params['vbframe_o_domain'] = (
                                (vi_o_sd, None, 2, (4,) verb_token, sem_domain_tokens)
                             )





# \\ N. Verb Frames, Sem Domain, Subj/Obj Only \\

# noun phrases, target=2, basis=4
VF_SD_SO_NP = pred_target.format(basis=f'''
        
    phrase typ=NP|PrNP function=Subj|Objc
        -heads> word sem_domain_code~{good_sem_codes}
''', pred_funct='Pred')

# prepositional phrases, target=2, basis=5
VF_SD_SO_PP = pred_target.format(basis=f'''

    p1:phrase typ=PP function=Subj|Objc
        -heads> word
        -prep_obj> word sem_domain_code~{good_sem_codes}
''', pred_funct='Pred')

# verbs with unfulfilled subject/object slots
VF_SD_SO_null = pred_target.format(basis='', pred_funct='Pred')

def vfsd_so_null_filter(results):
    '''
    Removes matches with subject/object functions.
    '''
    results = [r for r in results 
               if not {'Subj', 'Objc', 'PreO', 'PreS', 'PtcO', 'Rela'} & set(F.function.v(ph) for ph in L.d(r[0], 'phrase'))]
    return results

# Accompanying Tokenizer Functions
    
def sem_domain_null(basis, target):
    # basis tokenizer for blank frames
    return 'ø'
    
# All Parameters

params['vbframe_so_domains'] = (
                        (VF_SD_SO_NP, None, 2, (4,), verb_token, sem_domain_tokens),
                        (VF_SD_SO_PP, None, 2, (5,), verb_token, sem_domain_tokens),
                        (VF_SD_SO_null, vfsd_so_null_filter, 2, (2,), verb_token, sem_domain_null)
                     )