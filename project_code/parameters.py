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
    
def nuller(basis, target):
    # basis tokenizer for blank values
    return 'ø'

# standard predicate target template
pred_target = '''
clause domain=N
    phrase function={pred_funct}
        target:word pdp=verb

{basis}

lex freq_lex>9
   lexword:word 
   lexword = target
'''

all_preds = 'Pred|PreO|PresS|PtcO'

# - - - - - - Parameters - - - - - - -



# \\ 1.1 Verb Inventory, Subject Only, lexemes

vi_s_nsd = pred_target.format(basis='''

    phrase typ=NP|PrNP function=Subj
        -heads> word pdp=subs|nmpr

''', pred_funct='Pred|PreO|PtcO'
)

params['vbi_s_lex'] = (
                          (vi_s_nsd, None, 2, (4,), verb_token, lexer, False),
                      )




# \\ 1.2 Verb Inventory, Subject Only, sem domains

vi_s_sd = pred_target.format(basis=f'''

    phrase typ=NP|PrNP function=Subj
        -heads> word pdp=subs|nmpr sem_domain_code~{good_sem_codes}

''', pred_funct='Pred|PreO|PtcO'
)

params['vbi_s_domain'] = (
                             (vi_s_nsd, None, 2, (4,), verb_token, lexer, False),
                         )




# \\ 2.1 Verb Inventory, Object Only, presence/absence

vi_o_pa = pred_target.format(basis='''

    phrase function=Objc
    
''', pred_funct='Pred|PreS'
)

vi_o_pa_clRela = pred_target.format(basis='''

<mother- clause rela=Objc

''', pred_funct='Pred|PreS')
    
vi_o_pa_suffix = pred_target.format(basis='', pred_funct='PreO|PtcO')
vi_o_pa_null = pred_target.format(basis='', pred_funct='Pred|PreS')

def simple_object(basis, target):
    return 'object'

def notexist_relative(results):
    '''
    this function purposely excludes relative clauses
    since the database does not properly mark whether
    the particle serves as the implied object of the clause
    '''
    results = [r for r in results
                  if 'Rela' not in set(F.function.v(ph) for ph in L.d(r[0], 'phrase'))]
    return results

def notexist_o(results):
    # filter for absent objects (+omits relatives) within and between clauses
    obj = {'PreO', 'PtcO', 'Objc', 'Rela'}
    results = [r for r in results
                  if not obj & set(F.function.v(ph) for ph in L.d(r[0], 'phrase'))
                  and 'Objc' not in set(F.rela.v(cl) for cl in E.mother.t(r[0]))]
    return results

params['vbi_o_pa'] = (
                         (vi_o_pa, notexist_relative, 2, (3,), verb_token, simple_object, True),
                         (vi_o_pa_clRela, None, 2, (3), verb_token, simple_object, True)
                         (vi_o_pa_suffix, notexist_relative, 2, (2,), verb_token, simple_object, True),
                         (vi_o_pa_null, notexist_o, 2, (2,), verb_token, nuller, True)
                     )




# \\ 2.2, Verb Inventory, Object Only, lexemes

vi_o_lex_np = pred_target.format(basis='''

    phrase typ=NP|PrNP function=Objc
        -heads> word pdp=subs|nmpr

''', pred_funct='Pred|PreS'
)

vi_o_lex_pp = pred_target.format(basis='''

    phrase typ=PP function=Objc
        -heads> word pdp=prep
        -prep_obj> word pdp=subs|nmpr

''', pred_funct='Pred|PreS'
)

params['vbi_o_lex'] = (
                          (vi_o_lex_np, None, 2, (4,), verb_token, lexer, False),
                          (vi_o_lex_pp, None, 2, (5,), verb_token, lexer, False)
                      )




# \\ 2.3, Verb Inventory, Object Only, sem domains (basic)

vi_o_sd_np = pred_target.format(basis=f'''

    phrase typ=NP|PrNP function=Objc
        -heads> word pdp=subs|nmpr sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds
)

vi_o_sd_pp = pred_target.format(basis=f'''

    phrase typ=PP function=Objc
        -heads> word pdp=prep
        -prep_obj> word pdp=subs|nmpr sem_domain_code~{good_sem_codes}

''', pred_funct='Pred|PreS|PreO|PtcO'
)

params['vbi_o_domain'] = (
                             (vi_o_sd_np, None, 2, (4,), verb_token, sem_domain_tokens, False),
                             (vi_o_sd_pp, None, 2, (5,), verb_token, sem_domain_tokens, False)
                         )




# \\ 3.1, Verb Inventory, Complements, presence/absence

vi_cmp_pa = pred_target.format(basis='''

    phrase function=Cmpl

''', pred_funct=all_preds)

vi_cmp_pa_clRel = pred_target.format(basis='''

<mother- clause rela=Cmpl

''', pred_funct=all_preds)


vi_cmp_pa_null = pred_funct.format(basis='', pred_funct='Pred|PreO|PresS|PtcO')

def simple_cmpl_funct(basis, target):
    return F.function.v(basis)

def simple_cmpl_rela(basis, target):
    return F.rela.v(basis)

def notexist_cmpl(results):
    # checks for non-existing complements within and between clauses
    results = [r for r in results
                  if 'Cmpl' not in set(F.function.v(ph) for ph in L.d(r[0], 'phrase'))
                  and 'Cmpl' not in set(F.rela.v(cl) for cl in E.mother.t(r[0]))
              ]

params['vbi_cmpladj_pa'] = (
                            (vi_cmp_pa, None, 2, (3,), verb_token, simple_cmpl_funct, True),
                            (vi_cmp_pa_clRel, None, 2, (3,), verb_token, simple_cmpl_rela, True),
                            (vi_cmp_pa_null, notexist_cmpl, 3, (3,), verb_token, nuller, True)
                           )




# \\ 3.2, Verb Inventory, Complements, lexemes
vi_cmp_lex_pp = pred_target.format(basis='''

    phrase function=Cmpl typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp~^(?!prep)

''', pred_funct=all_preds)

vi_cmp_lex_np = pred_target.format(basis='''

    phrase function=Cmpl typ=NP|PrNP|AdvP
        -heads> word

''', pred_funct=all_preds)

# to do: complements with related clauses & verbs
# to do: write tokenizer that makes a prep_lex + prep_obj string