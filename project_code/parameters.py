'''
This module contains a set of 
experiment parameters that will be
fed to the experiments2 classes 
to construct target and basis elements.
These searches are the primary task of phase 3,
in which I seek the most informative features 
for verb clustering.

Each parameter consists of the following arguments:
(template, target_i, (bases_i's,), target_tokenizer, bases_tokenizer, collapse_instance)

template - string of a Text-Fabric search template
target_i - index of a target word in search result
bases_i's - tuple of basis indices in search result
target_tokenizer - function to convert target node into string 
basis_tokenizer - function to convert bases nodes into strings
collapse_instance - T/F on whether to count multiple instances of a basis token within a clause
'''

import re
from __main__ import F, E, T, L, S # Text-Fabric methods

params = {} # all parameters stored here


# - - - - - - General Functions - - - - - - -


good_sem_codes = '1\.00[1-3][0-9]*' # SDBH codes: objects, events, referents

def verb_token(target):
    # standard verb target tokenizer
    vs = F.vs.v(target)
    lex = F.lex.v(target)
    return f'{lex}.{vs}'

def code2tag(sem_domain_code):
    '''
    Maps SDBH semantic domains to three basic codes:
    animate, inanimate, and events. These codes are
    of interest to the semantic content of a verb.
    '''

    code = next(re.findall(good_sem_codes, sem_domain_code))
    
    animate = '|'.join(('1\.001001[0-9]*', 
                        '1\.00300100[3,5-6]', 
                        '1\.00300101[0,3]'))   
    inanimate = '|'.join(('1\.00100[2-6][0-9]*',
                          '1\.00300100[1-2, 4, 7-9]',
                          '1\.00300101[1-2]'))
    events = '|'.join(('1\.002[1-9]*',
                       '1\.003002[1-9]*'))
    
    if re.search(animate, code):
        return 'animate'
    elif re.search(inanimate, code):
        return 'inanimate'
    elif re.search(events, code):
        return 'event'
    
def domainer(basis, target):
    # basis tokenizer for semantic domains
    sem_category = code2tag(F.sem_domain_code.v(basis))
    return sem_category

def prep_o_domainer(basis, target):
    # makes prep_domain + prep_obj_domain tokens
    prep_obj = E.prep_obj.f(basis)[0]
    prep_o_domain = code2tag[F.sem_domain_code.v(prep_obj)]
    return f'{F.lex.v(basis)}_{prep_o_domain}'

def lexer(basis, target):
    # basis tokenizer for simple lexemes
    return F.lex.v(basis)

def prep_o_lexer(basis, target):
    # makes prep_lex + prep_obj_lex token
    prep_obj = E.prep_obj.f(basis)[0]
    return f'{F.lex.v(basis)}_{F.lex.v(prep_obj)}'
    
def nuller(basis, target):
    # basis tokenizer for blank values
    return 'ø'

def functioner(basis, target):
    # function basis tokens
    return F.function.v(basis)

def relationer(basis, target):
    # clause relation basis tokens
    return F.rela.v(basis)

# standard predicate target template
pred_target = '''
cl1:clause
    phrase function={pred_funct}
        target:word pdp=verb

{basis}

lex freq_lex>9
   lexword:word 
   lexword = target
'''

all_preds = 'Pred|PreO|PresS|PtcO'



# - - - - - - Parameters - - - - - - -



# 1.1 Verb Inventory, Subjects, lexemes

vi_s_nsd = pred_target.format(basis='''

    phrase typ=NP|PrNP function=Subj
        -heads> word pdp=subs|nmpr

''', pred_funct='Pred|PreO|PtcO'
)

params['vbi_s_lex'] = (
                          (vi_s_nsd, None, 2, (4,), verb_token, lexer, False),
                      )




# 1.2 Verb Inventory, Subjects, Semantic Domains

vi_s_sd = pred_target.format(basis=f'''

    phrase typ=NP|PrNP function=Subj
        -heads> word pdp=subs|nmpr sem_domain_code~{good_sem_codes}

''', pred_funct='Pred|PreO|PtcO'
)

params['vbi_s_domain'] = (
                             (vi_s_nsd, None, 2, (4,), verb_token, lexer, False),
                         )




# 2.1 Verb Inventory, Objects, Presence/Absence

vi_o_pa = pred_target.format(basis='''

    phrase function=Objc
    
''', pred_funct='Pred|PreS'
)

vi_o_pa_clRela = pred_target.format(basis='''

c2:clause rela=Objc
    c1 <mother- c2

''', pred_funct='Pred|PreS')
    
vi_o_pa_suffix = pred_target.format(basis='', pred_funct='PreO|PtcO')
vi_o_pa_null = pred_target.format(basis='', pred_funct='Pred|PreS')

def simple_object(basis, target):
    return 'object'

def notexist_relative(results):
    '''
    this function purposely excludes relative clauses
    since the database does not properly mark whether
    the relative serves as the implied object of the verb
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




# 2.2, Verb Inventory, Objects, Lexemes

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




# 2.3, Verb Inventory, Objects, Semantic Domains

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
                             (vi_o_sd_np, None, 2, (4,), verb_token,  domainer, False),
                             (vi_o_sd_pp, None, 2, (5,), verb_token, domainer, False)
                         )




# 3.1, Verb Inventory, Complements, Presence/Absence

vi_cmp_pa = pred_target.format(basis='''

    phrase function=Cmpl

''', pred_funct=all_preds)

vi_cmp_pa_clRel = pred_target.format(basis='''

c2:clause rela=Cmpl
    c1 <mother- c2
    
''', pred_funct=all_preds)


vi_cmp_pa_null = pred_funct.format(basis='', pred_funct='Pred|PreO|PresS|PtcO')

def notexist_cmpl(results):
    # checks for non-existing complements within and between clauses
    results = [r for r in results
                  if 'Cmpl' not in set(F.function.v(ph) for ph in L.d(r[0], 'phrase'))
                  and 'Cmpl' not in set(F.rela.v(cl) for cl in E.mother.t(r[0]))
              ]

params['vbi_cmpl_pa'] = (
                            (vi_cmp_pa, None, 2, (3,), verb_token, functioner, True),
                            (vi_cmp_pa_clRel, None, 2, (3,), verb_token, relationer, True),
                            (vi_cmp_pa_null, notexist_cmpl, 3, (3,), verb_token, nuller, True)
                        )




# 3.2, Verb Inventory, Complements, Lexemes
vi_cmpl_lex_pp = pred_target.format(basis='''

    phrase function=Cmpl typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp~^(?!prep)

''', pred_funct=all_preds)

vi_cmpl_lex_np = pred_target.format(basis='''

    phrase function=Cmpl typ=NP|PrNP|AdvP
        -heads> word

''', pred_funct=all_preds)

vi_cmpl_lex_clRela = pred_target.format(basis='''

c2:clause rela=Cmpl
    phrase typ=VP
    -heads> word pdp=verb
    
c2 -mother> c1

''')
    
params['vbi_cmpl_lex'] = (
                             (vi_cmpl_lex_pp, None, 2, (4,), verb_token, prep_o_lexer, False),
                             (vi_cmpl_lex_np, None, 2, (4,), verb_token, lexer, False),
                             (vi_cmpl_lex_clRela, None, 2, (5,), verb_token, lexer, False)
                         )




# 3.3, Verb Inventory, Complements, Semantic Domains
vi_cmpl_sd_pp = pred_target.format(basis=f'''

    phrase function=Cmpl typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp~^(?!prep) sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds)

vi_cmpl_sd_np = pred_target.format(basis=f'''

    phrase function=Cmpl typ=NP|PrNP|AdvP
        -heads> word sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds)

vi_cmpl_sd_clRela = pred_target.format(basis=f'''

c2:clause rela=Cmpl
    phrase typ=VP
    -heads> word pdp=verb sem_domain_code~{good_sem_codes}
    
c2 -mother> c1

''')
    
params['vbi_cmpl_domain'] = (
                                 (vi_cmpl_sd_pp, None, 2, (4,), verb_token, prep_o_domainer, False),
                                 (vi_cmpl_sd_np, None, 2, (4,), verb_token, domainer, False),
                                 (vi_cmpl_sd_clRela, None, 2, (5,), verb_token, domainer, False)
                             )




# 4.1, Verb Inventory, Adjuncts +(Location, Time, PrAd), Presence/Absence

vi_adj_pa = pred_target.format(basis='''

    phrase function=Adju|Time|Loca|PrAd

''', pred_funct=all_preds)

vi_adj_pa_clRel = pred_target.format(basis='''

c2:clause rela=Adju|PrAd
    c1 <mother- c2
    
''', pred_funct=all_preds)


vi_adj_pa_null = pred_funct.format(basis='', pred_funct='Pred|PreO|PresS|PtcO')

def notexist_adj(results):
    # checks for non-existing complements within and between clauses
    results = [r for r in results
                  if not {'Adju', 'Time', 'Loca', 'PrAd'} & set(F.function.v(ph) for ph in L.d(r[0], 'phrase'))
                  and not {'Adju', 'PrAd'} & set(F.rela.v(cl) for cl in E.mother.t(r[0]))
              ]

params['vbi_adj+_pa'] = (
                            (vi_adj_pa, None, 2, (3,), verb_token, functioner, True),
                            (vi_adj_pa_clRel, None, 2, (3,), verb_token, relationer, True),
                            (vi_adj_pa_null, notexist_adj, 3, (3,), verb_token, nuller, True)
                        )




# 4.2, Verb Inventory, Adjuncts+, Lexemes

vi_adj_lex_pp = pred_target.format(basis='''

    phrase function=Adju|Time|Loca|PrAd typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp~^(?!prep)

''', pred_funct=all_preds)

vi_adj_lex_np = pred_target.format(basis='''

    phrase function=Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word

''', pred_funct=all_preds)

vi_adj_lex_clRela = pred_target.format(basis='''

c2:clause rela=Adju|PrAd
    phrase typ=VP
    -heads> word pdp=verb
    
c2 -mother> c1

''')
    
params['vbi_adj+_lex'] = (
                             (vi_adj_lex_pp, None, 2, (4,), verb_token, prep_o_lexer, False),
                             (vi_adj_lex_np, None, 2, (4,), verb_token, lexer, False),
                             (vi_adj_lex_clRela, None, 2, (5,), verb_token, lexer, False)
                         )




# 4.3, Verb Inventory, Adjuncts+, Semantic Domains

vi_adj_sd_pp = pred_target.format(basis=f'''

    phrase function=Adju|Time|Loca|PrAd typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp~^(?!prep) sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds)

vi_adj_sd_np = pred_target.format(basis=f'''

    phrase function=Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds)

vi_adj_sd_clRela = pred_target.format(basis=f'''

c2:clause rela=Adju|PrAd
    phrase typ=VP
        -heads> word pdp=verb sem_domain_code~{good_sem_codes}
    
c2 -mother> c1

''')

    
params['vbi_adj+_domain'] = (
                                 (vi_adj_sd_pp, None, 2, (4,), verb_token, prep_o_domainer, False),
                                 (vi_adj_sd_np, None, 2, (4,), verb_token, domainer, False),
                                 (vi_adj_sd_clRela, None, 2, (5,), verb_token, domainer, False)
                            )




# 5.2, Verb Frames, All Arguments, Presence/Absence

vf_allarg_pa_np = pred_target.format(basis='''

    phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP

''', pred_funct=all_preds)

vf_allarg_pa_pp = pred_target.format(basis='''

    phrase function=Cmpl|Adju|Time|Loca|PrAd typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp~^(?!prep)

''', pred_funct=all_preds)

vf_allarg_pa_suffix = pred_target.format(basis='', pred_funct='PreO|PtcO')

vf_allarg_pa_null = pred_target.format(basis='', pred_funct=all_preds)

def prep_o_functioner(basis, target):
    # builds prep_lex + function basis tokens
    prep_lex = F.lex.v(basis)
    basis_function = F.function.v(L.u(basis, 'phrase')[0])
    return f'{prep_lex}_{basis_function}'

def notexist_allargs(results):
    all_args = {'Objc', 'Cmpl', 'Adju', 'Time', 'Loca', 'PrAd', 'Rela'}
    results = [r for r in results
                  if not all_args & set(F.function.v(ph) for ph in L.d(r[0], 'phrase'))
                  and not all_args & set(F.rela.v(cl) for cl in E.mother.t(r[0]))]
    return results

params['vf_argAll_pa'] = (
                              (vf_allarg_pa_np, None, 2, (3,), verb_token, functioner, False),
                              (vf_allarg_pa_pp, None, 2, (4,), verb_token, prep_o_functioner, False),
                              (vf_allarg_pa_suffix, None, 2, (2,), verb_token, simple_object, False),
                              (vf_allarg_pa_null, notexist_allargs, 2, (2,), verb_token, nuller, False)
                          )



# 5.2, Verb Frames, All Arguments, Lexemes

vf_allarg_lex_np = pred_target.format(basis='''

    phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word

''', pred_funct='Pred|PreS')

vf_allarg_lex_pp = pred_target.format(basis='''

    phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp~^(?!prep)

''', pred_funct='Pred|PreS')



params['vf_argAll_lex'] = (
                              (vf_allarg_lex_np, None, 2, (4,), verb_token, lexer, False),
                              (vf_allarg_lex_pp, None, 2, (4,), verb_token, prep_o_lexer, False),
                          )




# 5.3, Verb Frames, All Arguments, Semantic Domains

vf_allarg_sd_np = pred_target.format(basis=f'''

    phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word sem_domain_code~{good_sem_codes}

''', pred_funct='Pred|PreS')

vf_allarg_sd_pp = pred_target.format(basis=f'''

    phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp~^(?!prep) sem_domain_code~{good_sem_codes}

''', pred_funct='Pred|PreS')

params['vf_argAll_domain'] = (
                                (vf_allarg_sd_np, None, 2, (4,), verb_token, domainer, False),
                                (vf_allarg_sd_pp, None, 2, (4,), verb_token, prep_o_domainer, False),
                             )




# TODO: 7.1, Verb Frame, Complements, Presence/Absence
# TODO: 7.2, Verb Frame, Complements, Lexemes
# TODO: 7.3, Verb Frame, Complements, Semantic Domains




# TODO: 8.1, Verb Frame, Adjuncts, Presence/Absence
# TODO: 8.2, Verb Frame, Adjuncts, Lexemes
# TODO: 8.3, Verb Frame, Adjuncts, Semantic Domains




# 9.1, Verb Discourse, Parallelism, Lexemes

poetry = '|'.join(F.book.v(book) for book in F.otype.s('book') 
                      if 426595 < book < 426618) # Isaiah-Lamentations

vd_par_lex = '''

book book={poetry}
    verse
        half_verse label={half1}
            == clause domain=D|Q
                == clause_atom
                phrase function=Pred|PreS|PreO|PtcO
                    target:word pdp=verb

        half_verse label={half2}
            == clause domain=D|Q
                == clause_atom
                phrase function=Pred|PreS|PreO|PtcO
                    basis:word pdp=verb
            
lex freq_lex>9
   lexword:word 
   lexword = target
   
'''

vd_par_lex_AB = vd_par_lex.format(poetry=poetry, half1='A', half2='B')
vd_par_lex_BC = vd_par_lex.format(poetry=poetry, half1='B', half2='C')


def close_length(clause1, clause2):
    '''
    Checks the length of 2 clauses for
    proximity in word-length. 
    '''
    min_proximity = 3
    len_cl1 = len(L.d(clause1, 'word'))
    len_cl2 = len(L.d(clause2, 'word'))
    return abs(len_cl1 - len_cl2) <= min_proximity
    
def independent(clauseAt1, clauseAt2):
    '''
    Checks clause atom rela codes to ensure
    that both clauses are independent.
    '''
    ind_codes = list(range(100, 168)) + list(range(200, 202))\
                + list(range(400, 488))
    
    return all([F.code.v(clauseAt1) in ind_codes,
                F.code.v(clauseAt2) in ind_codes])

def parallelism_filter(results):
    # filters for parallelisms
    results = [r for r in results 
                   if close_length(r[3], r[8])
                   and independent(r[4], r[9])]
    return results
    
params['vd_par_lex'] = (
                           (vi_par_lex_AB, parallelism_filter, 6, (11,), verb_token, lexer, False),
                           (vi_par_lex_BC, parallelism_filter, 6, (11,), verb_token, lexer, False)
                       )




# TODO: 10.1, Verb Discourse, Context, Window-2 Content words
# TODO: 10.2, Verb Discourse, Context, Clause Content Words
# TODO: 10.3, Verb Discourse, Context, Mother-Daughter Chain Content Words
# TODO: 10.4, Verb Discourse, Context, Chapter Content Words