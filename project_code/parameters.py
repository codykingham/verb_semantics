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

import re, collections
from __main__ import F, E, T, L, S # Text-Fabric methods

params = collections.defaultdict(dict) # all parameters stored here


# - - - - - - General Functions - - - - - - -


good_sem_codes = '1\.00[1-3][0-9]*|2\.[0-9]*' # SDBH codes: objects, events, referents, contexts

def verb_token(target):
    # standard verb target tokenizer
    vs = F.vs.v(target)
    lex = F.lex.v(target)
    return f'{lex}.{vs}'

def code2tag(code):
    '''
    Maps SDBH semantic domains to three basic codes:
    animate, inanimate, and events. These codes are
    of interest to the semantic content of a verb.
    '''

    # get best code; more specific ones first
#    pref_code = '1\.001[0-9]*|2\.075[0-9]*' # prefer object codes + a custom code for human matches
#    long_sem = '1\.00[1-3][0-9][0-9]*' # prefer longer codes
#    short_sem = '1\.00[1-3][0-9]*'
#    frame_code = '2\.[0-9]*' # last resort, there are only a few of these in the dataset
#    code = re.findall(pref_code, sem_domain_code) or\
#           re.findall(long_sem, sem_domain_code) or\
#           re.findall(short_sem, sem_domain_code) or\
#            re.findall(frame_code, sem_domain_code)
#    
#    code = code[0]

    
    animate = '|'.join(('1\.001001[0-9]*', 
                        '1\.00300100[3,6]', 
                        '1\.00300101[0,3]',
                        '2\.075[0-9]*'
                       ))
    inanimate = '|'.join(('1\.00100[2-6][0-9]*',
                          '1\.00300100[1-2, 4, 7-9]',
                          '1\.00300101[1-2]',
                          '1\.00[1,3]$',
                          '1\.003001', 
                          '1\.002[1-9]*', # <- events counted as inanimate(as of 25 May 2018)
                          '1\.003002[1-9]*', # event references
                          '1\.002$', # events, shortform
                          '1\.003001005', # names of groups (!)
                          '2\.[0-9]*' # frames
                         ))
    
    if re.search(animate, code):
        return 'animate'
    elif re.search(inanimate, code):
        return 'inanimate'
    else:
        raise Exception(code)
    
def domainer(basis, target):
    # basis tokenizer for semantic domains
    sem_category = code2tag(F.sem_domain_code.v(basis))
    return sem_category

def prep_o_domainer(basis, target):
    # makes prep_domain + prep_obj_domain tokens
    prep_obj = E.prep_obj.f(basis)[0]
    prep_o_domain = code2tag(F.sem_domain_code.v(prep_obj))
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
c1:clause
    phrase function={pred_funct}
        target:word pdp=verb language=Hebrew

{basis}

lex freq_lex>9
   lexword:word 
   lexword = target
'''

all_preds = 'Pred|PreO|PresS|PtcO'



# - - - - - - Parameters - - - - - - -



# 1.1 Verb Inventory, Subjects, lexemes

vi_s_lex = pred_target.format(basis='''

    phrase typ=NP|PrNP function=Subj
        -heads> word pdp=subs|nmpr

''', pred_funct='Pred|PreO|PtcO'
)

params['inventory']['vi_s_lex'] = (
                                      (vi_s_lex, None, 2, (4,), verb_token, lexer, False),
                                  )




# 1.2 Verb Inventory, Subjects, Semantic Domains

vi_s_sd = pred_target.format(basis=f'''

    phrase typ=NP|PrNP function=Subj
        -heads> word pdp=subs|nmpr sem_domain_code~{good_sem_codes}

''', pred_funct='Pred|PreO|PtcO'
)

params['inventory']['vi_s_domain'] = (
                                         (vi_s_sd, None, 2, (4,), verb_token, domainer, False),
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
    return 'Objc'

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

params['inventory']['vi_o_pa'] = (
                                     (vi_o_pa, notexist_relative, 2, (3,), verb_token, simple_object, True),
                                     (vi_o_pa_clRela, None, 2, (3,), verb_token, simple_object, True),
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

params['inventory']['vi_o_lex'] = (
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

params['inventory']['vi_o_domain'] = (
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


vi_cmp_pa_null = pred_target.format(basis='', pred_funct='Pred|PreO|PresS|PtcO')

def notexist_cmpl(results):
    # checks for non-existing complements within and between clauses
    results = [r for r in results
                  if 'Cmpl' not in set(F.function.v(ph) for ph in L.d(r[0], 'phrase'))
                  and 'Cmpl' not in set(F.rela.v(cl) for cl in E.mother.t(r[0]))
              ]
    return results

params['inventory']['vi_cmpl_pa'] = (
                                        (vi_cmp_pa, None, 2, (3,), verb_token, functioner, True),
                                        (vi_cmp_pa_clRel, None, 2, (3,), verb_token, relationer, True),
                                        (vi_cmp_pa_null, notexist_cmpl, 2, (3,), verb_token, nuller, True)
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

''', pred_funct=all_preds)
    
params['inventory']['vi_cmpl_lex'] = (
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

''', pred_funct=all_preds)
    
params['inventory']['vi_cmpl_domain'] = (
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


vi_adj_pa_null = pred_target.format(basis='', pred_funct='Pred|PreO|PresS|PtcO')

def notexist_adj(results):
    # checks for non-existing complements within and between clauses
    results = [r for r in results
                  if not {'Adju', 'Time', 'Loca', 'PrAd'} & set(F.function.v(ph) for ph in L.d(r[0], 'phrase'))
                  and not {'Adju', 'PrAd'} & set(F.rela.v(cl) for cl in E.mother.t(r[0]))
              ]
    return results

params['inventory']['vi_adj+_pa'] = (
                                        (vi_adj_pa, None, 2, (3,), verb_token, functioner, True),
                                        (vi_adj_pa_clRel, None, 2, (3,), verb_token, relationer, True),
                                        (vi_adj_pa_null, notexist_adj, 2, (3,), verb_token, nuller, True)
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

''', pred_funct=all_preds)
    
params['inventory']['vi_adj+_lex'] = (
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

''', pred_funct=all_preds)

    
params['inventory']['vi_adj+_domain'] = (
                                             (vi_adj_sd_pp, None, 2, (4,), verb_token, prep_o_domainer, False),
                                             (vi_adj_sd_np, None, 2, (4,), verb_token, domainer, False),
                                             (vi_adj_sd_clRela, None, 2, (5,), verb_token, domainer, False)
                                        )




# 5.2, Verb Frames, All Arguments, Presence/Absence

vf_allarg_pa_np = pred_target.format(basis='''

    phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd typ~^(?!PP)

''', pred_funct=all_preds)

vf_allarg_pa_pp = pred_target.format(basis='''

    phrase function=Cmpl|Adju|Time|Loca|PrAd typ=PP
        -heads> word pdp=prep

''', pred_funct=all_preds)

vf_allarg_pa_pp_obj = pred_target.format(basis='''

    phrase function=Objc typ=PP

''', pred_funct=all_preds)

vf_allarg_pa_suffix = pred_target.format(basis='', pred_funct='PreO|PtcO')

vf_allarg_pa_null = pred_target.format(basis='', pred_funct='Pred|PreS')

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

params['frame']['vf_argAll_pa'] = (
                                      (vf_allarg_pa_np, None, 2, (3,), verb_token, functioner, False),
                                      (vf_allarg_pa_pp, None, 2, (4,), verb_token, prep_o_functioner, False),
                                      (vf_allarg_pa_pp_obj, None, 3, (3,), verb_token, functioner, False),
                                      (vf_allarg_pa_suffix, None, 2, (2,), verb_token, simple_object, False),
                                      (vf_allarg_pa_null, notexist_allargs, 2, (2,), verb_token, nuller, False)
                                  )



# 5.2, Verb Frames, All Arguments, Lexemes

vf_allarg_lex_np = pred_target.format(basis='''

    phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word

''', pred_funct='Pred|PreS')

vf_allarg_lex_pp = pred_target.format(basis='''

    phrase function=Cmpl|Adju|Time|Loca|PrAd typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp~^(?!prep)

''', pred_funct='Pred|PreS')

vf_allarg_lex_pp_obj = pred_target.format(basis='''

    phrase function=Objc typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp~^(?!prep)

''', pred_funct='Pred|PreS')

def funct_lexer(basis, target):
    # returns function + lexeme basis tokens
    function = F.function.v(L.u(basis, 'phrase')[0])
    return f'{function}.{F.lex.v(basis)}'

def funct_prep_o_lexer(basis, target):
    # returns function + preplex + preplex object
    function = F.function.v(L.u(basis, 'phrase')[0])
    prep_obj = E.prep_obj.f(basis)[0]
    return f'{function}.{F.lex.v(basis)}_{F.lex.v(prep_obj)}'

def notexist_badPhrases(results):
    '''
    Filter out clauses with undesirable argument phrases.
    These are:
    • Phrases with suffixes (can't measure their lexical referent)
    • Phrases with bad prepositional objects (have no valid lexical referent)
    • Phrases with an invalid phrase type (only a handfull of examples).
    If would be ideal for the Search templates to already have operators
    where I could explicitly exclude these examples. Those operators, "quantifiers",
    are currently still in the works.
    
    The most efficient way to do this right now is to use templates
    to find undesirable clause nodes. I do that below and filter the
    results accordingly.
    '''
    
    suffixed_phrases = S.search('''
    
clause
    phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd typ=PP
    -heads> word pdp=prep prs~^(?!absent)
    
    ''', shallow=True)
    
    bad_prep_obj = S.search('''
    
clause
     phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp=prep
    
    ''', shallow=True)
    
    good_types = {'NP', 'PrNP', 'AdvP', 'PP'}
    bad_types = '|'.join(typ for typ,count in F.typ.freqList(nodeTypes={'phrase'}) if typ not in good_types)
    
    bad_phrase_types = S.search(f'''
    
clause
    phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd typ={bad_types}
    
    ''', shallow=True)
    
    new_results = [r for r in results if r[0] not in suffixed_phrases|bad_prep_obj|bad_phrase_types]
    
    return new_results
    
    
params['frame']['vf_argAll_lex'] = (
                                      (vf_allarg_lex_np, notexist_badPhrases, 2, (4,), verb_token, funct_lexer, False),
                                      (vf_allarg_lex_pp_obj, notexist_badPhrases, 2, (4,), verb_token, funct_lexer, False),
                                      (vf_allarg_lex_pp, notexist_badPhrases, 2, (4,), verb_token, funct_prep_o_lexer, False),
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

def funct_domainer(basis, target):
    # basis tokenizer for semantic domains + functions
    function = F.function.v(L.u(basis, 'phrase')[0])
    sem_category = code2tag(F.sem_domain_code.v(basis))
    return f'{function}.{sem_category}'
    
def funct_prep_o_domainer(basis, target):
    # makes prep_domain + prep_obj_domain tokens + functions
    prep_obj = E.prep_obj.f(basis)[0]
    prep_o_domain = code2tag(F.sem_domain_code.v(prep_obj))
    function = F.function.v(L.u(basis, 'phrase')[0])
    return f'{function}.{F.lex.v(basis)}_{prep_o_domain}'

    
todo: Add a not-exist function for bad frame phrases.
Even better: figure out a way to do all of this more efficiently
than I have done above for the allArg_lex arguments. There it has become too verbose I think, but perhaps not.
    
params['frame']['vf_argAll_domain'] = (
                                          (vf_allarg_sd_np, None, 2, (4,), verb_token, funct_domainer, False),
                                          (vf_allarg_sd_pp, None, 2, (4,), verb_token, funct_prep_o_domainer, False),
                                      )





# 7.1, Verb Frame, Complements, Lexemes
params['frame']['vf_cmpl_lex'] = (
                                     (vi_cmpl_lex_pp, None, 2, (4,), verb_token, funct_prep_o_lexer, False),
                                     (vi_cmpl_lex_np, None, 2, (4,), verb_token, funct_lexer, False),
                                     (vi_cmpl_lex_clRela, None, 2, (5,), verb_token, funct_lexer, False)
                                 )




# 7.2, Verb Frame, Complements, Semantic Domains
params['frame']['vf_cmpl_domain'] = (
                                        (vi_cmpl_sd_pp, None, 2, (4,), verb_token, funct_prep_o_domainer, False),
                                        (vi_cmpl_sd_np, None, 2, (4,), verb_token, funct_domainer, False),
                                        (vi_cmpl_sd_clRela, None, 2, (5,), verb_token, funct_domainer, False)
                                    )





# 8.1, Verb Frame, Adjuncts, Lexemes
params['frame']['vf_adj+_lex'] = (
                                     (vi_adj_lex_pp, None, 2, (4,), verb_token, funct_prep_o_lexer, False),
                                     (vi_adj_lex_np, None, 2, (4,), verb_token, funct_lexer, False),
                                     (vi_adj_lex_clRela, None, 2, (5,), verb_token, funct_lexer, False)
                                  )




# 8.2, Verb Frame, Adjuncts, Semantic Domains
params['frame']['vf_adj+_domain'] = (
                                        (vi_adj_sd_pp, None, 2, (4,), verb_token, funct_prep_o_domainer, False),
                                        (vi_adj_sd_np, None, 2, (4,), verb_token, funct_domainer, False),
                                        (vi_adj_sd_clRela, None, 2, (5,), verb_token, funct_domainer, False)
                                    )




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
                    target:word pdp=verb language=Hebrew

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
    
params['inventory']['vd_par_lex'] = (
                                         (vd_par_lex_AB, parallelism_filter, 6, (11,), verb_token, lexer, False),
                                         (vd_par_lex_BC, parallelism_filter, 6, (11,), verb_token, lexer, False)
                                     )




# 10.1, Verb Discourse, Context, Window-2 Content words

content_words = {'subs', 'nmpr', 'verb', 'advb', 'adjv'}

vd_con_window = pred_target.format(basis='', pred_funct=all_preds)

def select_window(results):
    
    '''
    Changes the basis result to 2 
    word nodes on the left or right
    of the target verb.
    '''
    
    new_results = []
    
    for r in results:
        sentence = L.u(r[0], 'sentence')[0]
        target = r[2]
        sent_words = L.d(sentence, 'word')
        window = tuple(w for w in range(target-2, target+3) 
                           if w != target
                           and w in sent_words
                           and F.pdp.v(w) in content_words)
        result = list(r) + [window]
        new_results.append(tuple(result))
        
    return tuple(new_results)
        
def multiple_lexer(bases, target):
    # returns multiple lexeme bases elements in a tuple
    return tuple(F.lex.v(w) for w in bases)
    
params['inventory']['vd_con_window'] = (
                                           (vd_con_window, select_window, 2, (-1,), verb_token, multiple_lexer, False),
                                       )




# 10.2, Verb Discourse, Context, Clause Content Words

vd_con_clause = pred_target.format(basis='', pred_funct=all_preds)

def select_cl_words(results):
    '''
    adds all content clause words to end of results
    '''
    new_results = []
    
    # get matching clause words
    for r in results:
        target = r[2]
        clause_words = tuple(w for w in L.d(r[0], 'word')
                           if F.pdp.v(w) in content_words
                           and w != target)
        result = list(r) + [clause_words]
        new_results.append(tuple(result))
    
    return tuple(new_results)

params['inventory']['vd_con_clause'] = (
                                           (vd_con_clause, select_cl_words, 2, (-1,), verb_token, multiple_lexer, False),
                                       )




# 10.3, Verb Discourse, Context, Mother-Daughter Chain Content Words

vd_con_chain = pred_target.format(basis='', pred_funct=all_preds)

def climb_chain(clause_atom, chain_list):
    '''
    Recursive function to iterate through
    clause atom mother-daughter chains.
    '''
    for daughter in E.mother.t(clause_atom):
        chain_list.append(daughter)
        climb_chain(daughter, chain_list)

def select_chain_words(results):
    
    '''
    Selects words from a chain of 
    related clause atoms, related 
    via the mother-daughter relation.
    '''
    new_results = []
    
    for r in results:
        target = r[2]
        clause_atom = L.d(r[0], 'clause_atom')[0]
        clAt_chain = []
        climb_chain(clause_atom, clAt_chain) # climb down the chain
        chain_words = tuple(w for atom in clAt_chain
                                for w in L.d(atom, 'word')
                                if w != target and F.pdp.v(w) in content_words)
        result = list(r) + [chain_words]
        new_results.append(result)

    return tuple(new_results)

params['inventory']['vd_con_chain'] = (
                                          (vd_con_chain, select_chain_words, 2, (-1,), verb_token, multiple_lexer, True),
                                      )




# 10.4, Verb Discourse, Domain, Simple

vd_domain_simple = pred_target.format(basis='', pred_funct=all_preds)

def notexist_unknown_dom(results):
    # filters out unknown domains
    results = [r for r in results if F.domain.v(r[0]) != '?']
    return results
    
def discourse_domainer(basis, target):
    # returns simple domain tag: i.e. N/D/Q
    return F.domain.v(basis)

params['inventory']['vd_domain_simple'] = (
                                              (vd_domain_simple, notexist_unknown_dom, 2, (0,), verb_token, discourse_domainer, True),
                                          )




# 10.5, Verb Discourse, Domain, Embedding

def notexist_unknown_txt(results):
    # filters out unknown domains
    results = [r for r in results if '?' not in F.txt.v(r[0])]
    return results
    
def discourse_txter(basis, target):
    # returns simple domain tag: i.e. N/D/Q
    return F.txt.v(basis)

params['inventory']['vd_domain_embed'] = (
                                              (vd_domain_simple, notexist_unknown_txt, 2, (0,), verb_token, discourse_txter, True),
                                          )




# 11.1, Verb Grammar, Tense

vg_tense = pred_target.format(basis='', pred_funct=all_preds)

def tenser(basis, target):
    # makes tense basis tokens
    return F.vt.v(basis)

params['inventory']['vg_tense'] = (
                                      (vg_tense, None, 2, (2,), verb_token, tenser, True),
                                  )