'''
This module contains a set of 
experiment parameters that will be
fed to the experiments2 classes 
to construct target and basis elements.
These searches are the primary task of phase 3,
in which I seek the most informative features 
for verb clustering.

Each parameter consists of the following arguments:
(template, filter/sets, target_i, (bases_i's,), target_tokenizer, bases_tokenizer, collapse_instance)

template - string of a Text-Fabric search template
filter - either a filter function or a tuple containing filter and sets to post-process TF search results
target_i - index of a target word in search result
bases_i's - tuple of basis indices in search result
target_tokenizer - function to convert target node into string 
basis_tokenizer - function to convert bases nodes into strings
collapse_instance - T/F on whether to count multiple instances of a basis token within a clause
'''

import re, collections
from __main__ import F, E, T, L, S # Text-Fabric methods

params = collections.defaultdict(dict) # all parameters will be stored here


# - - - - - - General Functions & Parameters- - - - - - -


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

all_preds = 'Pred|PreO|PreS|PtcO' # all predicate phrase functions

def verb_token(target):
    # standard verb target tokenizer
    vs = F.vs.v(target)
    lex = F.lex.v(target)
    return f'{lex}.{vs}'

good_sem_codes = '1\.00[1-3][0-9]*|2\.[0-9]*' # SDBH codes: objects, events, referents, contexts

# ordered in terms of selection preferences, select animate first, etc.
code_priorities = (('1\.001001[0-9]*',  # ANIMATE
                   '1\.00300100[3,6]', 
                   '1\.00300101[0,3]',
                   '2\.075[0-9]*'),

                  ('1\.00100[2-6][0-9]*',  # INANIMATE
                   '1\.00300100[1-2, 4, 7-9]',
                   '1\.00300101[1-2]',
                   '1\.00[1,3]$',
                   '1\.003001', 
                   '1\.003001005', # names of groups (!)
                   '2\.[0-9]*'), # frames
    
                  ('1\.002[1-9]*', # EVENTS
                   '1\.003002[1-9]*',
                   '1\.002$'))

def code2tag(code):
    '''
    Maps SDBH semantic domains to three basic codes:
    animate, inanimate, and events. These codes are
    of interest to the semantic content of a verb.
    '''
    
    animate = '|'.join(code_priorities[0])
    inanimate = '|'.join(code_priorities[1])
    events = '|'.join(code_priorities[2])
    
    if re.search(animate, code):
        return 'animate'
    elif re.search(inanimate, code):
        return 'inanimate'
    elif re.search(events, code):
        return 'event'
    else:
        raise Exception(code) # avoid accidental selections
        
def code2domain(word):
    '''
    Selects the prefered SDBH semantic domain code
    and maps it to the longer form domain.
    '''
    
    code = F.sem_domain_code.v(word)
    domain = F.sem_domain.v(word)
    animate = '|'.join(code_priorities[0])
    inanimate = '|'.join(code_priorities[1])
    events = '|'.join(code_priorities[2])
    
    if re.search(animate, code):
        match = re.findall(animate, code)[0]
        code_index = code.split('|').index(match)
        return domain.split('|')[code_index]
        
    elif re.search(inanimate, code):
        match = re.findall(inanimate, code)[0]
        code_index = code.split('|').index(match)
        return domain.split('|')[code_index]
  
    elif re.search(events, code):
        match = re.findall(events, code)[0]
        code_index = code.split('|').index(match)   
        return domain.split('|')[code_index]
    else:
        raise Exception(code) # avoid accidental selections
        
    
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

def rela_prep_lexer(basis, target):
    # returns clause relation + prep + verb lex
    rela = F.rela.v(L.u(clause, 'phrase')[0])
    prep = next(w for w in L.u(basis, 'phrase') if F.pdp.v(w) == 'prep')
    prep_lex = F.lex.v(prep)
    return f'{rela}.{prep_lex}_{F.lex.v(basis)}'

def rela_conj_lexer(basis, target):
    # returns clause relation + conjunction string + verb lex
    rela = F.rela.v(L.u(basis, 'clause')[0])
    conj_phrase = next(ph for ph in L.d(L.u(basis, 'clause')[0], 'phrase') if F.typ.v(ph) == 'CP')
    conj_string = ''.join(F.lex.v(w) for w in L.d(conj_phrase, 'word'))
    return f'{rela}.{conj_string}_{F.lex.v(basis)}'
   
def rela_lexer(basis, target):
    # returns rela + lex
    rela = F.rela.v(L.u(clause, 'phrase')[0])
    return f'{rela}.{F.lex.v(basis)}'

'''
The following search templates are specialized for
selecting carefully defined clause relations. These
templates have been crafted to select elements from the 
clauses which can easily be lexicalized as basis strings.
It excludes a small numer of clause relations that cannot 
easily be lexicalized, such as verbless clauses without conjunction
elements (i.e. כאשר)
'''
    
clR_vc_CP = '''

#target @ 3
#basis @ 7

s1:sentence

    c2:clause
        p1:phrase typ=CP
    p2:phrase
    
    either:
        clause kind=VC rela={relas} typ#Ptcp
            p3:phrase function=Pred|PreS|PreO
            p3 = p2
    or:
        clause kind=VC rela={relas} typ=Ptcp
            p3:phrase function=PreC|PtcO
            p3 = p2
    end:
    
        basis:word pdp=verb {reqs}

c1 <mother- c2
s1 [[ c1
c2 [[ p2
p1 < p2
'''

clR_vc_prep = '''

#target @ 3
#basis @ 7

s1:sentence
    c2:clause
        no:
            ^ phrase typ=CP
        end:
    p2:phrase
    
    either:
        clause kind=VC rela={relas} typ#Ptcp
            p3:phrase function=Pred|PreS|PreO
            p3 = p2
    or:
        clause kind=VC rela={relas} typ=Ptcp
            p3:phrase function=PreC|PtcO
            p3 = p2
    end:
    
        word pdp=prep
        basis:word pdp=verb {reqs} 

c1 <mother- c2
s1 [[ c1
c2 [[ p2
'''

clR_vc_verb = '''

#target @ 3
#basis @ 6

s1:sentence
    c2:clause
        no:
            ^ phrase typ=CP
        end:
        no:
            ^ word pdp=prin|inrg
        end:
        
    p2:phrase
    
either:
    clause kind=VC rela={relas} typ#Ptcp
        p3:phrase function=Pred|PreS|PreO
            no:
                ^ word pdp=prep
            end:
        p3 = p2
or:
    clause kind=VC rela={relas} typ=Ptcp
        p3:phrase function=PreC|PtcO
            no:
                ^ word pdp=prep
            end:
        p3 = p2
end:
    
        basis:word pdp=verb {reqs}

s1 [[ c1
c1 <mother- c2
c2 [[ p2
'''

clR_nc_CP = '''
c2:clause kind=NC rela={relas}
    phrase typ=CP
    < phrase function=PreC
        -heads> word {reqs}

c1 <mother- c2
'''

clR_nc_PreC_adv = '''

#only for use with adj/cmpl relations 

c2:clause kind=NC rela={relas}
    no:
        ^ phrase typ=CP
    end:
    phrase function=PreC typ=AdvP
        -heads> word {reqs}

c1 <mother- c2
'''

clR_nc_PreC_prep = '''

#only for use with adj/cmpl functions 

c2:clause kind=NC rela={relas}
    no:
        ^ phrase typ=CP
    end:
    phrase function=PreC typ=PP
        -heads> word pdp=prep
        -prep_obj> word {reqs}

c1 <mother- c2
'''




# - - - - - - Parameters - - - - - - -




# 1.1 Verb Inventory, Subjects, lexemes

vi_s_lex_np = pred_target.format(basis='''

    phrase typ=NP|PrNP function=Subj
        -heads> word pdp#prep|prps|prde|prin|inrg

''', pred_funct=all_preds)


params['inventory']['vi_s_lex'] = (
                                      (vi_s_lex_np, None, 2, (4,), verb_token, lexer, False),
                                  )




# 1.2 Verb Inventory, Subjects, Semantic Domains

vi_s_sd = pred_target.format(basis=f'''

    phrase typ=NP|PrNP function=Subj
        -heads> word pdp#prep|prps|prde|prin|inrg sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds)

params['inventory']['vi_s_domain'] = (
                                         (vi_s_sd, None, 2, (4,), verb_token, domainer, False),
                                     )




# 2.1 Verb Inventory, Objects, Presence/Absence

vi_o_pa = pred_target.format(basis='''

    phrase function=Objc
        
''', pred_funct=all_preds)

vi_o_pa_clRela = pred_target.format(basis='''

c2:clause rela=Objc
c1 <mother- c2

''', pred_funct=all_preds)
    
vi_o_pa_suffix = pred_target.format(basis='', pred_funct='PreO|PtcO')

vi_o_pa_null = pred_target.format(basis='''

c2:clause
    no:
        <mother- clause rela=Objc
    end:
    no:
        ^ phrase function=Objc|PtcO|Rela
    end:

c1 = c2
''', pred_funct='Pred|PreS')

def simple_object(basis, target):
    return 'Objc'

params['inventory']['vi_o_pa'] = (
                                     (vi_o_pa, None, 2, (3,), verb_token, simple_object, True),
                                     (vi_o_pa_clRela, None, 2, (3,), verb_token, simple_object, True),
                                     (vi_o_pa_suffix, None, 2, (2,), verb_token, simple_object, True),
                                     (vi_o_pa_null, None, 2, (2,), verb_token, nuller, True)
                                 )




# 2.2, Verb Inventory, Objects, Lexemes

vi_o_lex_np = pred_target.format(basis='''

    phrase typ=NP|PrNP function=Objc
        -heads> word pdp#prep|prps|prde|prin|inrg

''', pred_funct=all_preds)

vi_o_lex_pp = pred_target.format(basis='''

    phrase typ=PP function=Objc
        -heads> word pdp=prep
        -prep_obj> word pdp#prep|prps|prde|prin|inrg

''', pred_funct=all_preds)

# Clause Relations
vi_objc_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Objc', reqs=''), 
                                       pred_funct=all_preds)
vi_objc_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Objc', reqs=''),
                                        pred_funct=all_preds)
vi_objc_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Objc', reqs=''),
                                        pred_funct=all_preds)
vi_objc_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Objc', reqs=''),
                                      pred_funct=all_preds)

def prep_verber(basis, target):
    # returns prep + verb lex, for e.g. infinitives
    prep = next(w for w in L.u(basis, 'phrase') if F.pdp.v(w) == 'prep')
    prep_lex = F.lex.v(prep)
    return f'{prep_lex}_{F.lex.v(basis)}'

def conj_lexer(basis, target):
    # returns conjunction string + verb lex
    conj_phrase = next(ph for ph in L.d(L.u(basis, 'clause')[0], 'phrase') if F.typ.v(ph) == 'CP')
    conj_string = ''.join(F.lex.v(w) for w in L.d(conj_phrase, 'word'))
    return f'{conj_string}_{F.lex.v(basis)}'

params['inventory']['vi_o_lex'] = (
                                      (vi_o_lex_np, None, 2, (4,), verb_token, lexer, False),
                                      (vi_o_lex_pp, None, 2, (5,), verb_token, lexer, False),
                                      (vi_objc_cr_vc_CP, None, 3, (7,), verb_token, conj_lexer, False),
                                      (vi_objc_cr_vc_prep, None, 3, (7,), verb_token, prep_verber, False),
                                      (vi_objc_cr_vc_verb, None, 3, (6,), verb_token, lexer, False),
                                      (vi_objc_cr_nc_CP, None, 2, (6,), verb_token, conj_lexer, False),
                                  )




# 2.3, Verb Inventory, Objects, Semantic Domains

vi_o_sd_np = pred_target.format(basis=f'''

    phrase typ=NP|PrNP function=Objc
        -heads> word pdp#prep|prps|prde|prin|inrg sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds
)

vi_o_sd_pp = pred_target.format(basis=f'''

    phrase typ=PP function=Objc
        -heads> word pdp=prep
        -prep_obj> word pdp#prep|prps|prde|prin|inrg sem_domain_code~{good_sem_codes}

''', pred_funct='Pred|PreS|PreO|PtcO'
)

# Clause Relations
vi_objcSD_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Objc', reqs='sem_domain_code~{good_sem_codes}'), 
                                        pred_funct=all_preds)
vi_objcSD_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Objc', reqs='sem_domain_code~{good_sem_codes}'),
                                          pred_funct=all_preds)
vi_objcSD_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Objc', reqs='sem_domain_code~{good_sem_codes}'),
                                          pred_funct=all_preds)
vi_objcSD_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Objc', reqs='sem_domain_code~{good_sem_codes}'),
                                        pred_funct=all_preds)

def prep_verbDomainer(basis, target):
    # combines a infinitive verb with its preposition
    prep = next(w for w in L.u(basis, 'phrase') if F.pdp.v(w) == 'prep')
    prep_lex = F.lex.v(prep)
    sem_category = code2tag(F.sem_domain_code.v(basis))
    return f'{prep_lex}_{sem_category}'
    
def conj_domainer(basis, target):
    # returns conjunction string + verb lex
    conj_phrase = next(ph for ph in L.d(L.u(basis, 'clause')[0], 'phrase') if F.typ.v(ph) == 'CP')
    conj_string = ''.join(F.lex.v(w) for w in L.d(conj_phrase, 'word'))
    sem_category = code2tag(F.sem_domain_code.v(basis))
    return f'{conj_string}_{sem_category}'
                                        
params['inventory']['vi_o_domain'] = (
                                         (vi_o_sd_np, None, 2, (4,), verb_token,  domainer, False),
                                         (vi_o_sd_pp, None, 2, (5,), verb_token, domainer, False),
                                         (vi_objcSD_cr_vc_CP, None, 3, (7,), verb_token, conj_domainer, False),
                                         (vi_objcSD_cr_vc_prep, None, 3, (7,), verb_token, prep_verbDomainer, False),
                                         (vi_objcSD_cr_vc_verb, None, 3, (6,), verb_token, domainer, False),
                                         (vi_objcSD_cr_nc_CP, None, 2, (6,), verb_token, conj_domainer, False),
                                     )




# 2.4, Verb Inventory, Objects, Semantic Domains - Longform

def domainer2(basis, target):
    # basis tokenizer for semantic domains
    sem_domain = code2domain(basis)
    return sem_category

def prep_o_domainer2(basis, target):
    # makes prep_domain + prep_obj_domain tokens
    prep_obj = E.prep_obj.f(basis)[0]
    sem_domain = code2domain(basis)
    return f'{F.lex.v(basis)}_{sem_domain}'

def prep_verbDomainer2(basis, target):
    # combines a infinitive verb with its preposition
    prep = next(w for w in L.u(basis, 'phrase') if F.pdp.v(w) == 'prep')
    prep_lex = F.lex.v(prep)
    sem_domain = code2domain(basis)
    return f'{prep_lex}_{sem_domain}'
    
def conj_domainer2(basis, target):
    # returns conjunction string + verb lex
    conj_phrase = next(ph for ph in L.d(L.u(basis, 'clause')[0], 'phrase') if F.typ.v(ph) == 'CP')
    conj_string = ''.join(F.lex.v(w) for w in L.d(conj_phrase, 'word'))
    sem_domain = code2domain(basis)
    return f'{conj_string}_{sem_domain}'
                                        
params['inventory']['vi_o_domain2'] = (
                                         (vi_o_sd_np, None, 2, (4,), verb_token,  domainer2, False),
                                         (vi_o_sd_pp, None, 2, (5,), verb_token, domainer2, False),
                                         (vi_objcSD_cr_vc_CP, None, 3, (7,), verb_token, conj_domainer2, False),
                                         (vi_objcSD_cr_vc_prep, None, 3, (7,), verb_token, prep_verbDomainer2, False),
                                         (vi_objcSD_cr_vc_verb, None, 3, (6,), verb_token, domainer2, False),
                                         (vi_objcSD_cr_nc_CP, None, 2, (6,), verb_token, conj_domainer2, False),
                                     )




# 3.1, Verb Inventory, Complements, Presence/Absence

vi_cmp_pa = pred_target.format(basis='''

    phrase function=Cmpl

''', pred_funct=all_preds)

vi_cmp_pa_clRel = pred_target.format(basis='''

c2:clause rela=Cmpl
c1 <mother- c2
    
''', pred_funct=all_preds)


vi_cmp_pa_null = pred_target.format(basis='''

c2:clause
    no:
        phrase function=Cmpl
    end:
    no:
        <mother- clause rela=Cmpl
    end:

c1 = c2
''', pred_funct=all_preds)


params['inventory']['vi_cmpl_pa'] = (
                                        (vi_cmp_pa, None, 2, (3,), verb_token, functioner, True),
                                        (vi_cmp_pa_clRel, None, 2, (3,), verb_token, relationer, True),
                                        (vi_cmp_pa_null, None, 2, (3,), verb_token, nuller, True)
                                    )




# 3.2, Verb Inventory, Complements, Lexemes
vi_cmpl_lex_pp = pred_target.format(basis='''

    phrase function=Cmpl typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp#prep|prps|prde|prin|inrg

''', pred_funct=all_preds)

vi_cmpl_lex_np = pred_target.format(basis='''

    phrase function=Cmpl typ=NP|PrNP|AdvP
        -heads> word pdp#prep|prps|prde|prin|inrg

''', pred_funct=all_preds)

# Clause Relations
vi_cmpl_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Cmpl', reqs=''), 
                                       pred_funct=all_preds)
vi_cmpl_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Cmpl', reqs=''),
                                        pred_funct=all_preds)
vi_cmpl_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Cmpl', reqs=''),
                                        pred_funct=all_preds)
vi_cmpl_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Cmpl', reqs=''),
                                      pred_funct=all_preds)
vi_cmpl_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Cmpl', reqs=''),
                                            pred_funct=all_preds)
vi_cmpl_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Cmpl', reqs=''),
                                            pred_funct=all_preds)

        
params['inventory']['vi_cmpl_lex'] = (
                                         (vi_cmpl_lex_pp, None, 2, (4,), verb_token, prep_o_lexer, False),
                                         (vi_cmpl_lex_np, None, 2, (4,), verb_token, lexer, False),
                                         (vi_cmpl_cr_vc_CP, None, 3, (7,), verb_token, conj_lexer, False),
                                         (vi_cmpl_cr_vc_prep, None, 3, (7,), verb_token, prep_verber, False),
                                         (vi_cmpl_cr_vc_verb, None, 3, (6,), verb_token, lexer, False),
                                         (vi_cmpl_cr_nc_CP, None, 2, (6,), verb_token, conj_lexer, False),
                                         (vi_cmpl_cr_nc_Prec_adv, None, 2, (5,), verb_token, lexer, False),
                                         (vi_cmpl_cr_nc_Prec_prep, None, 2, (6,), verb_token, prep_verber, False),
                                     )




# 3.3, Verb Inventory, Complements, Semantic Domains
vi_cmpl_sd_pp = pred_target.format(basis=f'''

    phrase function=Cmpl typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp#prep|prps|prde|prin|inrg sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds)

vi_cmpl_sd_np = pred_target.format(basis=f'''

    phrase function=Cmpl typ=NP|PrNP|AdvP
        -heads> word pdp#prep|prps|prde|prin|inrg sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds)

# Clause Relations
vi_cmplSD_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Cmpl', reqs='sem_domain_code~{good_sem_codes}'), 
                                        pred_funct=all_preds)
vi_cmplSD_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Cmpl', reqs='sem_domain_code~{good_sem_codes}'),
                                          pred_funct=all_preds)
vi_cmplSD_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Cmpl', reqs='sem_domain_code~{good_sem_codes}'),
                                          pred_funct=all_preds)
vi_cmplSD_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Cmpl', reqs='sem_domain_code~{good_sem_codes}'),
                                        pred_funct=all_preds)
vi_cmplSD_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Cmpl', reqs='sem_domain_code~{good_sem_codes}'),
                                              pred_funct=all_preds)
vi_cmplSD_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Cmpl', reqs='sem_domain_code~{good_sem_codes}'),
                                               pred_funct=all_preds)
    
params['inventory']['vi_cmpl_domain'] = (
                                             (vi_cmpl_sd_pp, None, 2, (4,), verb_token, prep_o_domainer, False),
                                             (vi_cmpl_sd_np, None, 2, (4,), verb_token, domainer, False),
                                             (vi_cmplSD_cr_vc_CP, None, 3, (7,), verb_token, conj_domainer, False),
                                             (vi_cmplSD_cr_vc_prep, None, 3, (7,), verb_token, prep_verbDomainer, False),
                                             (vi_cmplSD_cr_vc_verb, None, 3, (6,), verb_token, domainer, False),
                                             (vi_cmplSD_cr_nc_CP, None, 2, (6,), verb_token, conj_domainer, False),
                                             (vi_cmplSD_cr_nc_Prec_adv, None, 2, (5,), verb_token, domainer, False),
                                             (vi_cmplSD_cr_nc_Prec_prep, None, 2, (6,), verb_token, prep_verbDomainer, False),
                                         )




# 3.4, Verb Inventory, Complements, Semantic Domains - Longform
params['inventory']['vi_cmpl_domain2'] = (
                                             (vi_cmpl_sd_pp, None, 2, (4,), verb_token, prep_o_domainer2, False),
                                             (vi_cmpl_sd_np, None, 2, (4,), verb_token, domainer2, False),
                                             (vi_cmplSD_cr_vc_CP, None, 3, (7,), verb_token, conj_domainer2, False),
                                             (vi_cmplSD_cr_vc_prep, None, 3, (7,), verb_token, prep_verbDomainer2, False),
                                             (vi_cmplSD_cr_vc_verb, None, 3, (6,), verb_token, domainer2, False),
                                             (vi_cmplSD_cr_nc_CP, None, 2, (6,), verb_token, conj_domainer2, False),
                                             (vi_cmplSD_cr_nc_Prec_adv, None, 2, (5,), verb_token, domainer2, False),
                                             (vi_cmplSD_cr_nc_Prec_prep, None, 2, (6,), verb_token, prep_verbDomainer2, False),
                                         )




# 4.1, Verb Inventory, Adjuncts +(Location, Time, PrAd), Presence/Absence

vi_adj_pa = pred_target.format(basis='''

    phrase function=Adju|Time|Loca|PrAd

''', pred_funct=all_preds)

vi_adj_pa_clRel = pred_target.format(basis='''

c2:clause rela=Adju|PrAd
    c1 <mother- c2
    
''', pred_funct=all_preds)


vi_adj_pa_null = pred_target.format(basis='''

c2:clause
    no:
        phrase function=Adju|Time|Loca|PrAd
    end:
    no:
        <mother- clause rela=Adju|PrAd
    end:
    
c1 = c2

''', pred_funct=all_preds)


params['inventory']['vi_adj+_pa'] = (
                                        (vi_adj_pa, None, 2, (3,), verb_token, functioner, True),
                                        (vi_adj_pa_clRel, None, 2, (3,), verb_token, relationer, True),
                                        (vi_adj_pa_null, None, 2, (3,), verb_token, nuller, True)
                                    )




# 4.2, Verb Inventory, Adjuncts+, Lexemes

vi_adj_lex_pp = pred_target.format(basis='''

    phrase function=Adju|Time|Loca|PrAd typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp#prep|prps|prde|prin|inrg

''', pred_funct=all_preds)

vi_adj_lex_np = pred_target.format(basis='''

    phrase function=Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word pdp#prep|prps|prde|prin|inrg

''', pred_funct=all_preds)

# Clause Relations
vi_adj_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Adju|PrAd', reqs=''), 
                                     pred_funct=all_preds)
vi_adj_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Adju|PrAd', reqs=''),
                                       pred_funct=all_preds)
vi_adj_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Adju|PrAd', reqs=''),
                                       pred_funct=all_preds)
vi_adj_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Adju|PrAd', reqs=''),
                                     pred_funct=all_preds)
vi_adj_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Adju|PrAd', reqs=''),
                                           pred_funct=all_preds)
vi_adj_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Adju|PrAd', reqs=''),
                                            pred_funct=all_preds)
    
params['inventory']['vi_adj+_lex'] = (
                                         (vi_adj_lex_pp, None, 2, (4,), verb_token, prep_o_lexer, False),
                                         (vi_adj_lex_np, None, 2, (4,), verb_token, lexer, False),
                                         (vi_adj_cr_vc_CP, None, 3, (7,), verb_token, conj_lexer, False),
                                         (vi_adj_cr_vc_prep, None, 3, (7,), verb_token, prep_verber, False),
                                         (vi_adj_cr_vc_verb, None, 3, (6,), verb_token, lexer, False),
                                         (vi_adj_cr_nc_CP, None, 2, (6,), verb_token, conj_lexer, False),
                                         (vi_adj_cr_nc_Prec_adv, None, 2, (5,), verb_token, lexer, False),
                                         (vi_adj_cr_nc_Prec_prep, None, 2, (6,), verb_token, prep_verber, False),
                                     )




# 4.3, Verb Inventory, Adjuncts+, Semantic Domains

vi_adj_sd_pp = pred_target.format(basis=f'''

    phrase function=Adju|Time|Loca|PrAd typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp#prep|prps|prde|prin|inrg sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds)

vi_adj_sd_np = pred_target.format(basis=f'''

    phrase function=Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word pdp#prep|prps|prde|prin|inrg sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds)

# Clause Relations
vi_adjSD_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Adju|PrAd', reqs='sem_domain_code~{good_sem_codes}'), 
                                        pred_funct=all_preds)
vi_adjSD_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Adju|PrAd', reqs='sem_domain_code~{good_sem_codes}'),
                                          pred_funct=all_preds)
vi_adjSD_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Adju|PrAd', reqs='sem_domain_code~{good_sem_codes}'),
                                          pred_funct=all_preds)
vi_adjSD_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Adju|PrAd', reqs='sem_domain_code~{good_sem_codes}'),
                                        pred_funct=all_preds)
vi_adjSD_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Adju|PrAd', reqs='sem_domain_code~{good_sem_codes}'),
                                              pred_funct=all_preds)
vi_adjSD_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Adju|PrAd', reqs='sem_domain_code~{good_sem_codes}'),
                                               pred_funct=all_preds)

    
params['inventory']['vi_adj+_domain'] = (
                                             (vi_adj_sd_pp, None, 2, (4,), verb_token, prep_o_domainer, False),
                                             (vi_adj_sd_np, None, 2, (4,), verb_token, domainer, False),
                                             (vi_adjSD_cr_vc_CP, None, 3, (7,), verb_token, conj_domainer, False),
                                             (vi_adjSD_cr_vc_prep, None, 3, (7,), verb_token, prep_verbDomainer, False),
                                             (vi_adjSD_cr_vc_verb, None, 3, (6,), verb_token, domainer, False),
                                             (vi_adjSD_cr_nc_CP, None, 2, (6,), verb_token, conj_domainer, False),
                                             (vi_adjSD_cr_nc_Prec_adv, None, 2, (5,), verb_token, domainer, False),
                                             (vi_adjSD_cr_nc_Prec_prep, None, 2, (6,), verb_token, prep_verbDomainer, False),
                                        )




# 4.4, Verb Inventory, Adjuncts+, Semantic Domains - Longform

params['inventory']['vi_adj+_domain2'] = (
                                             (vi_adj_sd_pp, None, 2, (4,), verb_token, prep_o_domainer2, False),
                                             (vi_adj_sd_np, None, 2, (4,), verb_token, domainer2, False),
                                             (vi_adjSD_cr_vc_CP, None, 3, (7,), verb_token, conj_domainer2, False),
                                             (vi_adjSD_cr_vc_prep, None, 3, (7,), verb_token, prep_verbDomainer2, False),
                                             (vi_adjSD_cr_vc_verb, None, 3, (6,), verb_token, domainer2, False),
                                             (vi_adjSD_cr_nc_CP, None, 2, (6,), verb_token, conj_domainer2, False),
                                             (vi_adjSD_cr_nc_Prec_adv, None, 2, (5,), verb_token, domainer2, False),
                                             (vi_adjSD_cr_nc_Prec_prep, None, 2, (6,), verb_token, prep_verbDomainer2, False),
                                        )


# PICKUP HERE

# 5.1, Verb Frames, All Arguments, Presence/Absence

vf_allarg_pa_np = pred_target.format(basis='''

    p1:phrase
    either:
        p2:phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd typ#PP
        p1 = p2
    or:
        p2:phrase function=Objc typ=PP
        p1 = p2
    end:
    
''', pred_funct=all_preds)

vf_allarg_pa_pp = pred_target.format(basis='''

    phrase function=Cmpl|Adju|Time|Loca|PrAd typ=PP
        -heads> word pdp=prep

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
        -prep_obj> word
''', pred_funct='Pred|PreS')

vf_allarg_lex_pp_obj = pred_target.format(basis='''
    phrase function=Objc typ=PP
        -heads> word pdp=prep
        -prep_obj> word
''', pred_funct='Pred|PreS')

    
# Clause Relations
vf_args_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Objc|Cmpl|Adju', reqs=''), 
                                      pred_funct='Pred|PreS')
vf_args_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Objc|Cmpl|Adju', reqs=''),
                                        pred_funct='Pred|PreS')
vf_args_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Objc|Cmpl|Adju', reqs=''),
                                        pred_funct='Pred|PreS')
vf_args_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Objc|Cmpl|Adju', reqs=''),
                                      pred_funct='Pred|PreS')
vf_args_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Cmpl|Adju', reqs=''),
                                            pred_funct='Pred|PreS')
vf_args_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Cmpl|Adju', reqs=''),
                                             pred_funct='Pred|PreS')

def funct_lexer(basis, target):
    # returns function + lexeme basis tokens
    function = F.function.v(L.u(basis, 'phrase')[0])
    return f'{function}.{F.lex.v(basis)}'

def funct_prep_o_lexer(basis, target):
    # returns function + preplex + preplex object
    function = F.function.v(L.u(basis, 'phrase')[0])
    prep_obj = E.prep_obj.f(basis)[0]
    return f'{function}.{F.lex.v(basis)}_{F.lex.v(prep_obj)}'

'''
Method Notes:
Within the frame, every capturable element
must be present. If there is an uncapturable element, 
we must exclude the entire clause. Examples of "uncapturable
elements" are daughter clauses that are verbless without a 
conjunction. It is not possible to condense these down into
a lexical token, as can be done with כאשר + verb, for instance.
Thus, these kinds of clauses must be excluded.

In order to know which clauses should be excluded, we have
to run the experiment twice so that every clause relation can be
checked and validated. The first time we run it here in this module.
Two functions fulfill that role:
    get_goodDaughters
    filterAllClauses
    filterRela
Good daughters runs the experiments and returns a dictionary
of mappings from clause relations to a set of good daughter clauses.
That set will be used for the filter which is fed to the Experiment class,
filterAllClauses, which applies a filter for good daughters. Another function 
combines this filter with another, filterPreC, which is required for the first
three clause relation patterns.
'''
    
params = ((vf_args_cr_vc_CP, filterPreC),
          (vf_args_cr_vc_prep, filterPreC),
          (vf_args_cr_vc_verb, filterPreC),
          (vf_args_cr_nc_CP, None),
          (vf_args_cr_nc_Prec_adv, None),
          (vf_args_cr_nc_Prec_prep, None))

def get_goodDaughters(params):
    '''
    Runs all of the experiments
    and maps acceptable daughters
    to a dictionary by their relation.
    '''
    goodDaughters = collections.defaultdict(set)
    covered_clauses = set()
    for template, filt in params:
        results = [r for r in S.search(template, sets=sets)]
        results = results if not filt else filt(results)
        for res in results:
            rela = F.rela.v(res[3])
            good_Daughters[rela].add(res[3])
    return goodDaughters

goodDaughters = get_goodDaughters(params)

def filterAllClauses(results):
    '''
    Applies the goodDaughters dict
    as a filter. All applicable daughteres
    for a given clause must be validated.
    '''
    new_results = []
    for res in results:
        mother = res[0]
        daughters = [d for d in E.mother.t(mother) if F.rela.v(r) in {'Objc', 'Adju', 'Cmpl'}]
        daught_is_good = [d in goodDaughters[F.rela.v(d)] for d in daughters]
        if all(daught_is_good):
            new_results.append(res)
    
def filterRela(results):
    '''
    Applies both the daughter filter
    and the PreC filter.
    '''
    new_results = filterPreC(results)
    new_results2 = filterAllClauses(new_results)
    return new_results2
    
params['frame']['vf_argAll_lex'] = (
                                        (vf_allarg_lex_np, filterAllClauses, 2, (4,), verb_token, funct_lexer, False),
                                        (vf_allarg_lex_pp_obj, filterAllclauses, 2, (4,), verb_token, funct_lexer, False),
                                        (vf_allarg_lex_pp, filterAllClauses, 2, (4,), verb_token, funct_prep_o_lexer, False),
                                        (vf_args_cr_vc_CP, filterPreC, 2, (6,), verb_token, rela_conj_lexer, False),
                                        (vf_args_cr_vc_prep, (filterRela, RelaSets), 2, (6,), verb_token, rela_prep_lexer, False),
                                        (vf_args_cr_vc_verb, (filterRela, RelaSets), 2, (5,), verb_token, rela_lexer, False),
                                        (vf_args_cr_nc_CP, filterRela, 2, (6,), verb_token, rela_conj_lexer, False),
                                        (vf_args_cr_nc_Prec_adv, (filterRela, RelaSets), 2, (5,), verb_token, rela_lexer, False),
                                        (vf_args_cr_nc_Prec_prep, (filterRela, RelaSets), 2, (6,), verb_token, rela_prep_lexer, False),
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


# Clause Relations
vf_argsSD_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Objc|Cmpl|Adju', reqs='sem_domain_code~{good_sem_codes}'), 
                                        pred_funct='Pred|PreS')
vf_argsSD_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Objc|Cmpl|Adju', reqs='sem_domain_code~{good_sem_codes}'),
                                          pred_funct='Pred|PreS')
vf_argsSD_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Objc|Cmpl|Adju', reqs='sem_domain_code~{good_sem_codes}'),
                                          pred_funct='Pred|PreS')
vf_argsSD_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Objc|Cmpl|Adju', reqs='sem_domain_code~{good_sem_codes}'),
                                        pred_funct='Pred|PreS')
vf_argsSD_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Cmpl|Adju', reqs='sem_domain_code~{good_sem_codes}'),
                                              pred_funct='Pred|PreS')
vf_argsSD_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Cmpl|Adju', reqs='sem_domain_code~{good_sem_codes}'),
                                              pred_funct='Pred|PreS')

# run experiments for good daughters check (see explanation above)
paramsSD = ((vf_argsSD_cr_vc_CP, filterPreC),
           (vf_argsSD_cr_vc_prep, filterPreC),
           (vf_argsSD_cr_vc_verb, filterPreC),
           (vf_argsSD_cr_nc_CP, None),
           (vf_argsSD_cr_nc_Prec_adv, None),
           (vf_argsSD_cr_nc_Prec_prep, None))

goodDaughtersSD = get_goodDaughters(paramsSD)

def filterAllClausesSD(results):
    '''
    Applies the goodDaughters dict
    as a filter. All applicable daughteres
    for a given clause must be validated.
    '''
    new_results = []
    for res in results:
        mother = res[0]
        daughters = [d for d in E.mother.t(mother) if F.rela.v(r) in {'Objc', 'Adju', 'Cmpl'}]
        daught_is_good = [d in goodDaughtersSD[F.rela.v(d)] for d in daughters]
        if all(daught_is_good):
            new_results.append(res)
            
def filterRelaSD(results):
    '''
    Applies both the daughter filter
    and the PreC filter.
    '''
    new_results = filterPreC(results)
    new_results2 = filterAllClausesSD(new_results)
    return new_results2

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

def rela_prep_domainer(basis, target):
    # returns clause relation + prep + verb domain
    rela = F.rela.v(L.u(clause, 'phrase')[0])
    prep = next(w for w in L.u(basis, 'phrase') if F.pdp.v(w) == 'prep')
    prep_lex = F.lex.v(prep)
    sem_category = code2tag(F.sem_domain_code.v(basis))
    return f'{rela}.{prep_lex}_{sem_category}'

def rela_conj_domainer(basis, target):
    # returns clause relation + conjunction string + verb domain
    rela = F.rela.v(L.u(basis, 'clause')[0])
    conj_phrase = next(ph for ph in L.d(L.u(basis, 'clause')[0], 'phrase') if F.typ.v(ph) == 'CP')
    conj_string = ''.join(F.lex.v(w) for w in L.d(conj_phrase, 'word'))
    sem_category = code2tag(F.sem_domain_code.v(basis))
    return f'{rela}.{conj_string}_{sem_category}'
   
def rela_domainer(basis, target):
    # returns rela + domain
    rela = F.rela.v(L.u(clause, 'phrase')[0])
    sem_category = code2tag(F.sem_domain_code.v(basis))
    return f'{rela}.{sem_category}'
    
params['frame']['vf_argAll_domain'] = (
                                          (vf_allarg_sd_np, filterAllClausesSD, 2, (4,), verb_token, funct_domainer, False),
                                          (vf_allarg_sd_pp, filterAllClausesSD, 2, (4,), verb_token, funct_prep_o_domainer, False),
                                          (vf_args_cr_vc_CP, filterPreC, 2, (6,), verb_token, rela_conj_domainer, False),
                                          (vf_args_cr_vc_prep, (filterRelaSD, RelaSets), 2, (6,), verb_token, rela_prep_domainer, False),
                                          (vf_args_cr_vc_verb, (filterRelaSD, RelaSets), 2, (5,), verb_token, rela_domainer, False),
                                          (vf_args_cr_nc_CP, filterRelaSD, 2, (6,), verb_token, rela_conj_domainer, False),
                                          (vf_args_cr_nc_Prec_adv, (filterRelaSD, RelaSets), 2, (5,), verb_token, rela_domainer, False),
                                          (vf_args_cr_nc_Prec_prep, (filterRelaSD, RelaSets), 2, (6,), verb_token, rela_prep_domainer, False),
                                      )




# 5.4, Verb Frames, All Arguments, Semantic Domains - Longform

def funct_domainer2(basis, target):
    # basis tokenizer for semantic domains + functions
    function = F.function.v(L.u(basis, 'phrase')[0])
    sem_domain = code2domain(basis)
    return f'{function}.{sem_domain}'
    
def funct_prep_o_domainer2(basis, target):
    # makes prep_domain + prep_obj_domain tokens + functions
    prep_obj = E.prep_obj.f(basis)[0]
    sem_domain = code2domain(basis)
    function = F.function.v(L.u(basis, 'phrase')[0])
    return f'{function}.{F.lex.v(basis)}_{sem_domain}'

def rela_prep_domainer2(basis, target):
    # returns clause relation + prep + verb domain
    rela = F.rela.v(L.u(clause, 'phrase')[0])
    prep = next(w for w in L.u(basis, 'phrase') if F.pdp.v(w) == 'prep')
    prep_lex = F.lex.v(prep)
    sem_domain = code2domain(basis)
    return f'{rela}.{prep_lex}_{sem_domain}'

def rela_conj_domainer2(basis, target):
    # returns clause relation + conjunction string + verb domain
    rela = F.rela.v(L.u(basis, 'clause')[0])
    conj_phrase = next(ph for ph in L.d(L.u(basis, 'clause')[0], 'phrase') if F.typ.v(ph) == 'CP')
    conj_string = ''.join(F.lex.v(w) for w in L.d(conj_phrase, 'word'))
    sem_domain = code2domain(basis)
    return f'{rela}.{conj_string}_{sem_domain}'
   
def rela_domainer2(basis, target):
    # returns rela + domain
    rela = F.rela.v(L.u(clause, 'phrase')[0])
    sem_domain = code2domain(basis)
    return f'{rela}.{sem_domain}'

params['frame']['vf_argAll_domain2'] = (
                                          (vf_allarg_sd_np, filterAllClauses, 2, (4,), verb_token, funct_domainer2, False),
                                          (vf_allarg_sd_pp, filterAllClauses, 2, (4,), verb_token, funct_prep_o_domainer2, False),
                                          (vf_args_cr_vc_CP, filterPreC, 2, (6,), verb_token, rela_conj_domainer2, False),
                                          (vf_args_cr_vc_prep, (filterRela, RelaSets), 2, (6,), verb_token, rela_prep_domainer2, False),
                                          (vf_args_cr_vc_verb, (filterRela, RelaSets), 2, (5,), verb_token, rela_domainer2, False),
                                          (vf_args_cr_nc_CP, filterRela, 2, (6,), verb_token, rela_conj_domainer2, False),
                                          (vf_args_cr_nc_Prec_adv, (filterRela, RelaSets), 2, (5,), verb_token, rela_domainer2, False),
                                          (vf_args_cr_nc_Prec_prep, (filterRela, RelaSets), 2, (6,), verb_token, rela_prep_domainer2, False),
                                      )


''''

# 6.1, Verb Frame, Objects, Lexemes 

# 6.2, Verb Frame, Objects, Semantic Domains

# 6.3, Verb Frame, Objects, Semantic Domains - Longform



# 7.1, Verb Frame, Complements, Lexemes


def filterCmplClauses(results):
    '''
    Applies the goodDaughters dict
    as a filter. All applicable daughteres
    for a given clause must be validated.
    '''
    new_results = []
    for res in results:
        mother = res[0]
        daughters = [d for d in E.mother.t(mother) if F.rela.v(r) in {'Cmpl'}]
        daught_is_good = [d in goodDaughters[F.rela.v(d)] for d in daughters]
        if all(daught_is_good):
            new_results.append(res)


params['frame']['vf_cmpl_lex'] = (
                                     (vi_cmpl_lex_pp, ADDFUNCTION, 2, (4,), verb_token, funct_prep_o_lexer, False),
                                     (vi_cmpl_lex_np, ADDFUNCTION, 2, (4,), verb_token, funct_lexer, False),
                                     (vi_cmpl_lex_clRela, ADDFUNCTION, 2, (5,), verb_token, funct_lexer, False)
                                     (vi_cmpl_cr_vc_CP, filterPreC, 2, (6,), verb_token, rela_conj_lexer, False),
                                     (vi_cmpl_cr_vc_prep, (filterRela, RelaSets), 2, (6,), verb_token, rela_prep_lexer, False),
                                     (vi_cmpl_cr_vc_verb, (filterRela, RelaSets), 2, (5,), verb_token, rela_lexer, False),
                                     (vi_cmpl_cr_nc_CP, filterRela, 2, (6,), verb_token, rela_conj_lexer, False),
                                     (vi_cmpl_cr_nc_Prec_adv, (filterRela, RelaSets), 2, (5,), verb_token, rela_lexer, False),
                                     (vi_cmpl_cr_nc_Prec_prep, (filterRela, RelaSets), 2, (6,), verb_token, rela_prep_lexer, False),
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




# 9.2, Verb Frame, Adjuncts, Semantic Domains
params['frame']['vf_adj+_domain'] = (
                                        (vi_adj_sd_pp, None, 2, (4,), verb_token, funct_prep_o_domainer, False),
                                        (vi_adj_sd_np, None, 2, (4,), verb_token, funct_domainer, False),
                                        (vi_adj_sd_clRela, None, 2, (5,), verb_token, funct_domainer, False)
                                    )


''''
# 10.1, Verb Discourse, Parallelism, Lexemes

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




# 11.1, Verb Discourse, Context, Window-2 Content words

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




# 11.2, Verb Discourse, Context, Clause Content Words

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




# 11.3, Verb Discourse, Context, Mother-Daughter Chain Content Words

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




# 11.4, Verb Discourse, Domain, Simple

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




# 11.5, Verb Discourse, Domain, Embedding

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




# 12.1, Verb Grammar, Tense

vg_tense = pred_target.format(basis='', pred_funct=all_preds)

def tenser(basis, target):
    # makes tense basis tokens
    return F.vt.v(basis)

params['inventory']['vg_tense'] = (
                                      (vg_tense, None, 2, (2,), verb_token, tenser, True),
                                  )