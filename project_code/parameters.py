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

import re, collections, dill
from __main__ import F, E, T, L, S # Text-Fabric methods
from __main__ import cached_data

params = collections.defaultdict(dict) # all parameters will be stored here


# - - - - - - General Functions & Parameters- - - - - - -


# standard predicate target template

pred_target = '''

c1:clause
    p1:phrase

    /with/
    clause typ#Ptcp
        p:phrase function={pred_funct}
            -heads> word pdp=verb language=Hebrew
        p = p1
    /or/
    clause typ=Ptcp
        p:phrase function={ptcp_funct}
            -heads> word pdp=verb language=Hebrew
        p = p1
    /-/

        target:word pdp=verb
    
{basis}

lex freq_lex>9
   lexword:word 
   lexword = target
'''

all_preds = 'Pred|PreO|PreS' # all predicate phrase functions
all_ptcp = 'PreC|PtcO'

def verb_token(target):
    # standard verb target tokenizer
    vs = F.vs.v(target)
    lex = F.lex.v(target)
    return f'{lex}.{vs}'

good_sem_codes = '1\.00[1-3][0-9]*|2\.[0-9]*' # SDBH codes: objects, events, referents, semantic frames
animacy_codes = '1\.001[0-9]*|1\.003001[0-9]*|2\.[0-9]*' # eligible codes for animacy mapping

def code2tag(code):
    '''
    Maps SDBH semantic domains to three basic codes:
    animate, inanimate, and events. These codes are
    of interest to the semantic content of a verb.
    
    !! NOW DEFUNCT !!
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

def code2animacy(code):
    
    '''
    Maps SDBH semantic domains to one of two tags:
        animate or inanimate
    Codes fed to this algorithm must first be filtered
    through animacy_codes (cf. above)
    '''
    
    # animate object codes, all other sets of valid codes are inanimate:
    animate = '1\.001001[0-9]*|1\.00300100[3,5,6]|1\.003001010' 
    if re.search(animate, code):
        return 'animate'
    else:
        return 'inanimate'
        
def code2domain(word):
    '''
    Selects the prefered SDBH semantic domain code
    and maps it to the longer form domain.
    '''
    
    # ordered in terms of selection preferences, select animate first, etc.
    code_priorities = (('(1\.001001[0-9]*)',  # ANIMATE
                       '(1\.00300100[3,6])', 
                       '(1\.00300101[0,3])',
                       '(2\.075[0-9]*)',
                        '(1\.003001005$)|(1\.003001005)\|', # names of groups (!)
                       ),

                      ('(1\.00100[2-6][0-9]*)',  # INANIMATE
                       '(1\.00300100[1-2, 4, 7-9])',
                       '(1\.00300101[1-2])',
                       '(1\.00[1,3]$)',
                       '(1\.00[1,3])\|',
                       '(1\.003001$)',
                       '(1\.003001)\|',
                       '(2\.[0-9]*)'), # frames

                      ('(1\.002[0-9]*)', # EVENTS
                       '(1\.003002[0-9]*)',
                       '(1\.002$)|(1\.002)\|',
                       '(1.004003$)|(1.004003)\|',
                       '(1.004005$)|(1.004005)\|'))
    
    code = F.sem_domain_code.v(word)
    domain = F.sem_domain.v(word)
    animate = '|'.join(code_priorities[0])
    inanimate = '|'.join(code_priorities[1])
    events = '|'.join(code_priorities[2])
    if re.search(animate, code):
        match = next(match for group in re.findall(animate, code) for match in group if match)
        code_index = code.split('|').index(match)
        return domain.split('|')[code_index]

    elif re.search(inanimate, code):
        match = next(match for group in re.findall(inanimate, code) for match in group if match)
        code_index = code.split('|').index(match)
        return domain.split('|')[code_index]

    elif re.search(events, code):
        match = next(match for group in re.findall(events, code) for match in group if match)
        code_index = code.split('|').index(match)   
        return domain.split('|')[code_index]
    else:
        raise Exception(word) # avoid accidental selections


    
def animater(basis, target):
    # basis tokenizer for semantic domains
    sem_category = code2animacy(F.sem_domain_code.v(basis))
    return sem_category

def prep_o_animater(basis, target):
    # makes prep_domain + prep_obj_domain tokens
    prep_obj = E.prep_obj.f(basis)[0]
    prep_o_domain = code2animacy(F.sem_domain_code.v(prep_obj))
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
    rela = F.rela.v(L.u(basis, 'clause')[0])
    prep = next(w for w in L.d(L.u(basis, 'phrase')[0], 'word') if F.pdp.v(w) == 'prep')
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
    rela = F.rela.v(L.u(basis, 'clause')[0])
    return f'{rela}.{F.lex.v(basis)}'


'''
Frame Methodology Notes:
Within the frame, every capturable element
must be present. If there is an uncapturable element, 
we must exclude the entire clause. Examples of "uncapturable
elements" are daughter clauses that are verbless without a 
conjunction. It is not possible to condense these down into
a lexical token, as can be done with כאשר + verb, for instance.
Thus, not only these clauses, but also their mothers, must be excluded.

In order to know which clauses should be excluded, we have
to run the whole experiment twice so that every clause relation can be
checked and validated. The first time we run it here in this module.

The second time the queries are run in the Experiment class to produce results.
The results are then crossreferenced against the first run to make sure that all
elligible functions are present in the complete result.

The class validateFrame (below) completes this task. The data is prepared
within the module and is then called to filter the final results.
'''

class validateFrame:
    '''
    This class prepares frame validation data
    and then filters results based on the prepared
    data.
    '''
    
    def __init__(self, mother_templates=tuple(), 
                       daughter_templates=tuple(), 
                       mother_ri = 0,
                       daughter_ri = 3,
                       exp_name = ''):
    
        print(f'Preparing frame validation data for {exp_name}...')

        self.good_mothers = set()
        self.good_daughters = collections.defaultdict(set)
        self.daughter_ri = daughter_ri
        self.mother_ri = mother_ri
        relas = {'Objc', 'Cmpl', 'Adju', 'PrAd'}
        for rela in relas:
            self.good_daughters[rela] = set()
        
        print(f'\tpreparing good mother set...')
        for mom in mother_templates:
            results = set(S.search(mom))
            self.good_mothers |= set(r[mother_ri] for r in results)

        print(f'\tpreparing good daughter set...')
        for daught in daughter_templates:
            results = set(S.search(daught))
            for r in results:
                rela = F.rela.v(r[daughter_ri])
                self.good_daughters[rela].add(r[daughter_ri])

        print(f'\t√ Frame validation data prep complete.')
    
    def mothers(self, results):
        '''
        Checks both a mother and her daughters
        for validity.
        '''
        check_relas = set(self.good_daughters.keys())
        validated_results = []
        for r in results:
            mother = r[self.mother_ri]
            check_mother_daughters = all([d in self.good_daughters[F.rela.v(d)] for d in E.mother.t(mother)
                                              if F.rela.v(d) in check_relas])
            if mother in self.good_mothers and check_mother_daughters:
                validated_results.append(r)
        return validated_results
                
    def daughters(self, results):
        '''
        Checks daughters for validity.
        '''
        check_relas = set(self.good_daughters.keys())
        validated_results = []
        for r in results:
            if all([d in self.good_daughters[F.rela.v(d)] for d in E.mother.t(r[0]) # NB: Assume mother is i=0
                        if F.rela.v(d) in check_relas]):
                validated_results.append(r)
        return validated_results

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

#basis @ 6

c2:clause
    p1:phrase typ=CP
    p2:phrase
    /with/
    clause kind=VC rela={relas} typ#Ptcp
        p3:phrase function=Pred|PreS|PreO
        p3 = p2
    /or/
    clause kind=VC rela={relas} typ=Ptcp
        p3:phrase function=PreC|PtcO
        p3 = p2
    /-/

        basis:word pdp=verb {reqs}

c1 <mother- c2
c2 [[ p2
p1 < p2
'''

clR_vc_prep = '''

#basis @ 6

c2:clause
/without/
    phrase typ=CP
/-/
    p2:phrase
    /with/
    clause kind=VC rela={relas} typ#Ptcp
        p:phrase function=Pred|PreS|PreO
        p = p2
    /or/
    clause kind=VC rela={relas} typ=Ptcp
        p:phrase function=PreC|PtcO
        p = p2
    /-/
    
        word pdp=prep
        < word pdp=verb {reqs} 

c1 <mother- c2
'''

clR_vc_verb = '''

#basis @ 5

c2:clause
/without/
    phrase typ=CP
/-/
/without/
    word pdp=prin|inrg
/-/

    p2:phrase
    
    /with/
    clause kind=VC rela={relas} typ#Ptcp
        p:phrase function=Pred|PreS|PreO
        /without/
            word pdp=prep
        /-/
        p = p2
    /or/
    clause kind=VC rela={relas} typ=Ptcp
        p:phrase function=PreC|PtcO
        /without/
            word pdp=prep
        /-/
        p = p2
    /-/
    
        basis:word pdp=verb {reqs}

c1 <mother- c2
'''

clR_nc_CP = '''
c2:clause kind=NC rela={relas}
    phrase typ=CP
    < phrase function=PreC
        -heads> word pdp#prep|prps|prde|prin|inrg {reqs}

c1 <mother- c2
'''

clR_nc_PreC_adv = '''
#only for use with adj/cmpl relations 

c2:clause kind=NC rela={relas}
/without/
    phrase typ=CP
/-/
    phrase function=PreC typ=AdvP
        -heads> word pdp#prep|prps|prde|prin|inrg {reqs}

c1 <mother- c2
'''

clR_nc_PreC_prep = '''
#only for use with adj/cmpl functions 

c2:clause kind=NC rela={relas}
/without/
    phrase typ=CP
/-/
    phrase function=PreC typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp#prep|prps|prde|prin|inrg {reqs}

c1 <mother- c2
'''




# - - - - - - Parameters - - - - - - -
if cached_data:
    with open('/Users/cody/Documents/VF_cache.dill', 'rb') as infile:
        cache = dill.load(infile)
else:
    cache = {}



# 1.1 Verb Inventory, Subjects, lexemes

vi_s_lex_np = pred_target.format(basis='''

    phrase typ=NP|PrNP function=Subj
        -heads> word pdp#prep|prps|prde|prin|inrg

''', pred_funct=all_preds, ptcp_funct=all_ptcp)


params['inventory']['vi_subj_lex'] = (
                                      (vi_s_lex_np, None, 2, (4,), verb_token, lexer, False),
                                  )




# 1.2 Verb Inventory, Subjects, Semantic Domains

vi_s_sd = pred_target.format(basis=f'''

    phrase typ=NP|PrNP function=Subj
        -heads> word pdp#prep|prps|prde|prin|inrg sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

params['inventory']['vi_subj_domain'] = (
                                         (vi_s_sd, None, 2, (4,), verb_token, animater, False),
                                     )




# 2.1 Verb Inventory, Objects, Presence/Absence

vi_o_pa = pred_target.format(basis='''

    phrase function=Objc
        
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vi_o_pa_clRela = pred_target.format(basis='''

c2:clause rela=Objc
c1 <mother- c2

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vi_o_pa_null = pred_target.format(basis='''

c2:clause
/without/
<mother- clause rela=Objc
/-/
/without/
    phrase function=Objc|Rela
/-/
/without/
    ca:clause_atom
speech:clause_atom code=999
ca <mother- speech
/-/

c1 = c2

''', pred_funct='Pred|PreS', ptcp_funct='PreC')

vi_o_pa_speech = pred_target.format(basis='''

    ca1:clause_atom
    
ca2:clause_atom code=999

ca1 <mother- ca2
ca1 <: ca2
ca1 [[ p1
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vi_o_pa_suffix = pred_target.format(basis='', pred_funct='PreO', ptcp_funct='PtcO')

def simple_object(basis, target):
    return 'Objc'

params['inventory']['vi_objc_pa'] = (
                                     (vi_o_pa, None, 2, (3,), verb_token, simple_object, True),
                                     (vi_o_pa_clRela, None, 2, (3,), verb_token, simple_object, True),
                                     (vi_o_pa_suffix, None, 2, (2,), verb_token, simple_object, True),
                                     (vi_o_pa_speech, None, 2, (2,), verb_token, simple_object, True),
                                     (vi_o_pa_null, None, 2, (2,), verb_token, nuller, True)
                                 )




# 2.2, Verb Inventory, Objects, Lexemes

vi_o_lex_np = pred_target.format(basis='''

    phrase typ=NP|PrNP function=Objc
        -heads> word pdp#prep|prps|prde|prin|inrg

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vi_o_lex_pp = pred_target.format(basis='''

    phrase typ=PP function=Objc
        -heads> word pdp=prep
        -prep_obj> word pdp#prep|prps|prde|prin|inrg

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

# Clause Relations
vi_objc_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Objc', reqs=''), 
                                       pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_objc_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Objc', reqs=''),
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_objc_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Objc', reqs=''),
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_objc_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Objc', reqs=''),
                                      pred_funct=all_preds, ptcp_funct=all_ptcp)

def prep_verber(basis, target):
    # returns prep + verb lex, for e.g. infinitives
    prep = next(w for w in L.d(L.u(basis, 'phrase')[0], 'word') if F.pdp.v(w) == 'prep')
    prep_lex = F.lex.v(prep)
    return f'{prep_lex}_{F.lex.v(basis)}'

def conj_lexer(basis, target):
    # returns conjunction string + verb lex
    conj_phrase = next(ph for ph in L.d(L.u(basis, 'clause')[0], 'phrase') if F.typ.v(ph) == 'CP')
    conj_string = ''.join(F.lex.v(w) for w in L.d(conj_phrase, 'word'))
    return f'{conj_string}_{F.lex.v(basis)}'

params['inventory']['vi_objc_lex'] = (
                                      (vi_o_lex_np, None, 2, (4,), verb_token, lexer, False),
                                      (vi_o_lex_pp, None, 2, (4,), verb_token, lexer, False),
                                      (vi_objc_cr_vc_CP, None, 2, (6,), verb_token, conj_lexer, False),
                                      (vi_objc_cr_vc_prep, None, 2, (6,), verb_token, prep_verber, False),
                                      (vi_objc_cr_vc_verb, None, 2, (5,), verb_token, lexer, False),
                                      (vi_objc_cr_nc_CP, None, 2, (6,), verb_token, conj_lexer, False),
                                  )




# 2.3, Verb Inventory, Objects, Semantic Domains

vi_o_sd_np = pred_target.format(basis=f'''

    phrase typ=NP|PrNP function=Objc
        -heads> word pdp#prep|prps|prde|prin|inrg sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds, ptcp_funct=all_ptcp
)

vi_o_sd_pp = pred_target.format(basis=f'''

    phrase typ=PP function=Objc
        -heads> word pdp=prep
        -prep_obj> word pdp#prep|prps|prde|prin|inrg sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds, ptcp_funct=all_ptcp
)

# Clause Relations
vi_objcSD_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Objc', reqs=f'sem_domain_code~{good_sem_codes}'), 
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_objcSD_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Objc', reqs=f'sem_domain_code~{good_sem_codes}'),
                                          pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_objcSD_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Objc', reqs=f'sem_domain_code~{good_sem_codes}'),
                                          pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_objcSD_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Objc', reqs=f'sem_domain_code~{good_sem_codes}'),
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)

def domainer2(basis, target):
    # basis tokenizer for semantic domains
    sem_domain = code2domain(basis)
    return sem_domain

def prep_o_domainer2(basis, target):
    # makes prep_domain + prep_obj_domain tokens
    prep_obj = E.prep_obj.f(basis)[0]
    sem_domain = code2domain(prep_obj)
    return f'{F.lex.v(basis)}_{sem_domain}'

def prep_verbDomainer2(basis, target):
    # combines a infinitive verb with its preposition
    prep = next(w for w in L.d(L.u(basis, 'phrase')[0], 'word') if F.pdp.v(w) == 'prep')
    prep_lex = F.lex.v(prep)
    sem_domain = code2domain(basis)
    return f'{prep_lex}_{sem_domain}'
    
def conj_domainer2(basis, target):
    # returns conjunction string + verb lex
    conj_phrase = next(ph for ph in L.d(L.u(basis, 'clause')[0], 'phrase') if F.typ.v(ph) == 'CP')
    conj_string = ''.join(F.lex.v(w) for w in L.d(conj_phrase, 'word'))
    sem_domain = code2domain(basis)
    return f'{conj_string}_{sem_domain}'
                                        
params['inventory']['vi_objc_domain'] = (
                                         (vi_o_sd_np, None, 2, (4,), verb_token,  domainer2, False),
                                         (vi_o_sd_pp, None, 2, (5,), verb_token, domainer2, False),
                                         (vi_objcSD_cr_vc_CP, None, 2, (6,), verb_token, conj_domainer2, False),
                                         (vi_objcSD_cr_vc_prep, None, 2, (6,), verb_token, prep_verbDomainer2, False),
                                         (vi_objcSD_cr_vc_verb, None, 2, (5,), verb_token, domainer2, False),
                                         (vi_objcSD_cr_nc_CP, None, 2, (6,), verb_token, conj_domainer2, False),
                                     )




# 2.4, Verb Inventory, Objects, Animacy

vi_o_an_np = pred_target.format(basis=f'''

    phrase typ=NP|PrNP function=Objc
        -heads> word pdp#verb|prep|prps|prde|prin|inrg sem_domain_code~{animacy_codes}

''', pred_funct=all_preds, ptcp_funct=all_ptcp
)

vi_o_an_pp = pred_target.format(basis=f'''

    phrase typ=PP function=Objc
        -heads> word pdp=prep
        -prep_obj> word pdp#verb|prep|prps|prde|prin|inrg sem_domain_code~{animacy_codes}

''', pred_funct=all_preds, ptcp_funct=all_ptcp
)

# Clause Relations – N.B. Only non-verbal clauses for animacy experiments
vi_objcAN_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Objc', reqs=f'sem_domain_code~{animacy_codes}'),
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
    
def conj_animater(basis, target):
    # returns conjunction string + verb animacy
    conj_phrase = next(ph for ph in L.d(L.u(basis, 'clause')[0], 'phrase') if F.typ.v(ph) == 'CP')
    conj_string = ''.join(F.lex.v(w) for w in L.d(conj_phrase, 'word'))
    animacy = code2animacy(F.sem_domain_code.v(basis))
    return f'{conj_string}_{animacy}'
                 
params['inventory']['vi_objc_animacy'] = (
                                             (vi_o_an_np, None, 2, (4,), verb_token,  animater, False),
                                             (vi_o_an_pp, None, 2, (5,), verb_token, animater, False),
                                             (vi_objcAN_cr_nc_CP, None, 2, (6,), verb_token, conj_animater, False),
                                         )




# 3.1, Verb Inventory, Complements, Presence/Absence

vi_cmp_pa = pred_target.format(basis='''

    phrase function=Cmpl

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vi_cmp_pa_clRel = pred_target.format(basis='''

c2:clause rela=Cmpl
c1 <mother- c2
    
''', pred_funct=all_preds, ptcp_funct=all_ptcp)


vi_cmp_pa_null = pred_target.format(basis='''

c2:clause
/without/
    phrase function=Cmpl
/-/
/without/
<mother- clause rela=Cmpl
/-/

c1 = c2
''', pred_funct=all_preds, ptcp_funct=all_ptcp)


params['inventory']['vi_cmpl_pa'] = (
                                        (vi_cmp_pa, None, 2, (3,), verb_token, functioner, True),
                                        (vi_cmp_pa_clRel, None, 2, (3,), verb_token, relationer, True),
                                        (vi_cmp_pa_null, None, 2, (3,), verb_token, nuller, True)
                                    )




# 3.2, Verb Inventory, Complements, Lexemes

vi_cmpl_lex_np = pred_target.format(basis='''

    phrase function=Cmpl typ=NP|PrNP|AdvP
        -heads> word pdp#prep|prps|prde|prin|inrg

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vi_cmpl_lex_pp = pred_target.format(basis='''

    phrase function=Cmpl typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp#prep|prps|prde|prin|inrg

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

# Clause Relations
vi_cmpl_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Cmpl', reqs=''), 
                                       pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_cmpl_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Cmpl', reqs=''),
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_cmpl_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Cmpl', reqs=''),
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_cmpl_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Cmpl', reqs=''),
                                      pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_cmpl_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Cmpl', reqs=''),
                                            pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_cmpl_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Cmpl', reqs=''),
                                            pred_funct=all_preds, ptcp_funct=all_ptcp)

        
params['inventory']['vi_cmpl_lex'] = (
                                         (vi_cmpl_lex_np, None, 2, (4,), verb_token, lexer, False),
                                         (vi_cmpl_lex_pp, None, 2, (4,), verb_token, prep_o_lexer, False),
                                         (vi_cmpl_cr_vc_CP, None, 2, (6,), verb_token, conj_lexer, False),
                                         (vi_cmpl_cr_vc_prep, None, 2, (6,), verb_token, prep_verber, False),
                                         (vi_cmpl_cr_vc_verb, None, 2, (5,), verb_token, lexer, False),
                                         (vi_cmpl_cr_nc_CP, None, 2, (6,), verb_token, conj_lexer, False),
                                         (vi_cmpl_cr_nc_Prec_adv, None, 2, (5,), verb_token, lexer, False),
                                         (vi_cmpl_cr_nc_Prec_prep, None, 2, (6,), verb_token, prep_verber, False),
                                     )




# 3.3, Verb Inventory, Complements, Semantic Domains

vi_cmpl_sd_np = pred_target.format(basis=f'''

    phrase function=Cmpl typ=NP|PrNP|AdvP
        -heads> word pdp#prep|prps|prde|prin|inrg sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vi_cmpl_sd_pp = pred_target.format(basis=f'''

    phrase function=Cmpl typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp#prep|prps|prde|prin|inrg sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

# Clause Relations
vi_cmplSD_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Cmpl', reqs=f'sem_domain_code~{good_sem_codes}'), 
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_cmplSD_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Cmpl', reqs=f'sem_domain_code~{good_sem_codes}'),
                                          pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_cmplSD_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Cmpl', reqs=f'sem_domain_code~{good_sem_codes}'),
                                          pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_cmplSD_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Cmpl', reqs=f'sem_domain_code~{good_sem_codes}'),
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_cmplSD_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Cmpl', reqs=f'sem_domain_code~{good_sem_codes}'),
                                              pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_cmplSD_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Cmpl', reqs=f'sem_domain_code~{good_sem_codes}'),
                                               pred_funct=all_preds, ptcp_funct=all_ptcp)
    
params['inventory']['vi_cmpl_domain'] = (
                                             (vi_cmpl_sd_np, None, 2, (4,), verb_token, domainer2, False),
                                             (vi_cmpl_sd_pp, None, 2, (4,), verb_token, prep_o_domainer2, False),
                                             (vi_cmplSD_cr_vc_CP, None, 2, (6,), verb_token, conj_domainer2, False),
                                             (vi_cmplSD_cr_vc_prep, None, 2, (6,), verb_token, prep_verbDomainer2, False),
                                             (vi_cmplSD_cr_vc_verb, None, 2, (5,), verb_token, domainer2, False),
                                             (vi_cmplSD_cr_nc_CP, None, 2, (6,), verb_token, conj_domainer2, False),
                                             (vi_cmplSD_cr_nc_Prec_adv, None, 2, (5,), verb_token, domainer2, False),
                                             (vi_cmplSD_cr_nc_Prec_prep, None, 2, (6,), verb_token, prep_verbDomainer2, False),
                                         )




# 3.4, Verb Inventory, Complements, Animacy

vi_cmpl_an_np = pred_target.format(basis=f'''

    phrase function=Cmpl typ=NP|PrNP|AdvP
        -heads> word pdp#verb|prep|prps|prde|prin|inrg sem_domain_code~{animacy_codes}

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vi_cmpl_an_pp = pred_target.format(basis=f'''

    phrase function=Cmpl typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp#verb|prep|prps|prde|prin|inrg sem_domain_code~{animacy_codes}

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

# Clause Relations; N.B. Only non-verbal clauses for animacy experiments
vi_cmplAN_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Cmpl', reqs=f'sem_domain_code~{animacy_codes} sp#verb'),
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
    
params['inventory']['vi_cmpl_animacy'] = (
                                             (vi_cmpl_an_np, None, 2, (4,), verb_token, animater, False),
                                             (vi_cmpl_an_pp, None, 2, (4,), verb_token, prep_o_animater, False),
                                             (vi_cmplAN_cr_nc_CP, None, 2, (6,), verb_token, conj_animater, False),
                                         )




# 4.1, Verb Inventory, Adjuncts +(Location, Time, PrAd), Presence/Absence

vi_adj_pa = pred_target.format(basis='''

    phrase function=Adju|Time|Loca|PrAd

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vi_adj_pa_clRel = pred_target.format(basis='''

c2:clause rela=Adju|PrAd
    c1 <mother- c2
    
''', pred_funct=all_preds, ptcp_funct=all_ptcp)


vi_adj_pa_null = pred_target.format(basis='''

c2:clause
/without/
    phrase function=Adju|Time|Loca|PrAd
/-/
/without/
<mother- clause rela=Adju|PrAd
/-/
    
c1 = c2

''', pred_funct=all_preds, ptcp_funct=all_ptcp)


params['inventory']['vi_adj+_pa'] = (
                                        (vi_adj_pa, None, 2, (3,), verb_token, functioner, True),
                                        (vi_adj_pa_clRel, None, 2, (3,), verb_token, relationer, True),
                                        (vi_adj_pa_null, None, 2, (3,), verb_token, nuller, True)
                                    )




# 4.2, Verb Inventory, Adjuncts+, Lexemes

vi_adj_lex_np = pred_target.format(basis='''

    phrase function=Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word pdp#prep|prps|prde|prin|inrg

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vi_adj_lex_pp = pred_target.format(basis='''

    phrase function=Adju|Time|Loca|PrAd typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp#prep|prps|prde|prin|inrg

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

# Clause Relations
vi_adj_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Adju|PrAd', reqs=''), 
                                     pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_adj_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Adju|PrAd', reqs=''),
                                       pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_adj_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Adju|PrAd', reqs=''),
                                       pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_adj_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Adju|PrAd', reqs=''),
                                     pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_adj_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Adju|PrAd', reqs=''),
                                           pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_adj_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Adju|PrAd', reqs=''),
                                            pred_funct=all_preds, ptcp_funct=all_ptcp)
    
params['inventory']['vi_adj+_lex'] = (
                                         (vi_adj_lex_np, None, 2, (4,), verb_token, lexer, False),
                                         (vi_adj_lex_pp, None, 2, (4,), verb_token, prep_o_lexer, False),
                                         (vi_adj_cr_vc_CP, None, 2, (6,), verb_token, conj_lexer, False),
                                         (vi_adj_cr_vc_prep, None, 2, (6,), verb_token, prep_verber, False),
                                         (vi_adj_cr_vc_verb, None, 2, (5,), verb_token, lexer, False),
                                         (vi_adj_cr_nc_CP, None, 2, (6,), verb_token, conj_lexer, False),
                                         (vi_adj_cr_nc_Prec_adv, None, 2, (5,), verb_token, lexer, False),
                                         (vi_adj_cr_nc_Prec_prep, None, 2, (6,), verb_token, prep_verber, False),
                                     )




# 4.3, Verb Inventory, Adjuncts+, Semantic Domains

vi_adj_sd_np = pred_target.format(basis=f'''

    phrase function=Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word pdp#prep|prps|prde|prin|inrg sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vi_adj_sd_pp = pred_target.format(basis=f'''

    phrase function=Adju|Time|Loca|PrAd typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp#prep|prps|prde|prin|inrg sem_domain_code~{good_sem_codes}

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

# Clause Relations
vi_adjSD_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Adju|PrAd', reqs=f'sem_domain_code~{good_sem_codes}'), 
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_adjSD_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Adju|PrAd', reqs=f'sem_domain_code~{good_sem_codes}'),
                                          pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_adjSD_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Adju|PrAd', reqs=f'sem_domain_code~{good_sem_codes}'),
                                          pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_adjSD_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Adju|PrAd', reqs=f'sem_domain_code~{good_sem_codes}'),
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_adjSD_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Adju|PrAd', reqs=f'sem_domain_code~{good_sem_codes}'),
                                              pred_funct=all_preds, ptcp_funct=all_ptcp)
vi_adjSD_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Adju|PrAd', reqs=f'sem_domain_code~{good_sem_codes}'),
                                               pred_funct=all_preds, ptcp_funct=all_ptcp)

    
params['inventory']['vi_adj+_domain'] = (
                                             (vi_adj_sd_np, None, 2, (4,), verb_token, domainer2, False),
                                             (vi_adj_sd_pp, None, 2, (4,), verb_token, prep_o_domainer2, False),
                                             (vi_adjSD_cr_vc_CP, None, 2, (6,), verb_token, conj_domainer2, False),
                                             (vi_adjSD_cr_vc_prep, None, 2, (6,), verb_token, prep_verbDomainer2, False),
                                             (vi_adjSD_cr_vc_verb, None, 2, (5,), verb_token, domainer2, False),
                                             (vi_adjSD_cr_nc_CP, None, 2, (6,), verb_token, conj_domainer2, False),
                                             (vi_adjSD_cr_nc_Prec_adv, None, 2, (5,), verb_token, domainer2, False),
                                             (vi_adjSD_cr_nc_Prec_prep, None, 2, (6,), verb_token, prep_verbDomainer2, False),
                                        )




# 4.4, Verb Inventory, Adjuncts+, Animacy

vi_adj_an_np = pred_target.format(basis=f'''

    phrase function=Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word pdp#verb|prep|prps|prde|prin|inrg sem_domain_code~{animacy_codes}

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vi_adj_an_pp = pred_target.format(basis=f'''

    phrase function=Adju|Time|Loca|PrAd typ=PP
        -heads> word pdp=prep
        -prep_obj> word pdp#verb|prep|prps|prde|prin|inrg sem_domain_code~{animacy_codes}

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

# Clause Relations; NB only non-verbals for animacy
vi_adjAN_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Adju|PrAd', reqs=f'sem_domain_code~{good_sem_codes} sp#verb'),

    
params['inventory']['vi_adj+_animacy'] = (
                                             (vi_adj_sd_np, None, 2, (4,), verb_token, animater, False),
                                             (vi_adj_sd_pp, None, 2, (4,), verb_token, prep_o_animater, False),
                                             (vi_adjSD_cr_nc_CP, None, 2, (6,), verb_token, conj_animater, False),
                                        )






# 5.1, Verb Frames, All Arguments, Presence/Absence

vf_allarg_pa = pred_target.format(basis='''

c2:clause
/without/
    phrase function=Rela
/-/
    
    phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd
    
c1 = c2
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_allarg_pa_clRela = pred_target.format(basis='''

c2:clause rela=Objc|Cmpl|Adju|PrAd

c1 <mother- c2

''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_allarg_pa_null = pred_target.format(basis='''

c2:clause
/without/
<mother- clause rela=Objc|Cmpl|Adju|PrAd
/-/

/without/
<mother- phrase rela=PrAd
/-/

/without/
    phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd|Rela
/-/

/without/
    ca:clause_atom
speech:clause_atom code=999
ca <mother- speech
/-/

c1 = c2
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

# NB, a trick was required in this template so that the correct index
# can be provided to the validateFrame mothers class, i=3
# c3:clause must thus be awkwardly put first; when combined with target it becomes i=3
vf_allarg_pa_speech = pred_target.format(basis='''

c3:clause
c2:clause
/without/
    phrase function=Rela
/-/
    ca1:clause_atom
    
ca2:clause_atom code=999

c1 = c2
ca1 <mother- ca2
ca1 <: ca2
ca1 [[ p1
c3 [[ ca2
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

# NB: This template is required to validate clauses with no matched
# relations within the clause, but an exteral clause relation
vf_allarg_simpleNull = pred_target.format(basis='''

c2:clause
/without/
    phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd|Rela
/-/

c1 = c2
''', pred_funct='PreS|Pred', ptcp_funct='PreC')

vf_allarg_pa_suffix = pred_target.format(basis='''

c2:clause
/without/
    phrase function=Rela
/-/    
c1 = c2

''', pred_funct='PreO', ptcp_funct='PtcO')

if not cached_data:
    allArgVal = validateFrame(mother_templates=(vf_allarg_pa,
                                                vf_allarg_pa_suffix,
                                                vf_allarg_simpleNull),
                          daughter_templates=(vf_allarg_pa_clRela,
                                              vf_allarg_pa_speech),
                          exp_name='vf_argAll_pa')
    cache['allArgVal'] = allArgVal
else:
    allArgVal = cache['allArgVal']
    

params['frame']['vf_argAll_pa'] = (
                                      (vf_allarg_pa, allArgVal.daughters, 2, (4,), verb_token, functioner, False),
                                      (vf_allarg_pa_suffix, allArgVal.daughters, 2, (2,), verb_token, simple_object, False),
                                      (vf_allarg_pa_clRela, allArgVal.mothers, 2, (3,), verb_token, relationer, False),
                                      (vf_allarg_pa_speech, allArgVal.mothers, 2, (2,), verb_token, simple_object, False),
                                      (vf_allarg_pa_null, None, 2, (2,), verb_token, nuller, False)
                                  )




# 5.2, Verb Frames, All Arguments, Lexemes

# rules:
# • Select only lexical elements (e.g. not pronouns) within the frame with all argument functions
# • Any non-lexical elements with an argument function excludes the frame altogether.

# a set of conditions that must hold true
# for all frame elements that are selected
# at the clause level:
vf_clause_conditions = '''

c2:clause
/without/
    phrase function={relas} typ#NP|PrNP|AdvP|PP
/-/
{clause_reqs}

/where/
    phrase function={relas} typ#PP
/have/
    /where/
        -heads> w1:word
    /have/
        w2:word pdp#prep|prps|prde|prin|inrg {word_reqs}
        w1 = w2
    /-/
/-/

/where/
    phrase function={relas} typ=PP
/have/
    /where/
        -heads> word pdp=prep
    /have/
        -prep_obj> word pdp#prep|prps|prde|prin|inrg {word_reqs}
    /-/
/-/

c1 = c2
'''

vf_all_arg_conditions = vf_clause_conditions.format(relas='Objc|Cmpl|Adju|Time|Loca|PrAd', 
                                                    word_reqs='',
                                                    clause_reqs='/without/\n    phrase function=Rela\n/-/')

vf_allarg_lex_np = pred_target.format(basis=f'''

{vf_all_arg_conditions}

    phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word
        
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

vf_allarg_lex_pp = pred_target.format(basis=f'''

{vf_all_arg_conditions}

    phrase function=Cmpl|Adju|Time|Loca|PrAd typ=PP
        -heads> word
        -prep_obj> word
        
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

vf_allarg_lex_pp_obj = pred_target.format(basis=f'''

{vf_all_arg_conditions}

    phrase function=Objc typ=PP
        -heads> word
        -prep_obj> word
     
''', pred_funct='Pred|PreS', ptcp_funct='PreC')
    
# Clause Relations
vf_args_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Objc|Cmpl|Adju', reqs=''), 
                                      pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_args_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Objc|Cmpl|Adju', reqs=''),
                                        pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_args_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Objc|Cmpl|Adju', reqs=''),
                                        pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_args_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Objc|Cmpl|Adju', reqs=''),
                                      pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_args_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Cmpl|Adju', reqs=''),
                                            pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_args_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Cmpl|Adju', reqs=''),
                                             pred_funct='Pred|PreS', ptcp_funct='PreC')

def funct_lexer(basis, target):
    # returns function + lexeme basis tokens
    function = F.function.v(L.u(basis, 'phrase')[0])
    return f'{function}.{F.lex.v(basis)}'

def funct_prep_o_lexer(basis, target):
    # returns function + preplex + preplex object
    function = F.function.v(L.u(basis, 'phrase')[0])
    prep_obj = E.prep_obj.f(basis)[0]
    return f'{function}.{F.lex.v(basis)}_{F.lex.v(prep_obj)}'

if not cached_data:
    valLex = validateFrame(mother_templates=(vf_allarg_lex_np,
                                             vf_allarg_lex_pp, 
                                             vf_allarg_lex_pp_obj,
                                             vf_allarg_simpleNull),
                           daughter_templates = (vf_args_cr_vc_CP,
                                                 vf_args_cr_vc_prep, 
                                                 vf_args_cr_vc_verb,
                                                 vf_args_cr_nc_CP,
                                                 vf_args_cr_nc_Prec_adv,
                                                 vf_args_cr_nc_Prec_prep),
                           exp_name='vf_allarg_lex')
    cache['valLex'] = valLex
else:
    valLex = cache['valLex']
    
params['frame']['vf_argAll_lex'] = (
                                        (vf_allarg_lex_np, valLex.daughters, 2, (5,), verb_token, funct_lexer, False),
                                        (vf_allarg_lex_pp_obj, valLex.daughters, 2, (6,), verb_token, funct_lexer, False),
                                        (vf_allarg_lex_pp, valLex.daughters, 2, (5,), verb_token, funct_prep_o_lexer, False),
                                        (vf_args_cr_vc_CP, valLex.mothers, 2, (6,), verb_token, rela_conj_lexer, False),
                                        (vf_args_cr_vc_prep, valLex.mothers, 2, (6,), verb_token, rela_prep_lexer, False),
                                        (vf_args_cr_vc_verb, valLex.mothers, 2, (5,), verb_token, rela_lexer, False),
                                        (vf_args_cr_nc_CP, valLex.mothers, 2, (6,), verb_token, rela_conj_lexer, False),
                                        (vf_args_cr_nc_Prec_adv, valLex.mothers, 2, (5,), verb_token, rela_lexer, False),
                                        (vf_args_cr_nc_Prec_prep, valLex.mothers, 2, (6,), verb_token, rela_prep_lexer, False),
                                    )



# 5.3, Verb Frames, All Arguments, Semantic Domains

vf_all_arg_conditionsSD = vf_clause_conditions.format(relas='Objc|Cmpl|Adju|Time|Loca|PrAd', 
                                                      word_reqs=f'sem_domain_code~{good_sem_codes}',                                  
                                                      clause_reqs='/without/\n    phrase function=Rela\n/-/')

vf_allarg_sd_np = pred_target.format(basis=f'''

{vf_all_arg_conditionsSD}

    phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word
        
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

vf_allarg_sd_pp = pred_target.format(basis=f'''

{vf_all_arg_conditionsSD}

    phrase function=Cmpl|Adju|Time|Loca|PrAd typ=PP
        -heads> word
        -prep_obj> word
        
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

vf_allarg_sd_pp_obj = pred_target.format(basis=f'''

{vf_all_arg_conditionsSD}

    phrase function=Objc typ=PP
        -heads> word
        -prep_obj> word
     
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

# Clause Relations
vf_argsSD_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Objc|Cmpl|Adju', 
                                                               reqs=f'sem_domain_code~{good_sem_codes}'), 
                                                               pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_argsSD_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Objc|Cmpl|Adju', 
                                                                   reqs=f'sem_domain_code~{good_sem_codes}'),
                                                                   pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_argsSD_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Objc|Cmpl|Adju', 
                                                                   reqs=f'sem_domain_code~{good_sem_codes}'),
                                                                   pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_argsSD_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Objc|Cmpl|Adju', 
                                                               reqs=f'sem_domain_code~{good_sem_codes}'),
                                                               pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_argsSD_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Cmpl|Adju',
                                                                           reqs=f'sem_domain_code~{good_sem_codes}'),
                                                                           pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_argsSD_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Cmpl|Adju',
                                                                             reqs=f'sem_domain_code~{good_sem_codes}'),
                                                                             pred_funct='Pred|PreS', ptcp_funct='PreC')
if not cached_data:
    valSD = validateFrame(mother_templates=(vf_allarg_sd_np,
                                            vf_allarg_sd_pp, 
                                            vf_allarg_sd_pp_obj,
                                            vf_allarg_simpleNull),
                          daughter_templates = (vf_argsSD_cr_vc_CP,
                                                vf_argsSD_cr_vc_prep, 
                                                vf_argsSD_cr_vc_verb,
                                                vf_argsSD_cr_nc_CP,
                                                vf_argsSD_cr_nc_Prec_adv,
                                                vf_argsSD_cr_nc_Prec_prep),
                          exp_name='vf_allarg_sd')
    cache['valSD'] = valSD
else:
    valSD = cache['valSD']


def funct_domainer2(basis, target):
    # basis tokenizer for semantic domains + functions
    function = F.function.v(L.u(basis, 'phrase')[0])
    sem_domain = code2domain(basis)
    return f'{function}.{sem_domain}'
    
def funct_prep_o_domainer2(basis, target):
    # makes prep_domain + prep_obj_domain tokens + functions
    prep_obj = E.prep_obj.f(basis)[0]
    sem_domain = code2domain(prep_obj)
    function = F.function.v(L.u(basis, 'phrase')[0])
    return f'{function}.{F.lex.v(basis)}_{sem_domain}'

def rela_prep_domainer2(basis, target):
    # returns clause relation + prep + verb domain
    rela = F.rela.v(L.u(basis, 'clause')[0])
    prep = next(w for w in L.d(L.u(basis, 'phrase')[0], 'word') if F.pdp.v(w) == 'prep')
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
    rela = F.rela.v(L.u(basis, 'clause')[0])
    sem_domain = code2domain(basis)
    return f'{rela}.{sem_domain}'

params['frame']['vf_argAll_domain'] = (
                                          (vf_allarg_sd_np, valSD.daughters, 2, (5,), verb_token, funct_domainer2, False),
                                          (vf_allarg_sd_pp, valSD.daughters, 2, (5,), verb_token, funct_prep_o_domainer2, False),
                                          (vf_allarg_sd_pp_obj, valSD.daughters, 2, (6,), verb_token, funct_domainer2, False),
                                          (vf_argsSD_cr_vc_CP, valSD.mothers, 2, (6,), verb_token, rela_conj_domainer2, False),
                                          (vf_argsSD_cr_vc_prep, valSD.mothers, 2, (6,), verb_token, rela_prep_domainer2, False),
                                          (vf_argsSD_cr_vc_verb, valSD.mothers, 2, (5,), verb_token, rela_domainer2, False),
                                          (vf_argsSD_cr_nc_CP, valSD.mothers, 2, (6,), verb_token, rela_conj_domainer2, False),
                                          (vf_argsSD_cr_nc_Prec_adv, valSD.mothers, 2, (5,), verb_token, rela_domainer2, False),
                                          (vf_argsSD_cr_nc_Prec_prep, valSD.mothers, 2, (6,), verb_token, rela_prep_domainer2, False),
                                      )




# 5.4, Verb Frames, All Arguments, Animacy
                                       
vf_all_arg_conditionsAN = vf_clause_conditions.format(relas='Objc|Cmpl|Adju|Time|Loca|PrAd', 
                                                      word_reqs=f'sem_domain_code~{animacy_codes} sp#verb',                                  
                                                      clause_reqs='/without/\n    phrase function=Rela\n/-/')

vf_allarg_an_np = pred_target.format(basis=f'''

{vf_all_arg_conditionsAN}

    phrase function=Objc|Cmpl|Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word
        
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

vf_allarg_an_pp = pred_target.format(basis=f'''

{vf_all_arg_conditionsAN}

    phrase function=Cmpl|Adju|Time|Loca|PrAd typ=PP
        -heads> word
        -prep_obj> word
        
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

vf_allarg_an_pp_obj = pred_target.format(basis=f'''

{vf_all_arg_conditionsAN}

    phrase function=Objc typ=PP
        -heads> word
        -prep_obj> word
     
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

# Clause Relations; NB only non-verbal for animacy experiments
vf_argsAN_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Objc|Cmpl|Adju', 
                                                               reqs=f'sem_domain_code~{animacy_codes} sp#verb'),
                                                               pred_funct='Pred|PreS', ptcp_funct='PreC')

if not cached_data:
    valArgAN = validateFrame(mother_templates=(vf_allarg_an_np,
                                               vf_allarg_an_pp, 
                                               vf_allarg_an_pp_obj,
                                               vf_allarg_simpleNull),
                          daughter_templates = (vf_argsAN_cr_nc_CP,),
                          exp_name='vf_allarg_an')
    cache['valArgAN'] = valArgAN
else:
    valArgAN = cache['valArgAN']

def funct_animater(basis, target):
    # basis tokenizer for semantic domains + functions
    function = F.function.v(L.u(basis, 'phrase')[0])
    animacy = code2animacy(basis)
    return f'{function}.{animacy}'
    
def funct_prep_o_animater(basis, target):
    # makes prep_domain + prep_obj_domain tokens + functions
    prep_obj = E.prep_obj.f(basis)[0]
    animacy = code2animacy(prep_obj)
    function = F.function.v(L.u(basis, 'phrase')[0])
    return f'{function}.{F.lex.v(basis)}_{animacy}'

def rela_conj_animater(basis, target):
    # returns clause relation + conjunction string + animacy
    rela = F.rela.v(L.u(basis, 'clause')[0])
    conj_phrase = next(ph for ph in L.d(L.u(basis, 'clause')[0], 'phrase') if F.typ.v(ph) == 'CP')
    conj_string = ''.join(F.lex.v(w) for w in L.d(conj_phrase, 'word'))
    animacy = code2animacy(basis)
    return f'{rela}.{conj_string}_{animacy}'
   
params['frame']['vf_argAll_domain2'] = (
                                          (vf_allarg_sd_np, valArgAN.daughters, 2, (5,), verb_token, funct_animater, False),
                                          (vf_allarg_sd_pp, valArgAN.daughters, 2, (5,), verb_token, funct_prep_o_animater, False),
                                          (vf_allarg_sd_pp_obj, valArgAN.daughters, 2, (6,), verb_token, funct_animater, False),
                                          (vf_argsSD_cr_nc_CP, valArgAN.mothers, 2, (6,), verb_token, rela_conj_animater, False),

                                      )




# 6.1, Verb Frame, Objects, Presence/Absence
vf_obj_pa = pred_target.format(basis='''

c2:clause
/without/
    phrase function=Rela
/-/
    phrase function=Objc
    
c1 = c2
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_obj_pa_clRela = pred_target.format(basis='''

c2:clause rela=Objc

c1 <mother- c2
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_obj_pa_null = pred_target.format(basis='''

c2:clause
/without/
<mother- clause rela=Objc
/-/
/without/
    phrase function=Objc|Rela
/-/
/without/
    ca:clause_atom
speech:clause_atom code=999
ca <mother- speech
/-/

c1 = c2
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

vf_obj_pa_suffix = pred_target.format(basis='''

c2:clause
/without/
    phrase function=Rela
/-/    

c1 = c2
''', pred_funct='PreO', ptcp_funct='PtcO')

vf_obj_pa_speech = vf_allarg_pa_speech

vf_obj_simpleNull = pred_target.format(basis='''

c2:clause
/without/
    phrase function=Objc|PreO|PtcO|Rela
/-/

c1 = c2
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

if not cached_data:
    vfObjPa = validateFrame(mother_templates=(vf_obj_pa, 
                                              vf_obj_pa_suffix,
                                              vf_obj_simpleNull),
                              daughter_templates=(vf_obj_pa_clRela,
                                                  vf_obj_pa_speech),
                              exp_name='vf_obj_pa')
    cache['vfObjPa'] = vfObjPa
else:
    vfObjPa = cache['vfObjPa']

params['frame']['vf_obj_pa'] = (
                                    (vf_obj_pa, vfObjPa.daughters, 2, (4,), verb_token, functioner, False),
                                    (vf_obj_pa_suffix, vfObjPa.daughters, 2, (2,), verb_token, simple_object, False),
                                    (vf_obj_pa_clRela, vfObjPa.mothers, 2, (3,), verb_token, relationer, False),
                                    (vf_obj_pa_speech, vfObjPa.mothers, 2, (2,), verb_token, simple_object, False),
                                    (vf_obj_pa_null, None, 2, (2,), verb_token, nuller, False)
                                )


    
    
# 6.2, Verb Frame, Objects, Lexemes 
vf_obj_arg_conditions = vf_clause_conditions.format(relas='Objc', 
                                                    word_reqs='',                                  
                                                    clause_reqs='/without/\n    phrase function=Rela\n/-/')

vf_obj_lex_np = pred_target.format(basis=f'''

{vf_obj_arg_conditions}

    phrase function=Objc typ#PP
        -heads> word 
        
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

vf_obj_lex_pp = pred_target.format(basis=f'''

{vf_obj_arg_conditions}

    phrase function=Objc typ=PP
        -heads> word
        -prep_obj> word
     
''', pred_funct='Pred|PreS', ptcp_funct='PreC')
    
# Clause Relations
vf_obj_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Objc', reqs=''), 
                                      pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_obj_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Objc', reqs=''),
                                        pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_obj_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Objc', reqs=''),
                                        pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_obj_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Objc', reqs=''),
                                      pred_funct='Pred|PreS', ptcp_funct='PreC')

if not cached_data:
    valObjLex = validateFrame(mother_templates=(vf_obj_lex_np,
                                                vf_obj_lex_pp,
                                                vf_obj_simpleNull),
                           daughter_templates = (vf_obj_cr_vc_CP,
                                                 vf_obj_cr_vc_prep, 
                                                 vf_obj_cr_vc_verb,
                                                 vf_obj_cr_nc_CP),
                           exp_name='vf_obj_lex')
    cache['valObjLex'] = valObjLex
else:
    valObjLex = cache['valObjLex']
    
params['frame']['vf_obj_lex'] = (
                                    (vf_obj_lex_np, valObjLex.daughters, 2, (5,), verb_token, funct_lexer, False),
                                    (vf_obj_lex_pp, valObjLex.daughters, 2, (6,), verb_token, funct_lexer, False),
                                    (vf_obj_cr_vc_CP, valObjLex.mothers, 2, (6,), verb_token, rela_conj_lexer, False),
                                    (vf_obj_cr_vc_prep, valObjLex.mothers, 2, (6,), verb_token, rela_prep_lexer, False),
                                    (vf_obj_cr_vc_verb, valObjLex.mothers, 2, (5,), verb_token, rela_lexer, False),
                                    (vf_obj_cr_nc_CP, valObjLex.mothers, 2, (6,), verb_token, rela_conj_lexer, False),
                                )




# 6.3, Verb Frame, Objects, Semantic Domains
vf_obj_arg_conditionsSD = vf_clause_conditions.format(relas='Objc', 
                                                      word_reqs=f'sem_domain_code~{good_sem_codes}',                                  
                                                      clause_reqs='/without/\n    phrase function=Rela\n/-/')

vf_obj_sd_np = pred_target.format(basis=f'''

{vf_obj_arg_conditionsSD}

    phrase function=Objc typ=NP|PrNP|AdvP
        -heads> word
        
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

vf_obj_sd_pp = pred_target.format(basis=f'''

{vf_obj_arg_conditionsSD}

    phrase function=Objc typ=PP
        -heads> word
        -prep_obj> word
     
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

# Clause Relations
vf_objSD_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Objc', 
                                                              reqs=f'sem_domain_code~{good_sem_codes}'), 
                                       pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_objSD_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Objc', 
                                                                  reqs=f'sem_domain_code~{good_sem_codes}'),
                                         pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_objSD_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Objc', 
                                                                  reqs=f'sem_domain_code~{good_sem_codes}'),
                                         pred_funct='Pred|PreS', ptcp_funct='PreC')
vf_objSD_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Objc', 
                                                               reqs=f'sem_domain_code~{good_sem_codes}'),
                                       pred_funct='Pred|PreS', ptcp_funct='PreC')

if not cached_data:
    valObjSD = validateFrame(mother_templates=(vf_obj_sd_np,
                                               vf_obj_sd_pp,
                                               vf_obj_simpleNull),
                             daughter_templates = (vf_objSD_cr_vc_CP,
                                                   vf_objSD_cr_vc_prep, 
                                                   vf_objSD_cr_vc_verb,
                                                   vf_objSD_cr_nc_CP),
                             exp_name='vf_obj_sd')
    cache['valObjSD'] = valObjSD
else:
    valObjSD = cache['valObjSD']
    
params['frame']['vf_obj_domain'] = (
                                          (vf_obj_sd_np, valObjSD.daughters, 2, (5,), verb_token, funct_domainer2, False),
                                          (vf_obj_sd_pp, valObjSD.daughters, 2, (6,), verb_token, funct_domainer2, False),
                                          (vf_objSD_cr_vc_CP, valObjSD.mothers, 2, (6,), verb_token, rela_conj_domainer2, False),
                                          (vf_objSD_cr_vc_prep, valObjSD.mothers, 2, (6,), verb_token, rela_prep_domainer2, False),
                                          (vf_objSD_cr_vc_verb, valObjSD.mothers, 2, (5,), verb_token, rela_domainer2, False),
                                          (vf_objSD_cr_nc_CP, valObjSD.mothers, 2, (6,), verb_token, rela_conj_domainer2, False)
                                      )

                                       
                                       
                                       
# 6.4, Verb Frame, Objects, Animacy
vf_obj_arg_conditionsAN = vf_clause_conditions.format(relas='Objc', 
                                                      word_reqs=f'sem_domain_code~{animacy_codes} sp#verb',                                  
                                                      clause_reqs='/without/\n    phrase function=Rela\n/-/')

vf_obj_an_np = pred_target.format(basis=f'''

{vf_obj_arg_conditionsAN}

    phrase function=Objc typ=NP|PrNP|AdvP
        -heads> word
        
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

vf_obj_an_pp = pred_target.format(basis=f'''

{vf_obj_arg_conditionsAN}

    phrase function=Objc typ=PP
        -heads> word
        -prep_obj> word
     
''', pred_funct='Pred|PreS', ptcp_funct='PreC')

# Clause Relations
vf_objAN_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Objc', 
                                                              reqs=f'sem_domain_code~{animacy_codes} sp#verb'),
                                       pred_funct='Pred|PreS', ptcp_funct='PreC')

if not cached_data:
    valObjAN = validateFrame(mother_templates=(vf_obj_an_np,
                                               vf_obj_an_pp,
                                               vf_obj_simpleNull),
                             daughter_templates = (vf_objAN_cr_nc_CP),
                             exp_name='vf_obj_an')
    cache['valObjAN'] = valObjAN
else:
    valObjAN = cache['valObjAN']
    
params['frame']['vf_obj_animacy'] = (
                                          (vf_obj_an_np, valObjAN.daughters, 2, (5,), verb_token, funct_animater, False),
                                          (vf_obj_sd_pp, valObjAN.daughters, 2, (6,), verb_token, funct_animater, False),
                                          (vf_objSD_cr_nc_CP, valObjAN.mothers, 2, (6,), verb_token, rela_conj_animater, False)
                                      )




# 7.1, Verb Frame, Complements, Presence/Absence
vf_cmpl_pa = pred_target.format(basis='''

    phrase function=Cmpl
    
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_cmpl_pa_clRela = pred_target.format(basis='''

c2:clause rela=Cmpl

c1 <mother- c2
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_cmpl_pa_null = pred_target.format(basis='''

c2:clause
/without/
<mother- clause rela=Cmpl
/-/
/without/
    phrase function=Cmpl
/-/

c1 = c2
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_cmpl_simpleNull = pred_target.format(basis='''

c2:clause
/without/
    phrase function=Cmpl
/-/

c1 = c2
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

if not cached_data:
    vfCmplPa = validateFrame(mother_templates=(vf_cmpl_pa, 
                                               vf_cmpl_simpleNull),
                              daughter_templates=(vf_cmpl_pa_clRela,),
                              exp_name='vf_cmpl_pa')
    cache['vfCmplPa'] = vfCmplPa
else:
    vfCmplPa = cache['vfCmplPa']


params['frame']['vf_cmpl_pa'] = (
                                    (vf_cmpl_pa, vfCmplPa.daughters, 2, (3,), verb_token, functioner, False),
                                    (vf_cmpl_pa_clRela, vfCmplPa.mothers, 2, (3,), verb_token, relationer, False),
                                    (vf_cmpl_pa_null, None, 2, (2,), verb_token, nuller, False)
                                )




# 7.2, Verb Frame, Complements, Lexemes

vf_cmpl_conditions = vf_clause_conditions.format(relas='Cmpl', 
                                                 word_reqs='',
                                                 clause_reqs='')

vf_cmpl_lex_np = pred_target.format(basis=f'''

{vf_cmpl_conditions}

    phrase function=Cmpl typ=NP|PrNP|AdvP
        -heads> word
        
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_cmpl_lex_pp = pred_target.format(basis=f'''

{vf_cmpl_conditions}

    phrase function=Cmpl typ=PP
        -heads> word
        -prep_obj> word
        
''', pred_funct=all_preds, ptcp_funct=all_ptcp)


# Clause Relations
vf_cmpl_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Cmpl', reqs=''), 
                                      pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_cmpl_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Cmpl', reqs=''),
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_cmpl_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Cmpl', reqs=''),
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_cmpl_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Cmpl', reqs=''),
                                      pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_cmpl_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Cmpl', reqs=''),
                                            pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_cmpl_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Cmpl', reqs=''),
                                             pred_funct=all_preds, ptcp_funct=all_ptcp)
if not cached_data:
    valCmplLex = validateFrame(mother_templates=(vf_cmpl_lex_np,
                                                 vf_cmpl_lex_pp,
                                                 vf_cmpl_simpleNull),
                               daughter_templates = (vf_cmpl_cr_vc_CP,
                                                     vf_cmpl_cr_vc_prep, 
                                                     vf_cmpl_cr_vc_verb,
                                                     vf_cmpl_cr_nc_CP,
                                                     vf_cmpl_cr_nc_Prec_adv,
                                                     vf_cmpl_cr_nc_Prec_prep),
                               exp_name='vf_cmpl_lex')
    cache['valCmplLex'] = valCmplLex
else:
    valCmplLex = cache['valCmplLex']
    
params['frame']['vf_cmpl_lex'] = (
                                        (vf_cmpl_lex_np, valCmplLex.daughters, 2, (5,), verb_token, funct_lexer, False),
                                        (vf_cmpl_lex_pp, valCmplLex.daughters, 2, (5,), verb_token, funct_prep_o_lexer, False),
                                        (vf_cmpl_cr_vc_CP, valCmplLex.mothers, 2, (6,), verb_token, rela_conj_lexer, False),
                                        (vf_cmpl_cr_vc_prep, valCmplLex.mothers, 2, (6,), verb_token, rela_prep_lexer, False),
                                        (vf_cmpl_cr_vc_verb, valCmplLex.mothers, 2, (5,), verb_token, rela_lexer, False),
                                        (vf_cmpl_cr_nc_CP, valCmplLex.mothers, 2, (6,), verb_token, rela_conj_lexer, False),
                                        (vf_cmpl_cr_nc_Prec_adv, valCmplLex.mothers, 2, (5,), verb_token, rela_lexer, False),
                                        (vf_cmpl_cr_nc_Prec_prep, valCmplLex.mothers, 2, (6,), verb_token, rela_prep_lexer, False),
                                    )



# 7.3, Verb Frame, Complements, Domains


vf_cmpl_conditionsSD = vf_clause_conditions.format(relas='Cmpl', 
                                                   word_reqs=f'sem_domain_code~{good_sem_codes}',                                  
                                                   clause_reqs='')

vf_cmpl_sd_np = pred_target.format(basis=f'''

{vf_cmpl_conditionsSD}

    phrase function=Cmpl typ=NP|PrNP|AdvP
        -heads> word
        
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_cmpl_sd_pp = pred_target.format(basis=f'''

{vf_cmpl_conditionsSD}

    phrase function=Cmpl typ=PP
        -heads> word
        -prep_obj> word
        
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

# Clause Relations
vf_cmplSD_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Cmpl', 
                                                               reqs=f'sem_domain_code~{good_sem_codes}'), 
                                                               pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_cmplSD_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Cmpl', 
                                                                   reqs=f'sem_domain_code~{good_sem_codes}'),
                                                                    pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_cmplSD_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Cmpl', 
                                                                   reqs=f'sem_domain_code~{good_sem_codes}'),
                                                                   pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_cmplSD_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Cmpl', 
                                                               reqs=f'sem_domain_code~{good_sem_codes}'),
                                                               pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_cmplSD_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Cmpl',
                                                                           reqs=f'sem_domain_code~{good_sem_codes}'),
                                                                           pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_cmplSD_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Cmpl',
                                                                             reqs=f'sem_domain_code~{good_sem_codes}'),
                                                                             pred_funct=all_preds, ptcp_funct=all_ptcp)
if not cached_data:
    valCmplSD = validateFrame(mother_templates=(vf_cmpl_sd_np,
                                                vf_cmpl_sd_pp, 
                                                vf_cmpl_simpleNull),
                          daughter_templates = (vf_cmplSD_cr_vc_CP,
                                                vf_cmplSD_cr_vc_prep, 
                                                vf_cmplSD_cr_vc_verb,
                                                vf_cmplSD_cr_nc_CP,
                                                vf_cmplSD_cr_nc_Prec_adv,
                                                vf_cmplSD_cr_nc_Prec_prep),
                          exp_name='vf_cmpl_sd')
    cache['valCmplSD'] = valCmplSD
else:
    valCmplSD = cache['valCmplSD']

    
params['frame']['vf_cmpl_domain'] = (
                                          (vf_cmpl_sd_np, valCmplSD.daughters, 2, (5,), verb_token, funct_domainer2, False),
                                          (vf_cmpl_sd_pp, valCmplSD.daughters, 2, (5,), verb_token, funct_prep_o_domainer2, False),
                                          (vf_cmplSD_cr_vc_CP, valCmplSD.mothers, 2, (6,), verb_token, rela_conj_domainer2, False),
                                          (vf_cmplSD_cr_vc_prep, valCmplSD.mothers, 2, (6,), verb_token, rela_prep_domainer2, False),
                                          (vf_cmplSD_cr_vc_verb, valCmplSD.mothers, 2, (5,), verb_token, rela_domainer2, False),
                                          (vf_cmplSD_cr_nc_CP, valCmplSD.mothers, 2, (6,), verb_token, rela_conj_domainer2, False),
                                          (vf_cmplSD_cr_nc_Prec_adv, valCmplSD.mothers, 2, (5,), verb_token, rela_domainer2, False),
                                          (vf_cmplSD_cr_nc_Prec_prep, valCmplSD.mothers, 2, (6,), verb_token, rela_prep_domainer2, False),
                                      )




# 7.4, Verb Frame, Complements, Animacy
                                       
vf_cmpl_conditionsAN = vf_clause_conditions.format(relas='Cmpl', 
                                                   word_reqs=f'sem_domain_code~{animacy_codes}',                                  
                                                   clause_reqs='')

vf_cmpl_an_np = pred_target.format(basis=f'''

{vf_cmpl_conditionsAN}

    phrase function=Cmpl typ=NP|PrNP|AdvP
        -heads> word
        
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_cmpl_an_pp = pred_target.format(basis=f'''

{vf_cmpl_conditionsAN}

    phrase function=Cmpl typ=PP
        -heads> word
        -prep_obj> word
        
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

# Clause Relations
vf_cmplAN_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Cmpl', 
                                                               reqs=f'sem_domain_code~{animacy_codes} sp#verb'),
                                                               pred_funct=all_preds, ptcp_funct=all_ptcp)

if not cached_data:
    valCmplAN = validateFrame(mother_templates=(vf_cmpl_an_np,
                                                vf_cmpl_an_pp, 
                                                vf_cmpl_simpleNull),
                          daughter_templates = (vf_cmplAN_cr_nc_CP,),
                          exp_name='vf_cmpl_an')
    cache['valCmplAN'] = valCmplAN
else:
    valCmplAN = cache['valCmplAN']

    
params['frame']['vf_cmpl_animacy'] = (
                                          (vf_cmpl_an_np, valCmplAN.daughters, 2, (5,), verb_token, funct_animater, False),
                                          (vf_cmpl_an_pp, valCmplAN.daughters, 2, (5,), verb_token, funct_prep_o_animater, False),
                                          (vf_cmplAN_cr_nc_CP, valCmplAN.mothers, 2, (6,), verb_token, rela_conj_animater, False),
                                      )




# 8.1, Verb Frame, Adjuncts+, Presence/Absence
vf_adju_pa = pred_target.format(basis='''

    phrase function=Adju|Time|Loca|PrAd
    
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_adju_pa_clRela = pred_target.format(basis='''

c2:clause rela=Adju|PrAd

c1 <mother- c2
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_adju_pa_null = pred_target.format(basis='''

c2:clause
/without/
<mother- clause rela=Adju|PrAd
/-/

/without/
<mother- phrase rela=PrAd
/-/

/without/
    phrase function=Adju|Time|Loca|PrAd
/-/

c1 = c2
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_adju_simpleNull = pred_target.format(basis='''

c2:clause
/without/
    phrase function=Adju|Time|Loca|PrAd
/-/

c1 = c2
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

if not cached_data:
    vfAdjPa = validateFrame(mother_templates=(vf_adju_pa, 
                                              vf_adju_simpleNull),
                              daughter_templates=(vf_adju_pa_clRela,),
                              exp_name='vf_adju_pa')
    cache['vfAdjPa'] = vfAdjPa
else:
    vfAdjPa = cache['vfAdjPa']



params['frame']['vf_adju_pa'] = (
                                    (vf_adju_pa, vfAdjPa.daughters, 2, (3,), verb_token, functioner, False),
                                    (vf_adju_pa_clRela, vfAdjPa.mothers, 2, (3,), verb_token, relationer, False),
                                    (vf_adju_pa_null, None, 2, (2,), verb_token, nuller, False)
                                )




# 8.2, Verb Frame, Adjuncts+, Lexemes

vf_adju_conditions = vf_clause_conditions.format(relas='Adju|Time|Loca|PrAd', 
                                                 word_reqs='',
                                                 clause_reqs='')

vf_adju_lex_np = pred_target.format(basis=f'''

{vf_adju_conditions}

    phrase function=Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word
        
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_adju_lex_pp = pred_target.format(basis=f'''

{vf_adju_conditions}

    phrase function=Adju|Time|Loca|PrAd typ=PP
        -heads> word
        -prep_obj> word
        
''', pred_funct=all_preds, ptcp_funct=all_ptcp)


# Clause Relations
vf_adju_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Adju|PrAd', reqs=''), 
                                      pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_adju_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Adju|PrAd', reqs=''),
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_adju_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Adju|PrAd', reqs=''),
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_adju_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Adju|PrAd', reqs=''),
                                      pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_adju_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Adju|PrAd', reqs=''),
                                            pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_adju_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Adju|PrAd', reqs=''),
                                             pred_funct=all_preds, ptcp_funct=all_ptcp)
if not cached_data:
    valAdjuLex = validateFrame(mother_templates=(vf_adju_lex_np,
                                                 vf_adju_lex_pp,
                                                 vf_adju_simpleNull),
                               daughter_templates = (vf_adju_cr_vc_CP,
                                                     vf_adju_cr_vc_prep, 
                                                     vf_adju_cr_vc_verb,
                                                     vf_adju_cr_nc_CP,
                                                     vf_adju_cr_nc_Prec_adv,
                                                     vf_adju_cr_nc_Prec_prep),
                               exp_name='vf_adju_lex')
    cache['valAdjuLex'] = valAdjuLex
else:
    valAdjuLex = cache['valAdjuLex']
    
params['frame']['vf_adju_lex'] = (
                                        (vf_adju_lex_np, valAdjuLex.daughters, 2, (5,), verb_token, funct_lexer, False),
                                        (vf_adju_lex_pp, valAdjuLex.daughters, 2, (5,), verb_token, funct_prep_o_lexer, False),
                                        (vf_adju_cr_vc_CP, valAdjuLex.mothers, 2, (6,), verb_token, rela_conj_lexer, False),
                                        (vf_adju_cr_vc_prep, valAdjuLex.mothers, 2, (6,), verb_token, rela_prep_lexer, False),
                                        (vf_adju_cr_vc_verb, valAdjuLex.mothers, 2, (5,), verb_token, rela_lexer, False),
                                        (vf_adju_cr_nc_CP, valAdjuLex.mothers, 2, (6,), verb_token, rela_conj_lexer, False),
                                        (vf_adju_cr_nc_Prec_adv, valAdjuLex.mothers, 2, (5,), verb_token, rela_lexer, False),
                                        (vf_adju_cr_nc_Prec_prep, valAdjuLex.mothers, 2, (6,), verb_token, rela_prep_lexer, False),
                                    )



# 8.3, Verb Frame, Adjuncts+, Domains

vf_adjuSD_conditions = vf_clause_conditions.format(relas='Adju|Time|Loca|PrAd', 
                                                   word_reqs=f'sem_domain_code~{good_sem_codes}',
                                                   clause_reqs='')

vf_adju_sd_np = pred_target.format(basis=f'''

{vf_adjuSD_conditions}

    phrase function=Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word
        
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_adju_sd_pp = pred_target.format(basis=f'''

{vf_adjuSD_conditions}

    phrase function=Adju|Time|Loca|PrAd typ=PP
        -heads> word
        -prep_obj> word
        
''', pred_funct=all_preds, ptcp_funct=all_ptcp)


# Clause Relations
vf_adjuSD_cr_vc_CP = pred_target.format(basis=clR_vc_CP.format(relas='Adju|PrAd', reqs=f'sem_domain_code~{good_sem_codes}'), 
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_adjuSD_cr_vc_prep = pred_target.format(basis=clR_vc_prep.format(relas='Adju|PrAd', reqs=f'sem_domain_code~{good_sem_codes}'),
                                          pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_adjuSD_cr_vc_verb = pred_target.format(basis=clR_vc_verb.format(relas='Adju|PrAd', reqs=f'sem_domain_code~{good_sem_codes}'),
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_adjuSD_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Adju|PrAd', reqs=f'sem_domain_code~{good_sem_codes}'),
                                      pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_adjuSD_cr_nc_Prec_adv = pred_target.format(basis=clR_nc_PreC_adv.format(relas='Adju|PrAd', reqs=f'sem_domain_code~{good_sem_codes}'),
                                            pred_funct=all_preds, ptcp_funct=all_ptcp)
vf_adjuSD_cr_nc_Prec_prep = pred_target.format(basis=clR_nc_PreC_prep.format(relas='Adju|PrAd', reqs=f'sem_domain_code~{good_sem_codes}'),
                                             pred_funct=all_preds, ptcp_funct=all_ptcp)
if not cached_data:
    valAdjuSD = validateFrame(mother_templates=(vf_adju_sd_np,
                                                vf_adju_sd_pp,
                                                vf_adju_simpleNull),
                               daughter_templates = (vf_adjuSD_cr_vc_CP,
                                                     vf_adjuSD_cr_vc_prep, 
                                                     vf_adjuSD_cr_vc_verb,
                                                     vf_adjuSD_cr_nc_CP,
                                                     vf_adjuSD_cr_nc_Prec_adv,
                                                     vf_adjuSD_cr_nc_Prec_prep),
                               exp_name='vf_adju_sd')
    cache['valAdjuSD'] = valAdjuSD
else:
    valAdjuSD = cache['valAdjuSD']
    
params['frame']['vf_adju_domain'] = (
                                          (vf_adju_sd_np, valAdjuSD.daughters, 2, (5,), verb_token, funct_domainer2, False),
                                          (vf_adju_sd_pp, valAdjuSD.daughters, 2, (5,), verb_token, funct_prep_o_domainer2, False),
                                          (vf_adjuSD_cr_vc_CP, valAdjuSD.mothers, 2, (6,), verb_token, rela_conj_domainer2, False),
                                          (vf_adjuSD_cr_vc_prep, valAdjuSD.mothers, 2, (6,), verb_token, rela_prep_domainer2, False),
                                          (vf_adjuSD_cr_vc_verb, valAdjuSD.mothers, 2, (5,), verb_token, rela_domainer2, False),
                                          (vf_adjuSD_cr_nc_CP, valAdjuSD.mothers, 2, (6,), verb_token, rela_conj_domainer2, False),
                                          (vf_adjuSD_cr_nc_Prec_adv, valAdjuSD.mothers, 2, (5,), verb_token, rela_domainer2, False),
                                          (vf_adjuSD_cr_nc_Prec_prep, valAdjuSD.mothers, 2, (6,), verb_token, rela_prep_domainer2, False),
                                      )




# 8.4, Verb Frame, Adjuncts+, Animacy

vf_adjuAN_conditions = vf_clause_conditions.format(relas='Adju|Time|Loca|PrAd', 
                                                   word_reqs=f'sem_domain_code~{animacy_codes} sp#verb',
                                                   clause_reqs='')

vf_adju_an_np = pred_target.format(basis=f'''

{vf_adjuAN_conditions}

    phrase function=Adju|Time|Loca|PrAd typ=NP|PrNP|AdvP
        -heads> word
        
''', pred_funct=all_preds, ptcp_funct=all_ptcp)

vf_adju_an_pp = pred_target.format(basis=f'''

{vf_adjuAN_conditions}

    phrase function=Adju|Time|Loca|PrAd typ=PP
        -heads> word
        -prep_obj> word
        
''', pred_funct=all_preds, ptcp_funct=all_ptcp)


# Clause Relations
vf_adjuAN_cr_nc_CP = pred_target.format(basis=clR_nc_CP.format(relas='Adju|PrAd', reqs=f'sem_domain_code~{animacy_code} sp#verb'),
                                        pred_funct=all_preds, ptcp_funct=all_ptcp)

if not cached_data:
    valAdjuAN = validateFrame(mother_templates=(vf_adju_an_np,
                                                vf_adju_an_pp,
                                                vf_adju_simpleNull),
                               daughter_templates = (vf_adjuAN_cr_nc_CP,),
                               exp_name='vf_adju_an')
    cache['valAdjuAN'] = valAdjuAN
else:
    valAdjuAN = cache['valAdjuAN']
    
params['frame']['vf_adju_animacy'] = (
                                          (vf_adju_an_np, valAdjuAN.daughters, 2, (5,), verb_token, funct_animater, False),
                                          (vf_adju_an_pp, valAdjuAN.daughters, 2, (5,), verb_token, funct_prep_o_animater, False),
                                          (vf_adjuAN_cr_nc_CP, valAdjuAN.mothers, 2, (6,), verb_token, rela_conj_animater, False),
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

vd_con_window = pred_target.format(basis='', pred_funct=all_preds, ptcp_funct=all_ptcp)

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

vd_con_clause = pred_target.format(basis='', pred_funct=all_preds, ptcp_funct=all_ptcp)

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

vd_con_chain = pred_target.format(basis='', pred_funct=all_preds, ptcp_funct=all_ptcp)

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

vd_domain_simple = pred_target.format(basis='', pred_funct=all_preds, ptcp_funct=all_ptcp)

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

vg_tense = pred_target.format(basis='', pred_funct=all_preds, ptcp_funct=all_ptcp)

def tenser(basis, target):
    # makes tense basis tokens
    return F.vt.v(basis)

params['inventory']['vg_tense'] = (
                                      (vg_tense, None, 2, (2,), verb_token, tenser, True),
                                  )

# save the cache

with open('/Users/cody/Documents/VF_cache.dill', 'wb') as outfile:
    dill.dump(cache, outfile)

print('\nAll parameters ready!')