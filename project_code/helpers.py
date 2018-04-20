from IPython.display import HTML, display
from random import shuffle
from itertools import cycle

def get_lex(self, lex_string):
    '''
    Return ETCBC lex node number from a lexeme string.
    Requires a text fabric feature class with otype/lex features loaded.
    '''
    from __main__ import F
    lex = next(lex for lex in F.otype.s('lex') if F.lex.v(lex) == lex_string)
    return lex

def filter_results(results, levels={}):
    
    '''
    Filters out duplicate TF search results.
    Returns a set.
    Often TF search multiplies the results.
    This function filters out any result that differs below a 
    supplied level. For example, if duplicate results are caused
    by multiple matching words and level is set at "word",
    the function will remove any result that only differs at the word level.
    The filtered out levels will be replaced with a 0 integer to preserve
    the indexation.
    '''
    
    from __main__ import F
    
    no_duplicates = set()
    for r in results:
        new_r = tuple(obj if F.otype.v(obj) not in levels else 0 
                      for obj in r)
        no_duplicates.add(tuple(new_r))
    
    return no_duplicates


def show_results(results, cl_index=0, option=0, limit=100, highlight=[], random=False):
    
    '''
    Prints results from a TF search template for manual inspection.
    '''
    
    from __main__ import T, L, F
    
    reg_text = '<span style="font-family: Times New Roman; font-size: 18px; line-height: 1">{}</span>'
    heb_text = '<span style="font-family: Times New Roman; font-size: 24px; line-height: 1">{}</span>'
    high_text = '<span style="color: {}">{}</span>'
    colors = cycle(('blue', 'green'))
    
    if random:
        shuffle(results)
    
    # summarize results first
    print(len(results), 'results\n')
    
    for i, result in enumerate(results):
        
        clause = result[cl_index]
        ref = '{} {}:{}'.format(*T.sectionFromNode(clause))
        text = ''
        
        # format words and words with highlights
        
        text = ''
        highlights = {}
        
        # format highlighted words
        for index in highlight:
            hi = result[index]
            hcolor = next(colors)
            if F.otype.v(hi) == 'word':
                highlights[hi] = high_text.format(hcolor, T.text([hi]))
            else:
                for w in L.d(hi, otype='word'):
                    highlights[w] = high_text.format(hcolor, T.text([w]))
        
        # format the whole clause string
        for w in L.d(clause, otype='word'):
            if w in highlights:
                text += highlights[w]
            else:
                text += T.text([w])
    
        # display pretty Hebrew results
        display(HTML(reg_text.format(str(i+1) + '. ' + ref)))
        display(HTML(heb_text.format(text)))
        if option:
            opt_text = T.text(L.d(result[option], otype='word'))
            display(HTML(heb_text.format(opt_text)))
            
        print('-'*20, '\n')
        
        if i+1 == limit:
            print(f'results cut off at {limit}')
            break
