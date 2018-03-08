from IPython.display import HTML, display

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


def show_results(results, cl_index=0, option=0, limit=100, highlight=[]):
    
    '''
    Prints results from a TF search template for manual inspection.
    '''
    
    from __main__ import T, L, F
    
    reg_text = '<span style="font-family: Times New Roman; font-size: 14px; line-height: 1">{}</span>'
    heb_text = '<span style="font-family: Times New Roman; font-size: 20px; line-height: 1">{}</span>'
    high_text = '<span style="color: blue">{}</span>'
    
    for i, result in enumerate(results):
        
        clause = result[cl_index]
        ref = '{} {}:{}'.format(*T.sectionFromNode(clause))
        text = ''
        
        # format words and words with highlights
        for w in L.d(clause, otype='word'):
            highlights = []
            for h_index in highlight:
                h = result[h_index]
                if F.otype.v(h) == 'word':
                    highlights.append(h)
                else:
                    highlights.extend(L.d(h, otype='word'))
            w_text = T.text([w]) if w not in highlights else high_text.format(T.text([w]))
            text += w_text
    
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
