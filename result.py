#!/usr/bin/env python3
import pandas as pnd

def result_table(df_path):
    """
    Build and return a dataframe with evaluation measurements.

    Image level: `img-correct` and `cell-correct`

    Puzzle level: `grid-correct` and `grid-failure`


    `img-correct`: accuracy of the classifier on labelling handwritten digits

    `cell-correct`: percentage of cells containing the desired value after solving

    `grid-correct`: percentage of puzzles that 1) are solved 2) have desired values in each cell

    `grid-failure`: percentage of puzzles that 1) are solved 2) have at least one wrong value


    for each of those measures, absolute value is provided at `[measurename]_abs`
    

    Relevant measures included as well:

    `avg-time`: average computating time for puzzle-solving

    `givens-correct`: percentage of given cells from solved puzzle containing the desired value

    """
    table = dict()
    df = pnd.read_csv(df_path, index_col=0)
    df_solved = df.loc[df['count_nogoods'] > -1]
    
    ## for failed puzzle: number of right guesses
    count_right_guesses = df.loc[df['count_nogoods'] == -1, 'nr_right_givens_guess'].sum()
    #table['givens-correct'] = count_giv_corr / df['nr_givens'].sum()
    #table['givens-correct_abs'] = count_giv_corr
    

    ## for succeeded puzzle: nr right given sol    
    count_giv_corr = (df_solved['nr_givens'] - df_solved['nr_wrong_givens_sol']).sum()

    table['img-correct_abs'] = count_right_guesses + count_giv_corr
    table['img-correct'] = table['img-correct_abs'] / df['nr_givens'].sum()

    # cell-correct
    cell_wrong_count = df['nr_wrong_cells'].sum() 
    cell_wrong = cell_wrong_count / (len(df) * 81)
    table['cell-correct'] = 1 - cell_wrong
    table['cell-correct_abs'] = (len(df) * 81) - cell_wrong_count

    # grid-correct
    count_grid_correct = len(df_solved[df_solved['nr_wrong_cells'] ==0 ])
    table['grid-correct'] = count_grid_correct / len(df)
    table['grid-correct_abs'] = count_grid_correct

    # grid-failure
    count_grid_failure = sum(df['count_nogoods'] ==-1) #len(df_solved[df_solved['nr_wrong_cells'] > 0])
    table['grid-failure'] = count_grid_failure / len(df) 
    table['grid-failure_abs'] = count_grid_failure
    # time-out
    count_grid_failure = sum(df['count_nogoods'] ==-2) #len(df_solved[df_solved['nr_wrong_cells'] > 0])
    table['time-out'] = count_grid_failure / len(df) 
    table['time-out_abs'] = count_grid_failure

    # others
    #count_giv_corr = (df['nr_givens'] - df['nr_wrong_givens_sol']).sum()
    #table['givens-correct'] = count_giv_corr / df['nr_givens'].sum()
    #table['givens-correct_abs'] = count_giv_corr
    table['avg-time'] = df_solved['comput_time'].mean()
    table['avg-time_abs'] = df_solved['comput_time'].sum()

    rank_col = [name for name in df.columns if 'rank' in name]
    for r in rank_col:
        table['{}_abs'.format(r)] = df_solved[r].sum()
        table['{}'.format(r)] = table['{}_abs'.format(r)] / df_solved['nr_givens'].sum()

    return pnd.DataFrame(table, index=[df_path.split('.')[0],])
