#!/usr/bin/env python3
import argparse 
import torch
import os
import time
import numpy as np
import pandas as pnd
from mnist import build_conv_digit_net
from sudoku_solver import SudokuSolver, count_results
from tqdm.auto import tqdm
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='path to pretrained mnist classifier')
    parser.add_argument('--datadir', type=str, default='./data', help='path to data folder')
    parser.add_argument('--baseline', action='store_true', help='set this flag to use the baseline viz-sudoku solver')
    parser.add_argument('--log', type=int, default=0, help='log level')
    parser.add_argument('--oname', type=str, default='output', help='name of the compressed csv output file containing the results')
    parser.add_argument('--range', type=str, help='Two numbers in [0...N] separated by a hyphen, standing for the range of puzzle instances to evaluate', default='0-3000') 
    parser.add_argument('--max_nogoods', type=int, help='maximum iterations for adding nogoods and solving a sudoku', default=0)
    parser.add_argument('--instances', type=int, nargs='+', help="list of specific instances indices. If indices are provided, the 'range' arg is ignored.")
    parser.add_argument('--top', type=int, default=10, metavar='K', help='Limit search space by only considering top-k labels' )
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    print('loading data')
    # from https://github.com/locuslab/SATNet/blob/master/exps/sudoku.py
    with open(os.path.join(args.datadir,'features.pt'), 'rb') as f:
        X_in = torch.load(f)
    with open(os.path.join(args.datadir,'features_img.pt'), 'rb') as f:
        Ximg_in = torch.load(f)
    with open(os.path.join(args.datadir,'labels.pt'), 'rb') as f:
        Y_in = torch.load(f)
    
    n = X_in.size(1)
    
    if args.instances:
        indices = np.array(args.instances)
        print(indices)
        X_in = X_in[indices]
        Ximg_in = Ximg_in[indices]
        Y_in = Y_in[indices]
    else:
        first, last = (int(d) for d in args.range.split('-'))
        assert last <= X_in.size(0), 'last index out of range with input data of size {}'.format(X_in.size(0))
        X_in = X_in[first:last]
        Ximg_in = Ximg_in[first:last]
        Y_in = Y_in[first:last]
    
    # MNIST classifier
    convnet = build_conv_digit_net(args.model, 'calib' in args.model).to(device)
    convnet.eval()
    
    # flatten img input
    Ximg = Ximg_in.flatten(start_dim=1, end_dim=2).unsqueeze(2).float().to(device)
    
    with torch.no_grad(): 
        
        stats = {
            'nr_wrong_cells':[],
            'nr_wrong_givens_sol':[],
            'nr_right_givens_guess':[],
            'nr_givens':[],
            'count_nogoods':[],
            'comput_time':[]}
        [stats.update({'rank-{}'.format(i):[] }) for i in range(0, 9)]
        
        solver = SudokuSolver(not args.use_cppy)
            
            
        for puz_idx in tqdm(range(X_in.size(0)), desc='puzzle instances'):
            
            # get logprob from CNN
            logprobs = convnet(Ximg[puz_idx]) # 81 x 10
            
            # build the mask to identify empty cells
            padding_x = np.zeros((n,n,n+1))
            padding_y = np.zeros((n,n,n+1))
            padding_x[:,:,1:] = X_in[puz_idx]
            padding_y[:,:,1:] = Y_in[puz_idx]
            digits = np.argmax(padding_x, axis=2)
            solution = np.argmax(padding_y, axis=2)
            is_given = np.ma.masked_not_equal(digits, 0).mask

            logprobs = logprobs.view(n, n, -1).numpy()
            
            # solve puzzle instance
            start = time.time()
            output = solver.solve_sudoku(logprobs,
             is_given,
             args.baseline, args.top) 
            
            
            
            # digit classifier accuracy on sudoku digits
            digit_guess = np.argmax(logprobs, axis=2)
            cnt_givens, miss, miss_givens, ranks = count_results(digits, solution, output, logprobs)
            correct_guess_count = len(np.where(digit_guess[digits>0] == digits[digits>0])[0])
            

            # Add dominance constraints
            i = 0
            while not args.baseline and not solver.has_unique_solution(output, is_given) and i < args.max_nogoods:
                print('sudoku[{}] : {}, multiple solutions'.format(puz_idx, miss))
                print("let's try again!")
                output = solver.solve_nogood(output, is_given)
                if args.log >= 1:
                    print('obtained \n', output)
                    print('expected \n',solution)
                cnt_givens, miss, miss_givens, ranks = count_results(digits, solution, output, logprobs)
                i += 1
                
            stats['comput_time'].append(time.time() - start)
            
            # Stats
            
            # set count_nogoods to -1 if unfeasible game
            if (output == 0.0).all() or (output == None).any():
                stats['count_nogoods'].append(-1)
                if args.log >= 2:
                    print("unfeasible")
            else:
                stats['count_nogoods'].append(i)
            stats['nr_right_givens_guess'].append( correct_guess_count )
            stats['nr_givens'].append(cnt_givens)
            stats['nr_wrong_cells'].append(miss)
            stats['nr_wrong_givens_sol'].append(miss_givens)

            for i in range(0,9):
                stats['rank-{}'.format(i)].append(ranks[i])

            if args.log >= 3:
                print('obtained \n', output)
                print('expected \n',solution)
            if args.log >= 2:
                print(miss)
            elif args.log >= 1:
                if miss > 0:
                    print('sudoku[{}] : {}'.format(puz_idx, miss))
        
        df = pnd.DataFrame(data=stats)
        print(df.describe())        
        df.to_csv('{}.zip'.format(args.oname))
        
if __name__ in '__main__':
    main()
    

