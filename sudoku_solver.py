#!/usr/bin/env python3
#from cppy.cppy import *
import numpy as np
from scipy.stats import poisson
from ortools.sat.python import cp_model
from ortools.sat import sat_parameters_pb2
from collections import Counter

# Helpers

def cnn_output_simulation(puzzle):
    """
    Poisson distribution probability tensor containing a probability 
    score for each possible value in each cell.
    """
    probs = np.zeros(shape=(puzzle.shape + (len(puzzle)+ 1,) ))
    values = np.arange(len(puzzle)+ 1) 
    for r in range(len(puzzle)):
        for c in range(len(puzzle[0])):
            pss = poisson.pmf(values,puzzle[r,c]+1)
            pss[puzzle[r,c]] += 0.01 # break equal probs
            probs[r,c] = pss/np.sum(pss)
    return probs
    
    
def make_element(array, ind, name, cpmodel):
    aux = cpmodel.NewIntVar(int(np.min(array)),int(np.max(array)), name)
    cpmodel.AddElement(ind, array, aux)
    return aux

def make_boolean(board,i,j,c, cpmodel):
    # Declare our intermediate boolean variable.
    b = cpmodel.NewBoolVar('b')

    # Implement b == (board[i][j] == c).
    cpmodel.Add(board[(i,j)] == c).OnlyEnforceIf(b)
    cpmodel.Add(board[(i,j)] != c).OnlyEnforceIf(b.Not())
    return b


def sudoku_model(model):
    # from or-tools sudoku-sat example
    n = 9
    b = 3
    grid = {}
    for i in range(n):
        for j in range(n): 
            grid[(i,j)] = model.NewIntVar(1, n, 'grid {} {}'.format(i,j))
    
    # alldiff constraints
    for i in range(n):
        model.AddAllDifferent([grid[(i,j)] for j in range(n)])
        
    for j in range(n):
        model.AddAllDifferent([grid[(i,j)] for i in range(n)])

    for i in range(b):
        for j in range(b):
            cell = [grid[(
                i * b + di,
                j * b + dj
            )] for di in range(b) for dj in range(b)]
            model.AddAllDifferent(cell)
    return grid, model
    
def count_results(givens, solution, output, logprobs):
    yield givens[givens>0].size # count of given cells
    yield len(np.where(output != solution)[0])
    givens_idx = np.where(givens != 0)
    yield len(np.where(output[givens_idx] != givens[givens_idx])[0])
    
    # digits rank:
    ## find out, for each cell value, how it was initially ranked by the classifier
    clf_rank = np.argsort(np.argsort(-logprobs, axis=2))
    sol_rank = []
    for i in range(len(output)):
        for j in range(len(output)):
            if givens[i,j] != 0:
                sol_rank += [clf_rank[i, j, output[i,j]].item()]
    c = Counter(sol_rank)
    yield c
    

class NoMoreThanOneCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
     
    def OnSolutionCallback(self):
        self.StopSearch()
    
    
class SudokuSolver:
    def __init__(self, use_ortools=True):
        self.model = None
        #self.solver_cppy = MiniZincPython()
        self.solver_OR = cp_model.CpSolver()
        #self.solver_OR.search_branching = sat_parameters_pb2.SatParameters.PORTFOLIO_WITH_QUICK_RESTART_SEARCH
        self.use_ortools = use_ortools
        self.decision_vars = None
        
    def solve_sudoku(self, logprobs, is_given, baseline, top):
        return self.__solve_sudoku_ortools(logprobs, is_given, baseline, top)
        
    def __solve_sudoku_ortools(self, logprobs, is_given, baseline=False, top=10):
        
        n = len(is_given)
        grid, self.model = sudoku_model(cp_model.CpModel())
        self.decision_vars = grid
        
        # guess based on most likely digit
        ml_guess = np.argmax(logprobs, axis=2)
        if baseline:
            for i in range(n):
                for j in range(n):
                    if is_given[i,j]:
                        self.model.Add(grid[(i,j)] == ml_guess[i,j])
        else :
            logprobs = (-logprobs*1000).astype(int)
            obj = [make_element(logprobs[i,j].tolist(), grid[(i,j)], '-lprob {} {}'.format(i,j), self.model) for i in range(len(logprobs)) for j in range(len(logprobs[0])) if is_given[i,j]]
            self.model.Minimize(cp_model.LinearExpr.Sum(obj))
            # greedy search heuristic by branching on high likelihood digit value first
            for i  in range(n):
                for j in range(n):
                    if is_given[i,j]:
                        self.model.AddHint(grid[(i,j)], ml_guess[i,j])


        # prune domain to only keep top-k digit values for each cell
        if top < 10:
            for i in range(n):
                for j in range(n):
                    # only for givens
                    if is_given[i,j]:
                        # the indices ARE the values
                        order = np.argsort(logprobs[i,j,:])
                        # can not be any value after 'top' first ones
                        for v in order[top:]:
                            self.model.Add(grid[i,j] != v)
        
        
        solution = np.zeros((n,n))
        status = self.solver_OR.Solve(self.model)
        if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
            for i in range(n):
                for j in range(n):
                    solution[i,j] = int(self.solver_OR.Value(grid[(i,j)]))
        
        return solution.astype(int)
        
    def has_unique_solution(self, output, is_given):
        """
        Solve sudoku with values of given cells from the output
        and a nogood.
        """
        n = len(is_given)
        b = np.sqrt(n).astype(int)
        model = cp_model.CpModel()
        
        grid = {}
        for i in range(n):
            for j in range(n): 
                grid[(i,j)] = model.NewIntVar(1, n, 'grid {} {}'.format(i,j))
        
        for i in range(n):
            model.AddAllDifferent([grid[(i,j)] for j in range(n)])
            
        for j in range(n):
            model.AddAllDifferent([grid[(i,j)] for i in range(n)])

        for i in range(b):
            for j in range(b):
                cell = [grid[(
                    i * b + di,
                    j * b + dj
                )] for di in range(b) for dj in range(b)]
                model.AddAllDifferent(cell)
        
        for i in range(n):
            for j in range(n):
                if is_given[i,j]:
                    model.Add(grid[(i,j)] == output[i,j])

        # nogood for given and non-given cells value from the output
        variables = []
        fassign = []
        for i in range(n):
            for j in range(n): 
                variables.append(grid[(i,j)])
                fassign.append(output[i,j])
        model.AddForbiddenAssignments(variables, [tuple(fassign),])

        # callback to stop the search early on if necessary
        stopsearch = NoMoreThanOneCallback()
        status = self.solver_OR.SearchForAllSolutions(model, stopsearch)
        return status == cp_model.INFEASIBLE
                        
        
    def solve_nogood(self, output, is_given):
        assert self.model is not None, "run the solver on a sudoku instance first"
        
        grid = self.decision_vars
        n = np.sqrt(len(grid)).astype(int)

        # nogood: built-in forbidden assignement constraint 
        variables = []
        fassign = []
        for i in range(n):
            for j in range(n):
                if is_given[i,j]:
                    variables.append(grid[(i,j)])
                    fassign.append(output[i,j])

        self.model.AddForbiddenAssignments(variables, [tuple(fassign),])
        
        #TODO refactor
        solution = np.zeros((n,n))
        status = self.solver_OR.Solve(self.model)
        if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
            for i in range(n):
                for j in range(n):
                    solution[i,j] = int(self.solver_OR.Value(grid[(i,j)]))
                
        return solution.astype(int)
            
    