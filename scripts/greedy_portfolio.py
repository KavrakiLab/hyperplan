#!/usr/bin/env python

import sys
import numpy as np
import hpbandster.core.result as hpres

def quantile_with_threshold(x, threshold, q=.7):
    if len(x) > 0:
        result = np.quantile(x, q)
        if result <= threshold:
            return result
    return np.inf

def greedy_portfolio(resultdir):
    result = hpres.logged_results_to_HBS_result(resultdir)
    all_runs = result.get_all_runs()
    id2conf = result.get_id2config_mapping()
    best_id = result.get_incumbent_id()
    best_run = result.get_runs_by_id(best_id)[-1]
    print(best_run.config_id)
    #best_individual_losses = np.array([quantile_with_threshold(times, best_run.budget) for times in best_run.info['time']])
    #best_loss = best_run.loss
    #print(best_loss-sum(best_individual_losses))
    best_individual_losses = np.array([np.inf] * 10)
    best_loss = np.inf
    individual_losses = [np.array([quantile_with_threshold(times, run.budget)
                        for times in run.info['time']]) for run in all_runs]
    print(best_loss, best_individual_losses, id2conf[best_run.config_id])
    for _ in range(1):
        old_loss = best_loss
        for run, losses in zip(all_runs, individual_losses):
            candidate_losses = np.minimum(losses, best_individual_losses)
            #print(candidate_losses - losses)
            candidate_loss = sum(candidate_losses)
            if candidate_loss < best_loss:
                best_individual_losses = candidate_losses
                best_loss = candidate_loss
                best_run = run
        if old_loss > best_loss:
            print(best_run.config_id, best_loss, best_individual_losses, id2conf[best_run.config_id])
        

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("specify one or more output directories as command line arguments")
        exit(1)
    for path in sys.argv[1:]:
        greedy_portfolio(path)
