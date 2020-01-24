import os
import re
import subprocess
import logging
from pathlib import Path
from tempfile import mkstemp
import numpy as np
from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

class OptWorker(Worker):
    LOG_REGEXP = re.compile(
        r'[\d]+ properties for each run\n[\w\s]+\n' + \
        r'[\d]+ runs\n[\w\s.;e+-]+\n' + \
        r'[\d]+ progress properties for each run\n([\w\s]+)\n' + \
        r'[\d]+ runs\n([\w\s.;,e+-inf]+)\n[.]\n')
    MAX_COST = 1e8

    def __init__(self, config_files, *args, keep_log_files=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.problems = [fp.read() for fp in config_files]
        self.keep_log_files = keep_log_files

    def area_under_curve(self, time, cost):
        # the cost over the interval [0, time_to_first_solution] is set to be
        # equal to cost of first solution
        if not time:
            return OptWorker.MAX_COST
        cost.insert(0, cost[0])
        time.insert(0, 0.)
        return np.trapz(cost, time)

    def compute(self, config_id, config, budget, working_directory):
        cost_auc = []
        final_cost = []

        # possibly make these two variables a function of budget
        num_runs = 10
        duration = budget
        for problem in self.problems:
            planner = config['planner'].lower()
            params = '\n'.join([f'{planner}.{param}={value}' \
                for param, value in config.items() if param != 'planner'])
            problem += f'[benchmark]\ntime_limit={duration}\nmem_limit=100000\n' + \
                       f'run_count={num_runs}\n\n[planner]\n{planner}=\n{params}'
            cfg_file_handle, abs_path = mkstemp(suffix='.cfg', text=True)
            os.write(cfg_file_handle, problem.encode())
            os.close(cfg_file_handle)
            try:
                subprocess.run(['ompl_benchmark', abs_path],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as err:
                logging.warning(
                    f'This command terminated with error code {err.returncode}:\n\t{err.cmd}\n'
                    f'and produced the following output:\n\n{err.output}')
                cost_auc.append([OptWorker.MAX_COST])
                final_cost.append([np.nan])
                continue
            # if not self.keep_log_files:
            #     os.remove(abs_path)

            log_path = abs_path[:-3] + 'log'
            print('log_path = ' + log_path)
            with open(log_path, 'r') as logfile:
                match = OptWorker.LOG_REGEXP.search(logfile.read())
                properties = match.group(1).splitlines()
                values = [[x.split(',')[:-1] for x in line.split(';')[:-1]] for line in match.group(2).splitlines()]
                time_index = properties.index('time REAL')
                cost_index = properties.index('best cost REAL')

                time = [[float(ts[time_index]) for ts in run] for run in values]
                cost = [[float(ts[cost_index]) for ts in run] for run in values]
                cost_auc.append([self.area_under_curve(c, t) for c, t in zip(cost, time)])
            if not self.keep_log_files:
                os.remove(log_path)

        return {'loss': np.sum(cost_auc),
                'info': {'cost_auc': cost_auc, 'final_cost': final_cost}}

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        planner = CSH.CategoricalHyperparameter(
            name='planner',
            choices=[
                'AnytimePathShortening',
                'BITstar',
                'CForest',
                'LazyLBTRRT',
                'LazyPRMstar',
                'LBTRRT',
                'PRMstar',
                'RRTstar',
                'RRTXstatic',
                'SPARS',
                'SPARStwo',
                'SST'
            ]
        )
        cs.add_hyperparameter(planner)

        # parameters common among more than one planner
        rnge = CSH.UniformFloatHyperparameter(
            'range', lower=0.001, upper=1000, log=True)
        goal_bias = CSH.UniformFloatHyperparameter(
            'goal_bias', lower=0., upper=1., default_value=.05)
        use_k_nearest = CSH.UniformIntegerHyperparameter(
            'use_k_nearest', lower=0, upper=1, default_value=1)
        epsilon = CSH.UniformFloatHyperparameter(
            'epsilon', lower=0., upper=10., default_value=.4)
        rewire_factor = CSH.UniformFloatHyperparameter(
            'rewire_factor', lower=1., upper=2., default_value=1.1)
        informed_sampling = CSH.UniformIntegerHyperparameter(
            'informed_sampling', lower=0, upper=1, default_value=0)
        sample_rejection = CSH.UniformIntegerHyperparameter(
            'sample_rejection', lower=0, upper=1, default_value=0)
        number_sampling_attempts = CSH.UniformIntegerHyperparameter(
            'number_sampling_attempts', lower=10, upper=100000, default_value=100, log=True)
        stretch_factor = CSH.UniformFloatHyperparameter(
            'stretch_factor', lower=1.1, upper=3., default_value=3.)
        sparse_delta_fraction = CSH.UniformFloatHyperparameter(
            'sparse_delta_fraction', lower=0., upper=1., default_value=.25)
        dense_delta_fraction = CSH.UniformFloatHyperparameter(
            'dense_delta_fraction', lower=0., upper=.1, default_value=.001)
        max_failures = CSH.UniformIntegerHyperparameter(
            'max_failures', lower=100, upper=3000, default_value=1000, log=True)
        focus_search = CSH.UniformIntegerHyperparameter(
            'focus_search', lower=0, upper=1, default_value=1)

        cs.add_hyperparameters([
            rnge, goal_bias, use_k_nearest, epsilon, rewire_factor, informed_sampling,
            sample_rejection, number_sampling_attempts, stretch_factor, sparse_delta_fraction,
            dense_delta_fraction, max_failures, focus_search
        ])
        cs.add_conditions([
            CS.OrConjunction(
                CS.EqualsCondition(rnge, planner, 'LazyLBTRRT'),
                CS.EqualsCondition(rnge, planner, 'LazyPRMstar'),
                CS.EqualsCondition(rnge, planner, 'LBTRRT'),
                CS.EqualsCondition(rnge, planner, 'RRTXstatic'),
                CS.EqualsCondition(rnge, planner, 'SST')
            ),
            CS.OrConjunction(
                CS.EqualsCondition(goal_bias, planner, 'LazyLBTRRT'),
                CS.EqualsCondition(goal_bias, planner, 'LBTRRT'),
                CS.EqualsCondition(goal_bias, planner, 'RRTXstatic'),
                CS.EqualsCondition(goal_bias, planner, 'SST')
            ),
            CS.OrConjunction(
                CS.EqualsCondition(use_k_nearest, planner, 'BITstar'),
                CS.EqualsCondition(use_k_nearest, planner, 'RRTstar'),
                CS.EqualsCondition(use_k_nearest, planner, 'RRTXstatic')
            ),
            CS.OrConjunction(
                CS.EqualsCondition(epsilon, planner, 'LazyLBTRRT'),
                CS.EqualsCondition(epsilon, planner, 'LBTRRT'),
                CS.EqualsCondition(epsilon, planner, 'RRTXstatic')
            ),
            CS.OrConjunction(
                CS.EqualsCondition(rewire_factor, planner, 'BITstar'),
                CS.EqualsCondition(rewire_factor, planner, 'RRTstar'),
                CS.EqualsCondition(rewire_factor, planner, 'RRTXstatic')
            ),
            CS.OrConjunction(
                CS.EqualsCondition(informed_sampling, planner, 'RRTstar'),
                CS.EqualsCondition(informed_sampling, planner, 'RRTXstatic')
            ),
            CS.OrConjunction(
                CS.EqualsCondition(sample_rejection, planner, 'RRTstar'),
                CS.EqualsCondition(sample_rejection, planner, 'RRTXstatic')
            ),
            CS.OrConjunction(
                CS.EqualsCondition(number_sampling_attempts,
                                   planner, 'RRTstar'),
                CS.EqualsCondition(number_sampling_attempts,
                                   planner, 'RRTXstatic')
            ),
            CS.OrConjunction(
                CS.EqualsCondition(stretch_factor, planner, 'SPARS'),
                CS.EqualsCondition(stretch_factor, planner, 'SPARStwo')
            ),
            CS.OrConjunction(
                CS.EqualsCondition(sparse_delta_fraction, planner, 'SPARS'),
                CS.EqualsCondition(sparse_delta_fraction, planner, 'SPARStwo')
            ),
            CS.OrConjunction(
                CS.EqualsCondition(dense_delta_fraction, planner, 'SPARS'),
                CS.EqualsCondition(dense_delta_fraction, planner, 'SPARStwo')
            ),
            CS.OrConjunction(
                CS.EqualsCondition(max_failures, planner, 'SPARS'),
                CS.EqualsCondition(max_failures, planner, 'SPARStwo')
            ),
            CS.OrConjunction(
                CS.EqualsCondition(focus_search, planner, 'RRTstar'),
                CS.EqualsCondition(focus_search, planner, 'CForest')
            )
        ])
        cs.add_forbidden_clause(
            CS.ForbiddenAndConjunction(
                CS.ForbiddenEqualsClause(informed_sampling, 1),
                CS.ForbiddenEqualsClause(sample_rejection, 1)
            )
        )

        # AnytimePathShortening
        max_hybrid_paths = CSH.UniformIntegerHyperparameter(
            'max_hybrid_paths', lower=2, upper=128, default_value=24, log=True)
        num_planners = CSH.UniformIntegerHyperparameter(
            'num_planners', lower=2, upper=8, default_value=4)
        shortcut = CSH.UniformIntegerHyperparameter(
            'shortcut', lower=0, upper=1, default_value=1)
        hybridize = CSH.UniformIntegerHyperparameter(
            'hybridize', lower=0, upper=1, default_value=1)
        cs.add_hyperparameters([
            max_hybrid_paths, num_planners, shortcut, hybridize])
        cs.add_conditions([
            CS.EqualsCondition(max_hybrid_paths, planner,
                               'AnytimePathShortening'),
            CS.EqualsCondition(num_planners, planner, 'AnytimePathShortening'),
            CS.EqualsCondition(shortcut, planner, 'AnytimePathShortening'),
            CS.EqualsCondition(hybridize, planner, 'AnytimePathShortening')
        ])

        # BIT*
        samples_per_batch = CSH.UniformIntegerHyperparameter(
            'samples_per_batch', lower=1, upper=1000000, default_value=100)
        use_graphPtr_pruning = CSH.UniformIntegerHyperparameter(
            'use_graphPtr_pruning', lower=0, upper=1, default_value=1)
        prune_threshold_as_fractional_cost_change = CSH.UniformFloatHyperparameter(
            'prune_threshold_as_fractional_cost_change', lower=0., upper=1., default_value=.05)
        delay_rewiring_to_first_solution = CSH.UniformIntegerHyperparameter(
            'delay_rewiring_to_first_solution', lower=0, upper=1, default_value=1)
        use_just_in_time_sampling = CSH.UniformIntegerHyperparameter(
            'use_just_in_time_sampling', lower=0, upper=1, default_value=0)
        drop_unconnected_samples_on_prune = CSH.UniformIntegerHyperparameter(
            'drop_unconnected_samples_on_prune', lower=0, upper=1, default_value=0)
        use_strict_queue_ordering = CSH.UniformIntegerHyperparameter(
            'use_strict_queue_ordering', lower=0, upper=1, default_value=0)
        find_approximate_solutions = CSH.UniformIntegerHyperparameter(
            'find_approximate_solutions', lower=0, upper=1, default_value=0)
        cs.add_hyperparameters([
            samples_per_batch, use_graphPtr_pruning, prune_threshold_as_fractional_cost_change,
            delay_rewiring_to_first_solution, use_just_in_time_sampling,
            drop_unconnected_samples_on_prune, use_strict_queue_ordering,
            find_approximate_solutions
        ])
        cs.add_conditions([
            CS.EqualsCondition(samples_per_batch, planner, 'BITstar'),
            CS.EqualsCondition(use_graphPtr_pruning, planner, 'BITstar'),
            CS.EqualsCondition(
                prune_threshold_as_fractional_cost_change, planner, 'BITstar'),
            CS.EqualsCondition(
                delay_rewiring_to_first_solution, planner, 'BITstar'),
            CS.EqualsCondition(use_just_in_time_sampling, planner, 'BITstar'),
            CS.EqualsCondition(
                drop_unconnected_samples_on_prune, planner, 'BITstar'),
            CS.EqualsCondition(use_strict_queue_ordering, planner, 'BITstar'),
            CS.EqualsCondition(find_approximate_solutions, planner, 'BITstar')
        ])

        # CForest
        num_threads = CSH.UniformIntegerHyperparameter(
            'num_threads', lower=2, upper=8, default_value=4)
        cs.add_hyperparameter(num_threads)
        cs.add_condition(CS.EqualsCondition(num_threads, planner, 'CForest'))

        # RRT*
        delay_collision_checking = CSH.UniformIntegerHyperparameter(
            'delay_collision_checking', lower=0, upper=1, default_value=1)
        # tree_pruning = CSH.UniformIntegerHyperparameter(
        #     'tree_pruning', lower=0, upper=1, default_value=0)
        # prune_threshold = CSH.UniformFloatHyperparameter(
        #     'prune_threshold', lower=0., upper=1., default_value=.05)
        # pruned_measure = CSH.UniformIntegerHyperparameter(
        #     'pruned_measure', lower=0, upper=1, default_value=0)
        # new_state_rejection = CSH.UniformIntegerHyperparameter(
        #     'new_state_rejection', lower=0, upper=1, default_value=0)
        # use_admissible_heuristic = CSH.UniformIntegerHyperparameter(
        #     'use_admissible_heuristic', lower=0, upper=1, default_value=1)
        # ordered_sampling = CSH.UniformIntegerHyperparameter(
        #     'ordered_sampling', lower=0, upper=1, default_value=0)
        # ordering_batch_size = CSH.UniformIntegerHyperparameter(
        #     'ordering_batch_size', lower=1, upper=1000000, default_value=1, log=True)
        # cs.add_hyperparameters(
        #     [delay_collision_checking, tree_pruning, prune_threshold, pruned_measure,
        #      new_state_rejection, use_admissible_heuristic, ordered_sampling,
        #      ordering_batch_size])
        cs.add_hyperparameter(delay_collision_checking)
        cs.add_condition(CS.EqualsCondition(delay_collision_checking, planner, 'RRTstar'))
            # CS.EqualsCondition(tree_pruning, planner, 'RRTstar'),
            # CS.EqualsCondition(prune_threshold, planner, 'RRTstar'),
            # CS.EqualsCondition(pruned_measure, planner, 'RRTstar'),
            # CS.EqualsCondition(new_state_rejection, planner, 'RRTstar'),
            # CS.EqualsCondition(use_admissible_heuristic, planner, 'RRTstar')

        #     CS.AndConjunction(
        #         CS.EqualsCondition(ordered_sampling, planner, 'RRTstar'),
        #         CS.OrConjunction(
        #             CS.EqualsCondition(ordered_sampling, informed_sampling, 1),
        #             CS.EqualsCondition(ordered_sampling, sample_rejection, 1)
        #         )
        #     ),
        #     CS.AndConjunction(
        #         CS.EqualsCondition(ordering_batch_size, planner, 'RRTstar'),
        #         CS.EqualsCondition(ordering_batch_size, ordered_sampling, 1)
        #     )
        # ])
        # cs.add_forbidden_clause(
        #     CS.ForbiddenAndConjunction(
        #         CS.ForbiddenEqualsClause(informed_sampling, 1),
        #         CS.ForbiddenEqualsClause(ordered_sampling, 1)
        #     )
        # )

        # RRTXstatic
        update_children = CSH.UniformIntegerHyperparameter(
            'update_children', lower=0, upper=1, default_value=1)
        rejection_variant = CSH.UniformIntegerHyperparameter(
            'rejection_variant', lower=0, upper=3, default_value=0)
        rejection_variant_alpha = CSH.UniformFloatHyperparameter(
            'rejection_variant_alpha', lower=0., upper=1., default_value=1.)
        cs.add_hyperparameters(
            [update_children, rejection_variant, rejection_variant_alpha])
        cs.add_conditions([
            CS.EqualsCondition(update_children, planner, 'RRTXstatic'),
            CS.EqualsCondition(rejection_variant, planner, 'RRTXstatic'),
            CS.EqualsCondition(rejection_variant_alpha, planner, 'RRTXstatic')
        ])

        # SST
        selection_radius = CSH.UniformFloatHyperparameter(
            'selection_radius', lower=.1, upper=100., default_value=5., log=True)
        pruning_radius = CSH.UniformFloatHyperparameter(
            'pruning_radius', lower=.1, upper=100., default_value=3., log=True)
        cs.add_hyperparameters([selection_radius, pruning_radius])
        cs.add_conditions([
            CS.EqualsCondition(selection_radius, planner, 'SST'),
            CS.EqualsCondition(pruning_radius, planner, 'SST')
        ])

        return cs
