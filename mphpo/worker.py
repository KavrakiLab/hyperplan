######################################################################
# Software License Agreement (BSD License)
#
#  Copyright (c) 2020, Rice University
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   * Neither the name of Rice University nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
######################################################################

# Author: Mark Moll

import os
import re
import subprocess
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from tempfile import mkstemp
import numpy as np
from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

def quantile_with_fallback(x, fallback, q=.7):
    return np.quantile(x, q) if len(x) > 0 else fallback

class BaseWorker(Worker, ABC):
    LOG_PROPERTIES_REGEXP = re.compile(
        r'[\d]+ properties for each run\n([\w\s]+)\n[\d]+ runs\n([\w\s.;+-]+)\n' + \
        r'([.]|[\d]+ progress properties for each run)\n')
    LOG_PROGRESS_PROPERTIES_REGEXP = re.compile(
        r'[\d]+ progress properties for each run\n([\w\s]+)\n[\d]+ runs\n([\w\s.;,+-]+)\n[.]\n')
    MAX_COST = 1e8

    def __init__(self, config_files, selected_properties, *args,
                 selected_progress_properties={}, **kwargs):
        Worker.__init__(self, *args, **kwargs)
        self.problems = [fp.read() for fp in config_files]
        self.keep_log_files = True
        self.selected_properties = selected_properties
        self.selected_progress_properties = selected_progress_properties

    @abstractmethod
    def loss(self, budget, results):
        pass

    @abstractmethod
    def progress_loss(self, budget, progress_data):
        pass

    @abstractmethod
    def duration_runs(self, budget):
        pass

    def compute(self, config_id, config, budget, working_directory):
        duration, num_runs = self.duration_runs(budget)
        results = defaultdict(list)

        for problem in self.problems:
            try:
                config['problem.max_control_duration'] += config['problem.min_control_duration']
            except KeyError:
                pass

            for param, value in config.items():
                if param.startswith('problem.'):
                    problem += '\n%s=%s' % (param[8:], value)
            planner = config['planner'].lower()
            problem += f'\n\n[benchmark]\ntime_limit={duration}\nmem_limit=100000\n' \
                       f'run_count={num_runs}\n\n[planner]\n{planner}=\n'
            for param, value in config.items():
                if not param == 'planner' and not param.startswith('problem.'):
                    problem += f'{planner}.{param}={value}\n'
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
                for key in self.selected_properties.keys():
                    results[key].append([budget if key == 'time' else np.nan])
                for key in self.selected_progress_properties.keys():
                    results[key].append([self.MAX_COST if key == 'cost' else np.nan])
                continue
            if not self.keep_log_files:
                os.remove(abs_path)

            log_path = abs_path[:-3] + 'log'
            with open(log_path, 'r') as logfile:
                log = logfile.read()
                match = self.LOG_PROPERTIES_REGEXP.search(log)
                # process run properties
                properties = match.group(1).splitlines()
                values = [[float(x) if x else np.nan for x in line.split('; ')]
                          for line in match.group(2).splitlines()]
                for key, val in self.selected_properties.items():
                    if val in properties:
                        results[key].append([run[properties.index(val)] for run in values])
                    else:
                        logging.warning(f'property "{val}" was not found in log')
                        results[key].append(len(values) * [np.nan])

                # process progress properties
                if self.selected_progress_properties:
                    match = self.LOG_PROGRESS_PROPERTIES_REGEXP.search(log)
                    properties = match.group(1).splitlines()
                    values = [[[float(x) for x in tple.split(',')[:-1]]
                               for tple in line.split(';')[:-1]]
                              for line in match.group(2).splitlines()]
                    progress_data = defaultdict(list)
                    for key, val in self.selected_progress_properties.items():
                        if val in properties:
                            progress_data[key] = [[tple[properties.index(val)] for tple in run]
                                                  for run in values]
                        else:
                            logging.warning(f'progress property "{val}" was not found in log')
                            progress_data[key] = len(values) * [[np.nan]]
                    results['_progress_loss'] = self.progress_loss(budget, progress_data)
            if not self.keep_log_files:
                os.remove(log_path)

        return {'loss': self.loss(budget, results), 'info': results}

class SpeedWorker(BaseWorker):
    def __init__(self, config_files, *args, **kwargs):
        super().__init__(config_files,
                         {'time': 'time REAL', 'path_length': 'simplified solution length REAL'},
                         *args, **kwargs)

    def loss(self, budget, results):
        # skip last run since it is always a timeout with no solution found
        return np.sum([quantile_with_fallback(length[:-1], budget)
                       for length in results['time']])

    def progress_loss(self, budget, progress_data):
        raise Exception('Not implemented for this class')

    def duration_runs(self, budget):
        return budget, 0

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        planner = CSH.CategoricalHyperparameter(
            name='planner',
            choices=[
                'BiEST',
                'BKPIECE',
                'EST',
                'KPIECE',
                'LazyPRM',
                'LBKPIECE',
                'PDST',
                'PRM',
                'RRT',
                'RRTConnect',
                'SBL'])
        max_nearest_neighbors = CSH.UniformIntegerHyperparameter(
            'max_nearest_neighbors', lower=1, upper=20, default_value=8)
        rnge = CSH.UniformFloatHyperparameter(
            'range', lower=0.001, upper=10000, log=True)
        intermediate_states = CSH.UniformIntegerHyperparameter(
            'intermediate_states', lower=0, upper=1, default_value=0)
        goal_bias = CSH.UniformFloatHyperparameter(
            'goal_bias', lower=0., upper=1., default_value=.05)
        cs.add_hyperparameters([planner, max_nearest_neighbors, rnge, intermediate_states,
                                goal_bias])

        cs.add_conditions([
            CS.OrConjunction(CS.EqualsCondition(max_nearest_neighbors, planner, 'PRM'),
                             CS.EqualsCondition(max_nearest_neighbors, planner, 'LazyPRM')),
            CS.OrConjunction(CS.EqualsCondition(rnge, planner, 'LazyPRM'),
                             CS.EqualsCondition(rnge, planner, 'RRT'),
                             CS.EqualsCondition(rnge, planner, 'RRTConnect'),
                             CS.EqualsCondition(rnge, planner, 'EST'),
                             CS.EqualsCondition(rnge, planner, 'BiEST'),
                             CS.EqualsCondition(rnge, planner, 'SBL'),
                             CS.EqualsCondition(rnge, planner, 'KPIECE'),
                             CS.EqualsCondition(rnge, planner, 'BKPIECE'),
                             CS.EqualsCondition(rnge, planner, 'LBKPIECE'),
                             CS.EqualsCondition(rnge, planner, 'PDST')),
            CS.OrConjunction(CS.EqualsCondition(intermediate_states, planner, 'RRT'),
                             CS.EqualsCondition(intermediate_states, planner, 'RRTConnect')),
            CS.OrConjunction(CS.EqualsCondition(goal_bias, planner, 'RRT'),
                             CS.EqualsCondition(goal_bias, planner, 'EST'),
                             CS.EqualsCondition(goal_bias, planner, 'KPIECE'))
        ])
        return cs

class SpeedKinodynamicWorker(BaseWorker):
    def __init__(self, config_files, *args, **kwargs):
        super().__init__(config_files,
                         {'path_length': 'solution length REAL',
                          'goal_distance': 'solution difference REAL'},
                         *args, **kwargs)

    def loss(self, budget, results):
        # path length (=duration in seconds) + square of goal distance
        l = [np.array(pl) + np.array(gd)**2
             for pl, gd in zip(results['path_length'], results['goal_distance'])]
        return np.sum([quantile_with_fallback(ql, budget) for ql in l])

    def progress_loss(self, budget, progress_data):
        raise Exception('Not implemented for this class')

    def duration_runs(self, budget):
        return budget, 10

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        planner = CSH.CategoricalHyperparameter(
            name='planner',
            choices=[
                'EST',
                'KPIECE',
                'PDST',
                'RRT',
                'SST',
                'SyclopEST',
                'SyclopRRT'])
        controller = CSH.CategoricalHyperparameter(
            name='controller',
            choices=['LQR', 'random'])
        min_control_duration = CSH.UniformIntegerHyperparameter(
            'problem.min_control_duration', lower=1, upper=20, default_value=1, log=True)
        # this is really the difference between max duration and min min duration
        max_control_duration = CSH.UniformIntegerHyperparameter(
            'problem.max_control_duration', lower=1, upper=100, default_value=1, log=True)
        propagation_step_size = CSH.UniformFloatHyperparameter(
            'problem.propagation_step_size', lower=0.1, upper=10., log=True)
        rnge = CSH.UniformFloatHyperparameter(
            'range', lower=0.001, upper=10000, log=True)
        intermediate_states = CSH.UniformIntegerHyperparameter(
            'intermediate_states', lower=0, upper=1, default_value=0)
        goal_bias = CSH.UniformFloatHyperparameter(
            'goal_bias', lower=0., upper=1., default_value=.05)
        border_fraction = CSH.UniformFloatHyperparameter(
            'border_fraction', lower=0., upper=1., default_value=.8)
        pruning_radius = CSH.UniformFloatHyperparameter(
            'pruning_radius', lower=0.1, upper=100., log=True, default_value=.1)
        selection_radius = CSH.UniformFloatHyperparameter(
            'selection_radius', lower=0.1, upper=100., log=True, default_value=.2)
        cs.add_hyperparameters([planner, controller, min_control_duration, max_control_duration,
                                propagation_step_size, rnge,
                                intermediate_states, goal_bias, border_fraction,
                                pruning_radius, selection_radius])

        cs.add_conditions([
            CS.EqualsCondition(rnge, planner, 'EST'),
            CS.OrConjunction(CS.EqualsCondition(goal_bias, planner, 'EST'),
                             CS.EqualsCondition(goal_bias, planner, 'KPIECE'),
                             CS.EqualsCondition(goal_bias, planner, 'PDST'),
                             CS.EqualsCondition(goal_bias, planner, 'RRT'),
                             CS.EqualsCondition(goal_bias, planner, 'SST')),
            CS.EqualsCondition(border_fraction, planner, 'KPIECE'),
            CS.EqualsCondition(intermediate_states, planner, 'RRT'),
            CS.EqualsCondition(pruning_radius, planner, 'SST'),
            CS.EqualsCondition(selection_radius, planner, 'SST')
        ])
        return cs

class OptWorker(BaseWorker):
    def __init__(self, config_files, *args, **kwargs):
        super().__init__(config_files, {'path_length': 'solution length REAL'},
                         *args,
                         selected_progress_properties={'time': 'time REAL',
                                                       'cost': 'best cost REAL'},
                         **kwargs)

    def loss(self, budget, results):
        return quantile_with_fallback(results['_progress_loss'], self.MAX_COST)

    def area_under_curve(self, time, cost):
        # the cost over the interval [0, time_to_first_solution] is set to be
        # equal to cost of first solution
        if not time:
            return self.MAX_COST
        ind = np.isfinite(cost)
        cost = np.array(cost)[ind]
        time = np.array(time)[ind]
        if time.size == 0:
            return self.MAX_COST
        cost = np.insert(cost, 0, cost[0])
        time = np.insert(time, 0, 0.)
        return np.trapz(cost, time)

    def progress_loss(self, budget, progress_data):
        return [self.area_under_curve(t, c)
                for t, c in zip(progress_data['time'], progress_data['cost'])]

    def duration_runs(self, budget):
        return budget/10., 10

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
                'SPARS2',
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
                CS.EqualsCondition(stretch_factor, planner, 'SPARS2')
            ),
            CS.OrConjunction(
                CS.EqualsCondition(sparse_delta_fraction, planner, 'SPARS'),
                CS.EqualsCondition(sparse_delta_fraction, planner, 'SPARS2')
            ),
            CS.OrConjunction(
                CS.EqualsCondition(dense_delta_fraction, planner, 'SPARS'),
                CS.EqualsCondition(dense_delta_fraction, planner, 'SPARS2')
            ),
            CS.OrConjunction(
                CS.EqualsCondition(max_failures, planner, 'SPARS'),
                CS.EqualsCondition(max_failures, planner, 'SPARS2')
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
