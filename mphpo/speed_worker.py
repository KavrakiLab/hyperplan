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

def quantile_with_fallback(x, fallback):
    return np.quantile(x, .7) if x else fallback

class SpeedBaseWorker(Worker,ABC):
    LOG_REGEXP = re.compile(
        r'[\d]+ properties for each run\n([\w\s]+)\n' + \
        r'[\d]+ runs\n([\w\s.;e+-]+)([.]|[\d]+ progress properties for each run)\n')

    def __init__(self, config_files, selected_properties, *args, **kwargs):
        Worker.__init__(self, *args, **kwargs)
        self.problems = [fp.read() for fp in config_files]
        self.keep_log_files = True
        self.selected_properties = selected_properties
        self.selected_properties['time'] = 'time REAL'

    @abstractmethod
    def loss(self, budget, results):
        pass

    def compute(self, config_id, config, budget, working_directory):
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
            problem += f'\n\n[benchmark]\ntime_limit={budget}\nmem_limit=100000\nrun_count=0' \
                       f'\n\n[planner]\n{planner}=\n'
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
                continue
            if not self.keep_log_files:
                os.remove(abs_path)

            log_path = abs_path[:-3] + 'log'
            with open(log_path, 'r') as logfile:
                match = SpeedWorker.LOG_REGEXP.search(logfile.read())
                properties = match.group(1).splitlines()
                values = [[float(x) if x else np.nan for x in line.split('; ')]
                          for line in match.group(2).splitlines()]
                for key, val in self.selected_properties.items():
                    results[key].append([run[properties.index(val)] for run in values])
            if not self.keep_log_files:
                os.remove(log_path)

        return {'loss': self.loss(budget, results), 'info': results}

class SpeedWorker(SpeedBaseWorker):
    def __init__(self, config_files, *args, **kwargs):
        super().__init__(config_files, {'path_length': 'simplified solution length REAL'},
                         *args, **kwargs)

    def loss(self, budget, results):
        # skip last run since it is always a timeout with no solution found
        return np.sum([quantile_with_fallback(length, budget)
                       for length in results['path_length'][:-1]])

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

class SpeedKinodynamicWorker(SpeedBaseWorker):
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
        cs.add_hyperparameters([planner, min_control_duration, max_control_duration,
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
