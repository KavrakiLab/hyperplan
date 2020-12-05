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
import subprocess
import logging
from tempfile import mkstemp
from collections import defaultdict
from pathlib import Path
import numpy as np
import rospkg
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from . import quantile_with_fallback, nanquantile_with_fallback
from .base_worker import BaseWorker

class MoveItSpeedWorker(BaseWorker):
    def __init__(self, config_prefixes, *args, **kwargs):
        rospack = rospkg.RosPack()
        base_path = Path(rospack.get_path('hyperplan'))
        self.launch_files = [base_path / 'launch' / (config + '.launch')
            for config in config_prefixes]
        config_files = [base_path / 'examples' / (config + '.yaml')
            for config in config_prefixes]
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
    def compute(self, config_id, config, budget, working_directory):
        duration, num_runs = self.duration_runs(budget)
        results = defaultdict(list)

        template_values = {'num_rums': num_runs, 'duration': duration, 'output_dir': }
        for problem in self.problems:
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
            results = self.update_results(results, budget, log_path)

        return {'loss': self.loss(budget, results), 'info': results}

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        pipeline = CSH.CategoricalHyperparameter(
            name='pipeline',
            choices=[
                'ompl',
                'stomp',
                'chomp'
                ])
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
            CS.EqualsCondition(planner, pipeline, 'ompl'),
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
