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
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from . import quantile_with_fallback, nanquantile_with_fallback
from .base_worker import BaseWorker

class RobowflexBaseWorker(BaseWorker):
    def initialize_problems(self, configs):
        config_dir = Path(configs[0])
        self.config_template = open(config_dir / 'ompl_planning.yaml', 'r').read()
        self.problems = list(zip(sorted(config_dir.glob('*scene*.yaml')), sorted(config_dir.glob('*request*.yaml'))))

    def compute(self, config_id, config, budget, working_directory):
        duration, num_runs = self.duration_runs(budget)
        results = defaultdict(list)
        planner = config['planner']
        excluded_params = ['planner', 'projection']
        params = '\n'.join([f'    {key}: {value}' for key, value in config.items()
                            if not key in excluded_params])
        if 'projection' in config:
            projection = 'projection_evaluator: ' + config['projection']
        else:
            projection = '# no projection'
        cfg = self.config_template.format(
            **{'planner':planner, 'params':params, 'projection': projection})
        cfg_file_handle, abs_path = mkstemp(suffix='.yaml', text=True)
        os.write(cfg_file_handle, cfg.encode())
        os.close(cfg_file_handle)

        for scene, request in self.problems:
            try:
                log_dir = abs_path + '_logs/'
                subprocess.run(['rosrun', 'mphpo', 'robowflex_helper', str(scene), str(request), abs_path, str(duration),
                                str(num_runs), log_dir], check=True)
            except subprocess.CalledProcessError as err:
                logging.warning(
                    f'This command terminated with error code {err.returncode}:\n\t{err.cmd}')
                for key in self.selected_properties.keys():
                    results[key].append([budget if key == 'time' else np.nan])
                for key in self.selected_progress_properties.keys():
                    results[key].append([self.MAX_COST if key == 'cost' else np.nan])
                continue
            if not self.keep_log_files:
                os.remove(abs_path)
                os.remove(log_dir)

            log_path = log_dir + str(scene) + '.log'
            results = self.update_results(results, budget, log_path)

        return {'loss': self.loss(budget, results), 'info': results}

class SpeedWorker(RobowflexBaseWorker):
    def __init__(self, config_files, *args, **kwargs):
        super().__init__(config_files,
                         {'time': 'time REAL', 'path_length': 'length REAL'},
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
                'ProjEST',
                'KPIECE',
                'LazyPRM',
                'LBKPIECE',
                'PDST',
                'PRM',
                'RRT',
                'RRTConnect',
                'SBL',
                'STRIDE'])
        projection = CSH.CategoricalHyperparameter(
            name='projection',
            choices=[
                'joints(torso_lift_joint,shoulder_pan_joint)',
                'link(wrist_roll_link)'
            ]
        )
        max_nearest_neighbors = CSH.UniformIntegerHyperparameter(
            'max_nearest_neighbors', lower=1, upper=20, default_value=8)
        rnge = CSH.UniformFloatHyperparameter(
            'range', lower=0.001, upper=10000, log=True)
        intermediate_states = CSH.UniformIntegerHyperparameter(
            'intermediate_states', lower=0, upper=1, default_value=0)
        goal_bias = CSH.UniformFloatHyperparameter(
            'goal_bias', lower=0., upper=1., default_value=.05)
        cs.add_hyperparameters([planner, projection, max_nearest_neighbors, rnge, intermediate_states,
                                goal_bias])

        cs.add_conditions([
            CS.OrConjunction(CS.EqualsCondition(projection, planner, 'ProjEST'),
                             CS.EqualsCondition(projection, planner, 'SBL'),
                             CS.EqualsCondition(projection, planner, 'KPIECE'),
                             CS.EqualsCondition(projection, planner, 'BKPIECE'),
                             CS.EqualsCondition(projection, planner, 'LBKPIECE'),
                             CS.EqualsCondition(projection, planner, 'PDST'),
                             CS.EqualsCondition(projection, planner, 'STRIDE')),
            CS.OrConjunction(CS.EqualsCondition(max_nearest_neighbors, planner, 'PRM'),
                             CS.EqualsCondition(max_nearest_neighbors, planner, 'LazyPRM')),
            CS.OrConjunction(CS.EqualsCondition(rnge, planner, 'LazyPRM'),
                             CS.EqualsCondition(rnge, planner, 'RRT'),
                             CS.EqualsCondition(rnge, planner, 'RRTConnect'),
                             CS.EqualsCondition(rnge, planner, 'EST'),
                             CS.EqualsCondition(rnge, planner, 'ProjEST'),
                             CS.EqualsCondition(rnge, planner, 'BiEST'),
                             CS.EqualsCondition(rnge, planner, 'SBL'),
                             CS.EqualsCondition(rnge, planner, 'KPIECE'),
                             CS.EqualsCondition(rnge, planner, 'BKPIECE'),
                             CS.EqualsCondition(rnge, planner, 'LBKPIECE'),
                             CS.EqualsCondition(rnge, planner, 'PDST'),
                             CS.EqualsCondition(rnge, planner, 'STRIDE')),
            CS.OrConjunction(CS.EqualsCondition(intermediate_states, planner, 'RRT'),
                             CS.EqualsCondition(intermediate_states, planner, 'RRTConnect')),
            CS.OrConjunction(CS.EqualsCondition(goal_bias, planner, 'RRT'),
                             CS.EqualsCondition(goal_bias, planner, 'EST'),
                             CS.EqualsCondition(goal_bias, planner, 'ProjEST'),
                             CS.EqualsCondition(goal_bias, planner, 'KPIECE'),
                             CS.EqualsCondition(goal_bias, planner, 'STRIDE'))
        ])
        return cs
