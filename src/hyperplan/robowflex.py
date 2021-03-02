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
from .util import quantile_with_fallback
from .base_worker import BaseWorker


class RobowflexBaseWorker(BaseWorker):
    def initialize_problems(self, config):
        if config:
            self.config_template = open(config["ompl_config_template"]).read()
            self.scenes = sorted([p.resolve() for p in Path(config["input_dir"]).glob(config["input_scenes"])])
            self.requests = sorted([p.resolve() for p in Path(config["input_dir"]).glob(config["input_requests"])])

    def batch_test(self, opt_config, test_config):
        # save original template and problems
        config_template = self.config_template
        scenes = self.scenes
        requests = self.requests
        result = {}
        for testname, test in test_config['tests'].items():
            self.config_template = open(test['ompl_config_template']).read() if 'ompl_config_template' in test else config_template
            self.scenes = sorted(set([p.resolve() for p in Path(test_config["input_dir"]).glob(test["input_scenes"])]).difference(scenes))
            if not self.scenes:
                self.scenes = scenes
            self.requests = sorted(set([p.resolve() for p in Path(test_config["input_dir"]).glob(test["input_requests"])]).difference(requests))
            if not self.requests:
                self.requests = requests
            result[testname] = self.compute(None, opt_config, test_config['test_budget'], None)
        # restore original template and problems
        self.config_template = config_template
        self.scenes = scenes
        self.requests = requests
        return result

    def compute(self, config_id, config, budget, working_directory):
        duration, num_runs = self.duration_runs(budget)
        results = defaultdict(list)
        planner = config["planner"]
        excluded_params = ["planner", "projection"]
        params = "\n".join(
            [
                f"    {key}: {value}"
                for key, value in config.items()
                if not key in excluded_params
            ]
        )
        if "projection" in config:
            projection = "projection_evaluator: " + config["projection"]
        else:
            projection = "# no projection"
        cfg = self.config_template.format(
            **{"planner": planner, "params": params, "projection": projection}
        )
        cfg_file_handle, abs_path = mkstemp(suffix=".yaml", text=True)
        os.write(cfg_file_handle, cfg.encode())
        os.close(cfg_file_handle)

        for scene, request in zip(self.scenes, self.requests):
            try:
                log_dir = abs_path + "_logs/"
                cmdline = [
                    "rosrun",
                    "hyperplan",
                    "robowflex_helper",
                    str(scene),
                    str(request),
                    abs_path,
                    str(duration),
                    str(num_runs),
                    log_dir,
                ]
                if self.simplify:
                    cmdline.append("simplify")
                subprocess.run(cmdline, check=True)
            except subprocess.CalledProcessError as err:
                logging.warning(
                    f"This command terminated with error code {err.returncode}:\n\t{err.cmd}"
                )
                for key in self.selected_properties.keys():
                    results[key].append([budget if key == "time" else np.nan])
                for key in self.selected_progress_properties.keys():
                    results[key].append([self.MAX_COST if key == "cost" else np.nan])
                continue
            if not self.keep_log_files:
                os.remove(abs_path)
                os.remove(log_dir)

            log_path = log_dir + str(scene) + ".log"
            results = self.update_results(results, budget, log_path)

        return {"loss": self.loss(budget, results), "info": results}


class SpeedWorker(RobowflexBaseWorker):
    def __init__(self, config, *args, **kwargs):
        super().__init__(
            config,
            {
                "time": "time REAL",
                "path_length": "length REAL",
                "goal_distance": "goal_distance REAL",
            },
            *args,
            **kwargs,
        )

    def individual_losses(self, budget, results):
        return [
            quantile_with_fallback(t[:-1], budget + d[-1] * d[-1])
            for t, d in zip(results["time"], results["goal_distance"])
        ]

    def progress_loss(self, budget, progress_data):
        raise Exception("Not implemented for this class")

    def duration_runs(self, budget):
        return budget, 0

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        planner = CSH.CategoricalHyperparameter(
            name="planner",
            choices=[
                "BiEST",
                "BKPIECE",
                "EST",
                "ProjEST",
                "KPIECE",
                "LazyPRM",
                "LBKPIECE",
                "PDST",
                "PRM",
                "RRT",
                "RRTConnect",
                "SBL",
                "STRIDE",
            ],
        )
        projection = CSH.CategoricalHyperparameter(
            name="projection",
            choices=[
                "joints(torso_lift_joint,shoulder_pan_joint)",
                "link(wrist_roll_link)",
            ],
        )
        max_nearest_neighbors = CSH.UniformIntegerHyperparameter(
            "max_nearest_neighbors", lower=1, upper=20, default_value=8
        )
        rnge = CSH.UniformFloatHyperparameter("range", lower=1, upper=500)
        intermediate_states = CSH.UniformIntegerHyperparameter(
            "intermediate_states", lower=0, upper=1, default_value=0
        )
        goal_bias = CSH.UniformFloatHyperparameter(
            "goal_bias", lower=0.0, upper=1.0, default_value=0.05
        )
        cs.add_hyperparameters(
            [
                planner,
                projection,
                max_nearest_neighbors,
                rnge,
                intermediate_states,
                goal_bias,
            ]
        )

        cs.add_conditions(
            [
                CS.OrConjunction(
                    CS.EqualsCondition(projection, planner, "ProjEST"),
                    CS.EqualsCondition(projection, planner, "SBL"),
                    CS.EqualsCondition(projection, planner, "KPIECE"),
                    CS.EqualsCondition(projection, planner, "BKPIECE"),
                    CS.EqualsCondition(projection, planner, "LBKPIECE"),
                    CS.EqualsCondition(projection, planner, "PDST"),
                    CS.EqualsCondition(projection, planner, "STRIDE"),
                ),
                CS.OrConjunction(
                    CS.EqualsCondition(max_nearest_neighbors, planner, "PRM"),
                    CS.EqualsCondition(max_nearest_neighbors, planner, "LazyPRM"),
                ),
                CS.OrConjunction(
                    CS.EqualsCondition(rnge, planner, "LazyPRM"),
                    CS.EqualsCondition(rnge, planner, "RRT"),
                    CS.EqualsCondition(rnge, planner, "RRTConnect"),
                    CS.EqualsCondition(rnge, planner, "EST"),
                    CS.EqualsCondition(rnge, planner, "ProjEST"),
                    CS.EqualsCondition(rnge, planner, "BiEST"),
                    CS.EqualsCondition(rnge, planner, "SBL"),
                    CS.EqualsCondition(rnge, planner, "KPIECE"),
                    CS.EqualsCondition(rnge, planner, "BKPIECE"),
                    CS.EqualsCondition(rnge, planner, "LBKPIECE"),
                    CS.EqualsCondition(rnge, planner, "PDST"),
                    CS.EqualsCondition(rnge, planner, "STRIDE"),
                ),
                CS.OrConjunction(
                    CS.EqualsCondition(intermediate_states, planner, "RRT"),
                    CS.EqualsCondition(intermediate_states, planner, "RRTConnect"),
                ),
                CS.OrConjunction(
                    CS.EqualsCondition(goal_bias, planner, "RRT"),
                    CS.EqualsCondition(goal_bias, planner, "EST"),
                    CS.EqualsCondition(goal_bias, planner, "ProjEST"),
                    CS.EqualsCondition(goal_bias, planner, "KPIECE"),
                    CS.EqualsCondition(goal_bias, planner, "STRIDE"),
                ),
            ]
        )
        return cs


class SpeedKinodynamicWorker(RobowflexBaseWorker):
    def __init__(self, config, *args, **kwargs):
        super().__init__(
            config,
            {
                "time": "time REAL",
                "path_length": "length REAL",
                "goal_distance": "goal_distance REAL",
            },
            *args,
            **kwargs,
        )
        self.simplify = 1
        self.speed = 10

    def individual_losses(self, budget, results):
        # planning time + path length (=duration in seconds) + square of goal distance
        return [
            quantile_with_fallback(
                np.array(t[:-1]) + np.array(pl[:-1]) / self.speed,
                self.MAX_COST + d[-1] * d[-1] / (self.speed * self.speed)
                if np.isfinite(d[-1])
                else 2 * self.MAX_COST,
            )
            for t, pl, d in zip(
                results["time"], results["path_length"], results["goal_distance"]
            )
        ]

    def progress_loss(self, budget, progress_data):
        raise Exception("Not implemented for this class")

    def duration_runs(self, budget):
        return budget, 0

    @staticmethod
    def get_configspace():
        return SpeedWorker.get_configspace()

    # @staticmethod
    # def get_configspace():
    #     cs = CS.ConfigurationSpace()
    #     planner = CSH.CategoricalHyperparameter(
    #         name='planner',
    #         choices=[
    #             'EST',
    #             'KPIECE',
    #             'PDST',
    #             'RRT',
    #             'SST',
    #             'SyclopEST',
    #             'SyclopRRT'])
    #     controller = CSH.CategoricalHyperparameter(
    #         name='problem.controller',
    #         choices=['LQR', 'random'])
    #     min_control_duration = CSH.UniformIntegerHyperparameter(
    #         'problem.min_control_duration', lower=1, upper=20, default_value=1, log=True)
    #     # this is really the difference between max duration and min min duration
    #     max_control_duration = CSH.UniformIntegerHyperparameter(
    #         'problem.max_control_duration', lower=1, upper=100, default_value=1, log=True)
    #     propagation_step_size = CSH.UniformFloatHyperparameter(
    #         'problem.propagation_step_size', lower=0.1, upper=10., log=True)
    #     rnge = CSH.UniformFloatHyperparameter(
    #         'range', lower=0.001, upper=10000, log=True)
    #     intermediate_states = CSH.UniformIntegerHyperparameter(
    #         'intermediate_states', lower=0, upper=1, default_value=0)
    #     goal_bias = CSH.UniformFloatHyperparameter(
    #         'goal_bias', lower=0., upper=1., default_value=.05)
    #     border_fraction = CSH.UniformFloatHyperparameter(
    #         'border_fraction', lower=0., upper=1., default_value=.8)
    #     pruning_radius = CSH.UniformFloatHyperparameter(
    #         'pruning_radius', lower=0.1, upper=100., log=True, default_value=.1)
    #     selection_radius = CSH.UniformFloatHyperparameter(
    #         'selection_radius', lower=0.1, upper=100., log=True, default_value=.2)
    #     cs.add_hyperparameters([planner, controller, min_control_duration, max_control_duration,
    #                             propagation_step_size, rnge,
    #                             intermediate_states, goal_bias, border_fraction,
    #                             pruning_radius, selection_radius])

    #     cs.add_conditions([
    #         CS.EqualsCondition(rnge, planner, 'EST'),
    #         CS.OrConjunction(CS.EqualsCondition(goal_bias, planner, 'EST'),
    #                          CS.EqualsCondition(goal_bias, planner, 'KPIECE'),
    #                          CS.EqualsCondition(goal_bias, planner, 'PDST'),
    #                          CS.EqualsCondition(goal_bias, planner, 'RRT'),
    #                          CS.EqualsCondition(goal_bias, planner, 'SST')),
    #         CS.EqualsCondition(border_fraction, planner, 'KPIECE'),
    #         CS.EqualsCondition(intermediate_states, planner, 'RRT'),
    #         CS.EqualsCondition(pruning_radius, planner, 'SST'),
    #         CS.EqualsCondition(selection_radius, planner, 'SST')
    #     ])
    #     return cs


class OptWorker(RobowflexBaseWorker):
    def __init__(self, config, *args, **kwargs):
        super().__init__(
            config,
            {
                "path_length": "length REAL",
                "goal_distance": "goal_distance REAL",
            },
            *args,
            selected_progress_properties={
                "time": "time REAL",
                "cost": "best cost REAL",
            },
            **kwargs,
        )

    def problem_loss(self, progress_loss, goal_distance):
        return np.quantile(
            [
                pl
                if np.isfinite(pl)
                else self.MAX_COST + gd
                if np.isfinite(gd)
                else 2 * self.MAX_COST
                for pl, gd in zip(progress_loss, goal_distance)
            ],
            0.7,
        )

    def individual_losses(self, _, results):
        return [
            self.problem_loss(pls, gds)
            for pls, gds in zip(results["_progress_loss"], results["goal_distance"])
        ]

    def area_under_curve(self, time, cost):
        # the cost over the interval [0, time_to_first_solution] is set to be
        # equal to cost of first solution
        if not time:
            return np.nan
        ind = np.isfinite(cost)
        cost = np.array(cost)[ind]
        time = np.array(time)[ind]
        if time.size == 0:
            return np.nan
        cost = np.insert(cost, 0, cost[0])
        time = np.insert(time, 0, 0.0)
        return np.trapz(cost, time)

    def progress_loss(self, budget, progress_data):
        return [
            self.area_under_curve(t, c) / budget
            for t, c in zip(progress_data["time"], progress_data["cost"])
        ]

    def duration_runs(self, budget):
        return budget / 10.0, 10

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        planner = CSH.CategoricalHyperparameter(
            name="planner",
            choices=[
                "AITstar",
                #"AnytimePathShortening",
                "BITstar",
                #"CForest",
                # "FMT",
                # 'LazyLBTRRT',
                # 'LazyPRMstar',
                "LBTRRT",
                # 'PRMstar',
                "RRTstar",
                "RRTXstatic",
                # 'SPARS',
                # 'SPARS2',
                "SST",
            ],
        )
        cs.add_hyperparameter(planner)

        # parameters common among more than one planner
        rnge = CSH.UniformFloatHyperparameter("range", lower=1, upper=500)
        goal_bias = CSH.UniformFloatHyperparameter(
            "goal_bias", lower=0.0, upper=1.0, default_value=0.05
        )
        use_k_nearest = CSH.UniformIntegerHyperparameter(
            "use_k_nearest", lower=0, upper=1, default_value=1
        )
        # epsilon = CSH.UniformFloatHyperparameter(
        #     'epsilon', lower=0., upper=10., default_value=.4)
        # rewire_factor = CSH.UniformFloatHyperparameter(
        #     'rewire_factor', lower=1., upper=2., default_value=1.1)
        # informed_sampling = CSH.UniformIntegerHyperparameter(
        #     'informed_sampling', lower=0, upper=1, default_value=0)
        # sample_rejection = CSH.UniformIntegerHyperparameter(
        #     'sample_rejection', lower=0, upper=1, default_value=0)
        # number_sampling_attempts = CSH.UniformIntegerHyperparameter(
        #     'number_sampling_attempts', lower=10, upper=100000, default_value=100, log=True)
        # stretch_factor = CSH.UniformFloatHyperparameter(
        #     'stretch_factor', lower=1.1, upper=3., default_value=3.)
        # sparse_delta_fraction = CSH.UniformFloatHyperparameter(
        #     'sparse_delta_fraction', lower=0., upper=1., default_value=.25)
        # dense_delta_fraction = CSH.UniformFloatHyperparameter(
        #     'dense_delta_fraction', lower=0., upper=.1, default_value=.001)
        # max_failures = CSH.UniformIntegerHyperparameter(
        #     'max_failures', lower=100, upper=3000, default_value=1000, log=True)
        #focus_search = CSH.UniformIntegerHyperparameter(
        #    "focus_search", lower=0, upper=1, default_value=1
        #)
        samples_per_batch = CSH.UniformIntegerHyperparameter(
            "samples_per_batch", lower=1, upper=10000, default_value=100, log=True
        )

        cs.add_hyperparameters(
            [rnge, goal_bias, use_k_nearest, samples_per_batch]
        )
        # rnge, goal_bias, use_k_nearest, epsilon, rewire_factor, informed_sampling,
        # sample_rejection, number_sampling_attempts, stretch_factor, sparse_delta_fraction,
        # dense_delta_fraction, max_failures, focus_search
        cs.add_conditions(
            [
                CS.OrConjunction(
                    # CS.EqualsCondition(rnge, planner, 'LazyLBTRRT'),
                    # CS.EqualsCondition(rnge, planner, 'LazyPRMstar'),
                    CS.EqualsCondition(rnge, planner, "LBTRRT"),
                    CS.EqualsCondition(rnge, planner, "RRTXstatic"),
                    CS.EqualsCondition(rnge, planner, "SST"),
                ),
                CS.OrConjunction(
                    # CS.EqualsCondition(goal_bias, planner, 'LazyLBTRRT'),
                    CS.EqualsCondition(goal_bias, planner, "LBTRRT"),
                    CS.EqualsCondition(goal_bias, planner, "RRTstar"),
                    CS.EqualsCondition(goal_bias, planner, "RRTXstatic"),
                    CS.EqualsCondition(goal_bias, planner, "SST"),
                ),
                CS.OrConjunction(
                    CS.EqualsCondition(use_k_nearest, planner, "AITstar"),
                    CS.EqualsCondition(use_k_nearest, planner, "BITstar"),
                    # CS.EqualsCondition(use_k_nearest, planner, "FMT"),
                    CS.EqualsCondition(use_k_nearest, planner, "RRTstar"),
                    CS.EqualsCondition(use_k_nearest, planner, "RRTXstatic"),
                ),
                # CS.OrConjunction(
                #     #CS.EqualsCondition(epsilon, planner, 'LazyLBTRRT'),
                #     CS.EqualsCondition(epsilon, planner, 'LBTRRT'),
                #     CS.EqualsCondition(epsilon, planner, 'RRTXstatic')
                # ),
                # CS.OrConjunction(
                #     CS.EqualsCondition(rewire_factor, planner, 'BITstar'),
                #     CS.EqualsCondition(rewire_factor, planner, 'RRTstar'),
                #     CS.EqualsCondition(rewire_factor, planner, 'RRTXstatic')
                # ),
                # CS.OrConjunction(
                #     CS.EqualsCondition(informed_sampling, planner, 'RRTstar'),
                #     CS.EqualsCondition(informed_sampling, planner, 'RRTXstatic')
                # ),
                # CS.OrConjunction(
                #     CS.EqualsCondition(sample_rejection, planner, 'RRTstar'),
                #     CS.EqualsCondition(sample_rejection, planner, 'RRTXstatic')
                # ),
                # CS.OrConjunction(
                #     CS.EqualsCondition(number_sampling_attempts,
                #                        planner, 'RRTstar'),
                #     CS.EqualsCondition(number_sampling_attempts,
                #                        planner, 'RRTXstatic')
                # ),
                # CS.OrConjunction(
                #     CS.EqualsCondition(stretch_factor, planner, 'SPARS'),
                #     CS.EqualsCondition(stretch_factor, planner, 'SPARS2')
                # ),
                # CS.OrConjunction(
                #     CS.EqualsCondition(sparse_delta_fraction, planner, 'SPARS'),
                #     CS.EqualsCondition(sparse_delta_fraction, planner, 'SPARS2')
                # ),
                # CS.OrConjunction(
                #     CS.EqualsCondition(dense_delta_fraction, planner, 'SPARS'),
                #     CS.EqualsCondition(dense_delta_fraction, planner, 'SPARS2')
                # ),
                # CS.OrConjunction(
                #     CS.EqualsCondition(max_failures, planner, 'SPARS'),
                #     CS.EqualsCondition(max_failures, planner, 'SPARS2')
                # ),
                #CS.OrConjunction(
                #    CS.EqualsCondition(focus_search, planner, "RRTstar"),
                #    CS.EqualsCondition(focus_search, planner, "CForest"),
                #),
                CS.OrConjunction(
                    CS.EqualsCondition(samples_per_batch, planner, "AITstar"),
                    CS.EqualsCondition(samples_per_batch, planner, "BITstar"),
                ),
            ]
        )
        # cs.add_forbidden_clause(
        #     CS.ForbiddenAndConjunction(
        #         CS.ForbiddenEqualsClause(informed_sampling, 1),
        #         CS.ForbiddenEqualsClause(sample_rejection, 1)
        #     )
        # )

        # AnytimePathShortening
        #max_hybrid_paths = CSH.UniformIntegerHyperparameter(
        #    "max_hybrid_paths", lower=2, upper=128, default_value=24, log=True
        #)
        #num_planners = CSH.UniformIntegerHyperparameter(
        #    "num_planners", lower=2, upper=8, default_value=4
        #)
        # shortcut = CSH.UniformIntegerHyperparameter(
        #     'shortcut', lower=0, upper=1, default_value=1)
        # hybridize = CSH.UniformIntegerHyperparameter(
        #     'hybridize', lower=0, upper=1, default_value=1)
        #cs.add_hyperparameters(
        #    [
        #        # max_hybrid_paths, num_planners, shortcut, hybridize])
        #        max_hybrid_paths,
        #        num_planners,
        #    ]
        #)
        #cs.add_conditions(
        #    [
        #        CS.EqualsCondition(max_hybrid_paths, planner, "AnytimePathShortening"),
        #        CS.EqualsCondition(num_planners, planner, "AnytimePathShortening")  # ,
        #        # CS.EqualsCondition(shortcut, planner, 'AnytimePathShortening'),
        #        # CS.EqualsCondition(hybridize, planner, 'AnytimePathShortening')
        #    ]
        #)

        # BIT*
        # use_graphPtr_pruning = CSH.UniformIntegerHyperparameter(
        #     'use_graphPtr_pruning', lower=0, upper=1, default_value=1)
        # prune_threshold_as_fractional_cost_change = CSH.UniformFloatHyperparameter(
        #     'prune_threshold_as_fractional_cost_change', lower=0., upper=1., default_value=.05)
        # delay_rewiring_to_first_solution = CSH.UniformIntegerHyperparameter(
        #     'delay_rewiring_to_first_solution', lower=0, upper=1, default_value=1)
        # use_just_in_time_sampling = CSH.UniformIntegerHyperparameter(
        #     'use_just_in_time_sampling', lower=0, upper=1, default_value=0)
        # drop_unconnected_samples_on_prune = CSH.UniformIntegerHyperparameter(
        #     'drop_unconnected_samples_on_prune', lower=0, upper=1, default_value=0)
        # use_strict_queue_ordering = CSH.UniformIntegerHyperparameter(
        #     'use_strict_queue_ordering', lower=0, upper=1, default_value=0)
        # find_approximate_solutions = CSH.UniformIntegerHyperparameter(
        #     'find_approximate_solutions', lower=0, upper=1, default_value=0)
        # cs.add_hyperparameters([
        #     samples_per_batch, use_graphPtr_pruning, prune_threshold_as_fractional_cost_change,
        #     delay_rewiring_to_first_solution, use_just_in_time_sampling,
        #     drop_unconnected_samples_on_prune, use_strict_queue_ordering,
        #     find_approximate_solutions
        # ])
        # cs.add_conditions([
        #     CS.EqualsCondition(samples_per_batch, planner, 'BITstar'),
        #     CS.EqualsCondition(use_graphPtr_pruning, planner, 'BITstar'),
        #     CS.EqualsCondition(
        #         prune_threshold_as_fractional_cost_change, planner, 'BITstar'),
        #     CS.EqualsCondition(
        #         delay_rewiring_to_first_solution, planner, 'BITstar'),
        #     CS.EqualsCondition(use_just_in_time_sampling, planner, 'BITstar'),
        #     CS.EqualsCondition(
        #         drop_unconnected_samples_on_prune, planner, 'BITstar'),
        #     CS.EqualsCondition(use_strict_queue_ordering, planner, 'BITstar'),
        #     CS.EqualsCondition(find_approximate_solutions, planner, 'BITstar')
        # ])

        # CForest
        #num_threads = CSH.UniformIntegerHyperparameter(
        #    "num_threads", lower=2, upper=8, default_value=4
        #)
        #cs.add_hyperparameter(num_threads)
        #cs.add_condition(CS.EqualsCondition(num_threads, planner, "CForest"))

        # FMT
        # num_samples = CSH.UniformIntegerHyperparameter(
        #    "num_samples", lower=1000, upper=50000, default_value=1000, log=True
        # )
        # cs.add_hyperparameter(num_samples)
        # cs.add_condition(CS.EqualsCondition(num_samples, planner, "FMT"))

        # RRT*
        delay_collision_checking = CSH.UniformIntegerHyperparameter(
            "delay_collision_checking", lower=0, upper=1, default_value=1
        )
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
        cs.add_condition(
            CS.EqualsCondition(delay_collision_checking, planner, "RRTstar")
        )
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
        # update_children = CSH.UniformIntegerHyperparameter(
        #     'update_children', lower=0, upper=1, default_value=1)
        rejection_variant = CSH.UniformIntegerHyperparameter(
            "rejection_variant", lower=0, upper=3, default_value=0
        )
        # rejection_variant_alpha = CSH.UniformFloatHyperparameter(
        #     'rejection_variant_alpha', lower=0., upper=1., default_value=1.)
        cs.add_hyperparameters([rejection_variant])
        cs.add_conditions(
            [
                # CS.EqualsCondition(update_children, planner, 'RRTXstatic'),
                CS.EqualsCondition(rejection_variant, planner, "RRTXstatic")  # ,
                # CS.EqualsCondition(rejection_variant_alpha, planner, 'RRTXstatic')
            ]
        )

        # SST
        selection_radius = CSH.UniformFloatHyperparameter(
            "selection_radius", lower=.01, upper=5.0, default_value=2.0
        )
        pruning_radius = CSH.UniformFloatHyperparameter(
            "pruning_radius", lower=.01, upper=5.0, default_value=0.2
        )
        cs.add_hyperparameters([selection_radius, pruning_radius])
        cs.add_conditions(
            [
                CS.EqualsCondition(selection_radius, planner, "SST"),
                CS.EqualsCondition(pruning_radius, planner, "SST"),
            ]
        )

        return cs
