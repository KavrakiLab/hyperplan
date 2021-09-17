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

from math import pi
from tempfile import NamedTemporaryFile
from collections import defaultdict
from pathlib import Path
import numpy as np
import yaml
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from .util import quantile_with_fallback
from .base_worker import BaseWorker


def resolve_package(path):
    """Resolves `package://` URLs to their canonical form. The path does not need
    to exist, but the package does. Can be used to write new files in packages.

    Returns "" on failure.
    """
    if not path:
        return ""

    package_name = ""
    package_path1 = ""
    PREFIX = "package://"
    if PREFIX in path:
        path = path[len(PREFIX) :]  # Remove "package://"
        if "/" not in path:
            package_name = path
            path = ""
        else:
            package_name = path[: path.find("/")]
            path = path[path.find("/") :]

        package_path1 = (
            subprocess.check_output(["rospack", "find", package_name]).decode().strip()
        )

    elif "~" in path:
        path = os.path.expanduser(path)

    new_path = os.path.realpath(package_path1 + path)
    return new_path


def resolve_path(path):
    """Resolves `package://` URLs and relative file paths to their canonical form.
    Returns "" on failure.
    """
    full_path = resolve_package(path)
    if not os.path.exists(full_path):
        logging.warn("File {} does not exist".format(full_path))
        return ""
    return full_path


class RobowflexWorkcellWorker(BaseWorker):
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

    def initialize_problems(self, config):
        if config:
            self.config = config

    def _pose_param_value(self, config, pose, param):
        key = f"{pose}_{param}"
        return config[key] if key in config else float(self.config[pose][param])

    def _pose_to_list(self, config, pose):
        return [
            self._pose_param_value(config, pose, param)
            for param in ["x", "y", "z", "rotx", "roty", "rotz"]
        ]

    def compute(self, config_id, config, budget, working_directory):
        duration, num_runs = self.duration_runs(budget)
        results = defaultdict(list)
        excluded_params = [
            "robot_type",
            "tool_offset",
            "pallet",
            "pallet_pick_percentage",
            "robot_pose",
            "conveyor_pose",
            "planner",
            "projection",
        ]
        planner_config = {}
        for key, value in config.items():
            for param in excluded_params:
                if key.startswith(param):
                    break
            else:
                planner_config[key] = value
        planner_config["type"] = "geometric::" + config["planner"]
        robot = self.config["robot_type"][config["robot_type"]]
        joint_group = robot["joint_group"]
        ompl_config = yaml.load(
            open(resolve_path(robot["ompl_planning"]), "r"), Loader=yaml.FullLoader
        )
        ompl_config[joint_group["name"]]["planner_configs"] = ["planner"]
        ompl_config["planner_configs"] = {"planner": planner_config}
        if "projection" in config:
            ompl_config[joint_group["name"]]["projection_evaluator"] = joint_group[
                config["projection"]
            ]
        else:
            ompl_config[joint_group["name"]].pop("projection_evaluator", None)
        # write ompl_planning.yaml
        ompl_cfg_file = NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=(not self.keep_log_files)
        )
        yaml.dump(ompl_config, ompl_cfg_file)

        # write robowflex_workcell_helper config
        robot["ompl_planning"] = ompl_cfg_file.name
        helper_config = {
            "robot_type": robot,
            "tool_offset": config["tool_offset"],
            "pallet": self.config["pallet_type"][config["pallet_type"]],
            "pallet_pick_percentage": config["pallet_pick_percentage"],
            "robot_pose": self._pose_to_list(config, "robot_pose"),
            "conveyor_pose": self._pose_to_list(config, "conveyor_pose"),
            "conveyor_dimensions": self.config["conveyor_dimensions"],
            "duration": duration,
            "num_runs": num_runs,
        }
        cfg_file = NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=(not self.keep_log_files)
        )
        yaml.dump(helper_config, cfg_file)
        return {"loss": 0, "info": {}}
        # write ompl_config

    def individual_losses(self, budget, results):
        return [
            quantile_with_fallback(t[:-1], budget + d[-1] * d[-1])
            for t, d in zip(results["time"], results["goal_distance"])
        ]

    def progress_loss(self, budget, progress_data):
        raise Exception("Not implemented for this class")

    def duration_runs(self, budget):
        return budget, 0

    def _get_pose_params(self, pose_name):
        pose_params = []
        pose_dict = self.config[pose_name]
        for param in ["x", "y", "z"]:
            if param in pose_dict and isinstance(pose_dict[param], list):
                pose_params.append(
                    CSH.UniformFloatHyperparameter(
                        name=pose_name + "_" + param,
                        lower=pose_dict[param][0],
                        upper=pose_dict[param][1],
                    )
                )
        for param in ["rotx", "roty", "rotz"]:
            if param in pose_dict:
                val = pose_dict[param]
                if isinstance(val, list):
                    pose_params.append(
                        CSH.UniformFloatHyperparameter(
                            name=pose_name + "_" + param, lower=val[0], upper=val[1]
                        )
                    )
                elif val == True:
                    pose_params.append(
                        CSH.UniformFloatHyperparameter(
                            name=pose_name + "_" + param, lower=-pi, upper=pi
                        )
                    )
        return pose_params

    def get_configspace(self):
        cs = CS.ConfigurationSpace()
        robot_type = CSH.CategoricalHyperparameter(
            name="robot_type", choices=self.config["robot_type"].keys()
        )
        tool_offset = CSH.UniformFloatHyperparameter(
            name="tool_offset",
            lower=self.config["tool_offset"][0],
            upper=self.config["tool_offset"][1],
        )
        pallet_type = CSH.CategoricalHyperparameter(
            name="pallet_type", choices=self.config["pallet_type"].keys()
        )
        pallet_pick_percentage = CSH.UniformIntegerHyperparameter(
            "pallet_pick_percentage", lower=1, upper=100, default_value=50
        )
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
                "shoulder",
                "wrist",
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
                robot_type,
                tool_offset,
                pallet_type,
                pallet_pick_percentage,
                planner,
                projection,
                max_nearest_neighbors,
                rnge,
                intermediate_states,
                goal_bias,
            ]
            + self._get_pose_params("robot_pose")
            + self._get_pose_params("conveyor_pose")
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