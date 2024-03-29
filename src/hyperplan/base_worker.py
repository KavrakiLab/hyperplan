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
from tempfile import mkstemp
from collections import defaultdict
from abc import ABC, abstractmethod
import numpy as np
from hpbandster.core.worker import Worker


class BaseWorker(Worker, ABC):
    LOG_PROPERTIES_REGEXP = re.compile(
        r"[\d]+ properties for each run\n([\w\s]+)\n[\d]+ runs\n([infean\d\s.;+-]+)\n"
        + r"([.]|[\d]+ progress properties for each run)\n"
    )
    LOG_PROGRESS_PROPERTIES_REGEXP = re.compile(
        r"[\d]+ progress properties for each run\n([\w\s]+)\n[\d]+ runs\n([\w\s.;,+-]+)\n[.]\n"
    )
    MAX_COST = 1e5

    def __init__(
        self,
        config,
        selected_properties,
        *args,
        selected_progress_properties={},
        **kwargs,
    ):
        Worker.__init__(self, *args, **kwargs)
        self.initialize_problems(config)
        self.keep_log_files = True
        self.selected_properties = selected_properties
        self.selected_progress_properties = selected_progress_properties
        self.simplify = 0

    @abstractmethod
    def initialize_problems(self, configs):
        pass

    @abstractmethod
    def individual_losses(self, budget, results):
        pass

    def loss(self, budget, results):
        return np.mean(self.individual_losses(budget, results))

    @abstractmethod
    def progress_loss(self, budget, progress_data):
        pass

    @abstractmethod
    def duration_runs(self, budget):
        pass

    def offline_loss(self, budget, logfiles):
        """Helper function to compute the loss function for benchmark log files produced
        outside of the hyperplan framework."""

        results = defaultdict(list)
        for logfile in logfiles:
            results = self.update_results(results, budget, logfile)
        return self.loss(budget, results)

    def compute(self, config_id, config, budget, working_directory):
        duration, num_runs = self.duration_runs(budget)
        results = defaultdict(list)

        for problem in self.problems:
            try:
                config["problem.max_control_duration"] += config[
                    "problem.min_control_duration"
                ]
            except KeyError:
                pass

            for param, value in config.items():
                if param.startswith("problem."):
                    problem += "\n%s=%s" % (param[8:], value)
            planner = config["planner"].lower()
            problem += (
                f"\n\n[benchmark]\ntime_limit={duration}\nmem_limit=100000\n"
                f"run_count={num_runs}\nsimplify={self.simplify}\n\n[planner]\n{planner}=\n"
            )
            for param, value in config.items():
                if not param == "planner" and not param.startswith("problem."):
                    problem += f"{planner}.{param}={value}\n"
            cfg_file_handle, abs_path = mkstemp(suffix=".cfg", text=True)
            os.write(cfg_file_handle, problem.encode())
            os.close(cfg_file_handle)
            try:
                subprocess.run(
                    ["ompl_benchmark", abs_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            except subprocess.CalledProcessError as err:
                logging.warning(
                    f"This command terminated with error code {err.returncode}:\n\t{err.cmd}\n"
                    f"and produced the following output:\n\n{err.output}"
                )
                for key in self.selected_properties.keys():
                    results[key].append([budget if key == "time" else np.nan])
                for key in self.selected_progress_properties.keys():
                    results[key].append([np.nan])
                continue
            if not self.keep_log_files:
                os.remove(abs_path)

            log_path = abs_path[:-3] + "log"
            results = self.update_results(results, budget, log_path)

        return {"loss": self.loss(budget, results), "info": results}

    def update_results(self, results, budget, log_path):
        with open(log_path, "r") as logfile:
            log = logfile.read()
            match = self.LOG_PROPERTIES_REGEXP.search(log)
            # process run properties
            properties = match.group(1).splitlines()
            values = [
                [float(x) if x else np.nan for x in line.split("; ")]
                for line in match.group(2).splitlines()
            ]
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
                values = [
                    [
                        [float(x) for x in tple.split(",")[:-1]]
                        for tple in line.split(";")[:-1]
                    ]
                    for line in match.group(2).splitlines()
                ]
                progress_data = defaultdict(list)
                for key, val in self.selected_progress_properties.items():
                    if val in properties:
                        progress_data[key] = [
                            [tple[properties.index(val)] for tple in run]
                            for run in values
                        ]
                    else:
                        logging.warning(
                            f'progress property "{val}" was not found in log'
                        )
                        progress_data[key] = len(values) * [[np.nan]]
                results["_progress_loss"].append(
                    self.progress_loss(budget, progress_data)
                )
        if not self.keep_log_files:
            os.remove(log_path)
        return results

    def greedy_portfolio(self, result):
        id2conf = result.get_id2config_mapping()
        best_id = result.get_incumbent_id()
        best_run = result.get_runs_by_id(best_id)[-1]
        all_runs = [
            run for run in result.get_all_runs() if run.budget >= best_run.budget
        ]
        best_individual_losses = self.individual_losses(best_run.budget, best_run.info)
        best_loss = np.mean(best_individual_losses)
        individual_losses = [
            self.individual_losses(run.budget, run.info) for run in all_runs
        ]
        portfolio = [id2conf[best_run.config_id]]
        print(best_loss, best_individual_losses, portfolio[-1])
        for _ in individual_losses:
            old_loss = best_loss
            for run, losses in zip(all_runs, individual_losses):
                candidate_losses = np.minimum(losses, best_individual_losses)
                candidate_loss = np.mean(candidate_losses)
                if candidate_loss < best_loss:
                    best_individual_losses = candidate_losses
                    best_loss = candidate_loss
                    best_run = run
            if old_loss > best_loss:
                portfolio.append(id2conf[best_run.config_id])
                print(best_loss, best_individual_losses, portfolio[-1])
            else:
                break
