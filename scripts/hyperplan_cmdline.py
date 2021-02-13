#!/usr/bin/env python3

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
from pathlib import Path
import argparse
import pickle
import time
import logging
import yaml
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
from hyperplan import default_network_interface, csv_dump, worker_types

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Motion Planning Hyperparameter Optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_workers", type=int, default=1, help="Number of workers to run in parallel."
    )
    parser.add_argument(
        "--worker", action="store_true", help="Flag to turn this into a worker process"
    )
    parser.add_argument(
        "--run_id", type=int, default=-1, help="Run id. Use -1 to auto select next run id."
    )
    parser.add_argument(
        "--nic_name",
        type=str,
        default=default_network_interface(),
        help="Which network interface to use for communication.",
    )
    parser.add_argument(
        "config",
        type=argparse.FileType("r"),
        help="YAML configuration file specifying a hyperparameter optimization problem",
    )

    args = parser.parse_args()
    config = yaml.load(args.config, Loader=yaml.FullLoader)
    working_dir = (
        Path(config["output_dir"]).resolve()
        / Path(config["input_dir"]).name
        / config["loss_function"]
    )

    run_id = args.run_id
    if run_id == -1:
        runs = sorted(working_dir.glob("*"))
        run_id = 0 if not runs else (int(runs[-1].name) + 1)
    working_dir = working_dir / str(run_id)

    # Every process has to lookup the hostname
    host = hpns.nic_name_to_host(args.nic_name)

    WorkerType = worker_types[(config["backend"], config["loss_function"])]

    if args.worker:
        time.sleep(
            5
        )  # short artificial delay to make sure the nameserver is already running
        w = WorkerType(config=config, run_id=run_id, host=host)
        w.load_nameserver_credentials(working_directory=working_dir)
        w.run(background=False)
        exit(0)

    result_logger = hpres.json_result_logger(directory=working_dir, overwrite=False)

    # Start a nameserver:
    # We now start the nameserver with the host name from above and a random open port
    # (by setting the port to 0)
    NS = hpns.NameServer(
        run_id=run_id, host=host, port=0, working_directory=working_dir
    )
    ns_host, ns_port = NS.start()

    # Most optimizers are so computationally inexpensive that we can afford to run a
    # worker in parallel to it. Note that this one has to run in the background to
    # not block!
    w = WorkerType(
        config=config,
        run_id=run_id,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
    )
    w.run(background=True)

    # Run an optimizer
    bohb = BOHB(
        configspace=WorkerType.get_configspace(),
        run_id=run_id,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        result_logger=result_logger,
        min_budget=config["max_budget"] * 3.0 ** -config["num_iterations"],
        max_budget=config["max_budget"],
        random_fraction=config["random_fraction"],
    )
    print(
        config,
        config["max_budget"] * 3.0 ** -config["num_iterations"],
    )
    res = bohb.run(n_iterations=config["num_iterations"], min_n_workers=args.n_workers)

    with open(Path(working_dir) / "results.pkl", "wb") as fh:
        pickle.dump(res, fh)

    csv_dump(res, Path(working_dir) / "results.csv")

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
