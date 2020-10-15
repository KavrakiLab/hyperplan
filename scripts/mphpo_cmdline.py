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
import platform
import subprocess
import argparse
import pickle
import time
import logging
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
from mphpo import omplapp, robowflex

def default_network_interface():
    operating_system = platform.system()
    network_interface = 'eth0'
    if operating_system == 'Linux':
        try:
            output = subprocess.run('route | grep \'^default\' | grep -o \'[^ ]*$\'', shell=True, capture_output=True, check=True)
            network_interface = output.stdout.decode().strip()
        except subprocess.CalledProcessError:
            pass
    elif operating_system == 'Darwin':
        try:
            output = subprocess.run('route -n get default | grep \'interface:\' | grep -o \'[^ ]*$\'', shell=True, capture_output=True, check=True)
            network_interface = output.stdout.decode().strip()
        except subprocess.CalledProcessError:
            pass
    return network_interface


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    worker_types = {
        ('omplapp', 'speed'): omplapp.SpeedWorker,
        ('omplapp', 'speed_kinodynamic'): omplapp.SpeedKinodynamicWorker,
        ('omplapp', 'opt') : omplapp.OptWorker,
        ('robowflex', 'speed'): robowflex.SpeedWorker
    }

    parser = argparse.ArgumentParser(description='Motion Planning Hyperparameter Optimization.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--min_budget', type=float, default=6,
                        help='Minimum budget used during the optimization.')
    parser.add_argument('--max_budget', type=float, default=3600,
                        help='Maximum budget used during the optimization.')
    parser.add_argument('--n_iterations', type=int, default=4,
                        help='Number of iterations performed by the optimizer')
    parser.add_argument('--n_workers', type=int, default=2,
                        help='Number of workers to run in parallel.')
    parser.add_argument('--random_fraction', type=float, default=1/3,
                        help='fraction of purely random configurations that are sampled from the '
                             'prior without the model')
    parser.add_argument('--worker', action='store_true',
                        help='Flag to turn this into a worker process')
    parser.add_argument('--run_id', type=str, default="0",
                        help='A unique id for this optimization run \
                             (e.g., the job id of the cluster\'s scheduler).')
    parser.add_argument('--nic_name', type=str, default=default_network_interface(),
                        help='Which network interface to use for communication.')
    parser.add_argument('--shared_directory', type=str,
                        default=os.environ['HOME'] + '/Bubox/archive/mmoll/mark_moll/mphpo/results',
                        help='A directory that is accessible for all processes, e.g. a NFS share.')
    parser.add_argument('--opt', type=str, choices={key[1] for key in worker_types.keys()}, default='speed',
                        help='Type of hyperparameter optimization to perform')
    parser.add_argument('--backend', type=str, choices={key[0] for key in worker_types.keys()}, default='omplapp',
                        help='Backend used for evaluating planner configurations')
    parser.add_argument('config', nargs='+', type=str,
                        help='configuration file/directory specifying a benchmark problem')

    args = parser.parse_args()
    working_dir = f'{args.shared_directory}/{args.run_id}'

    # Every process has to lookup the hostname
    host = hpns.nic_name_to_host(args.nic_name)

    WorkerType = worker_types[(args.backend, args.opt)]

    if args.worker:
        time.sleep(5)    # short artificial delay to make sure the nameserver is already running
        w = WorkerType(config_files=args.config, run_id=args.run_id, host=host)
        w.load_nameserver_credentials(working_directory=working_dir)
        w.run(background=False)
        exit(0)

    result_logger = hpres.json_result_logger(directory=working_dir, overwrite=False)

    # Start a nameserver:
    # We now start the nameserver with the host name from above and a random open port
    # (by setting the port to 0)
    NS = hpns.NameServer(run_id=args.run_id,
                         host=host, port=0,
                         working_directory=working_dir)
    ns_host, ns_port = NS.start()

    # Most optimizers are so computationally inexpensive that we can afford to run a
    # worker in parallel to it. Note that this one has to run in the background to
    # not block!
    w = WorkerType(config_files=args.config,
                   run_id=args.run_id,
                   host=host,
                   nameserver=ns_host,
                   nameserver_port=ns_port)
    w.run(background=True)

    # Run an optimizer
    bohb = BOHB(configspace=WorkerType.get_configspace(),
                run_id=args.run_id,
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port,
                result_logger=result_logger,
                min_budget=args.min_budget,
                max_budget=args.max_budget,
                random_fraction=args.random_fraction
               )
    res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

    with open(os.path.join(working_dir, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
