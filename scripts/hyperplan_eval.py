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

import platform
import subprocess
import argparse
import logging
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hyperplan import omplapp, robowflex, default_network_interface, worker_types

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


    parser = argparse.ArgumentParser(description='Motion Planning Hyperparameter Optimization.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--budget', type=float, default=6,
                        help='Budget used for optimization.')
    parser.add_argument('--nic_name', type=str, default=default_network_interface(),
                        help='Which network interface to use for communication.')
    parser.add_argument('--opt', type=str, choices={key[1] for key in worker_types.keys()}, default='speed',
                        help='Type of hyperparameter optimization to perform')
    parser.add_argument('--backend', type=str, choices={key[0] for key in worker_types.keys()}, default='omplapp',
                        help='Backend used for evaluating planner configurations')
    parser.add_argument('--param_id', type=str, default=None,
                        help='Id of the form 1,2,3, corresponding to the hyperparameter config id in the results.pkl file in the working directory')
    parser.add_argument('--working_dir', type=str, required=True,
                        help='A directory that is accessible for all processes, e.g. a NFS share.')
    parser.add_argument('config', nargs='+', type=str,
                        help='configuration file/directory specifying a benchmark problem')

    args = parser.parse_args()
    logged_results = hpres.logged_results_to_HBS_result(args.working_dir)
    id2conf = logged_results.get_id2config_mapping()
    selected_id = logged_results.get_incumbent_id() if args.param_id==None else \
        tuple([int(s) for s in args.param_id.split(',')])
    config = id2conf[selected_id]['config']

    host = hpns.nic_name_to_host(args.nic_name)
    WorkerType = worker_types[(args.backend, args.opt)]
    NS = hpns.NameServer(run_id=0,
                         host=host, port=0,
                         working_directory=args.working_dir)
    ns_host, ns_port = NS.start()
    w = WorkerType(config_files=args.config,
                   run_id=0,
                   host=host,
                   nameserver=ns_host,
                   nameserver_port=ns_port)
    results = w.compute(config_id=None, config=config, budget=args.budget, working_directory=args.working_dir)
    print(f'id={selected_id}, config={config}, input={args.config}')
    print(results)
