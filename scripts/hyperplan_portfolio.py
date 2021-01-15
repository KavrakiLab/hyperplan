#!/usr/bin/env python

import sys
import argparse
import logging
import numpy as np
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hyperplan import default_network_interface, worker_types

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Motion Planning Hyperparameter Optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nic_name",
        type=str,
        default=default_network_interface(),
        help="Which network interface to use for communication.",
    )
    parser.add_argument(
        "--opt",
        type=str,
        choices={key[1] for key in worker_types.keys()},
        default="speed",
        help="Type of hyperparameter optimization to perform",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices={key[0] for key in worker_types.keys()},
        default="omplapp",
        help="Backend used for evaluating planner configurations",
    )
    parser.add_argument(
        "result_dir",
        nargs="+",
        type=str,
        help="directory with results from running hyperplan_cmdline",
    )

    args = parser.parse_args()

    host = hpns.nic_name_to_host(args.nic_name)
    WorkerType = worker_types[(args.backend, args.opt)]
    NS = hpns.NameServer(run_id=0, host=host)
    ns_host, ns_port = NS.start()
    worker = WorkerType(
        config_files=None,
        run_id=0,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
    )
    for path in args.result_dir:
        print(path)
        worker.greedy_portfolio(hpres.logged_results_to_HBS_result(path))
    NS.shutdown()
