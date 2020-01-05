import os
import argparse
import pickle
import time
import logging
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
from mphpo import SpeedWorker

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Motion Planning Hyperparameter Optimization.')
parser.add_argument('--min_budget', type=float, default=60,
                    help='Minimum budget used during the optimization.')
parser.add_argument('--max_budget', type=float, default=3600,
                    help='Maximum budget used during the optimization.')
parser.add_argument('--n_iterations', type=int, default=4,
                    help='Number of iterations performed by the optimizer')
parser.add_argument('--n_workers', type=int, default=2,
                    help='Number of workers to run in parallel.')
parser.add_argument('--worker', action='store_true',
                    help='Flag to turn this into a worker process')
parser.add_argument('--run_id', type=str,
                    help='A unique id for this optimization run \
                         (e.g., the job id of the cluster\'s scheduler).')
parser.add_argument('--nic_name', type=str, default='enp0s31f6',
                    help='Which network interface to use for communication.')
parser.add_argument('--shared_directory', type=str,
                    default='/home/mmoll/Bubox/archive/mmoll/mark_moll/mphpo/results',
                    help='A directory that is accessible for all processes, e.g. a NFS share.')


args = parser.parse_args()

# Every process has to lookup the hostname
host = hpns.nic_name_to_host(args.nic_name)


if args.worker:
    time.sleep(5)    # short artificial delay to make sure the nameserver is already running
    w = SpeedWorker(run_id=args.run_id, host=host)
    w.load_nameserver_credentials(working_directory=args.shared_directory)
    w.run(background=False)
    exit(0)

result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=False)

# Start a nameserver:
# We now start the nameserver with the host name from above and a random open port
# (by setting the port to 0)
NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
ns_host, ns_port = NS.start()

# Most optimizers are so computationally inexpensive that we can affort to run a
# worker in parallel to it. Note that this one has to run in the background to
# not block!
w = SpeedWorker(run_id=args.run_id,
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port)
w.run(background=True)

# Run an optimizer
# We now have to specify the host, and the nameserver information
bohb = BOHB(configspace=SpeedWorker.get_configspace(),
            run_id=args.run_id,
            host=host,
            nameserver=ns_host,
            nameserver_port=ns_port,
            result_logger=result_logger,
            min_budget=args.min_budget,
            max_budget=args.max_budget
           )
res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

# In a cluster environment, you usually want to store the results for later analysis.
# One option is to simply pickle the Result object
with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
    pickle.dump(res, fh)


# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()
