# Motion Planning Hyperparameter Optimization (MPHPO)

Tool for automatic selection of a motion planning algorithm and its parameters
that optimize some performance metric over a given set of problems.

It uses [HpBandSter](https://github.com/automl/HpBandSter) as the underlying
optimization package.

## Installation

First, build a recent version of [OMPL.app](http://ompl.kavrakilab.org)
somewhere. Make sure that `ompl_benchmark` is somewhere in the $PATH. 
Next run the following commands:

    python3 -m venv venv
    source venv/bin/activate
    pip install --isolated -r requirements.txt

## Running the optimization

Find planner configuration that optimizes speed of geometric planning:

    ./local_cluster.py --nic_name=en0 --n_workers 10 --shared_directory /some/network/drive examples/cubicles.cfg
    # repeat 9 times on worker nodes:
    ./local_cluster.py --worker --nic_name=en0 --n_workers 10 --shared_directory /some/network/drive examples/cubicles.cfg

Find planner configuration that optimizes speed of kinodynamic planning:

    ./local_cluster.py --nic_name=en0 --n_workers 10 --shared_directory /some/network/drive --type speed_kinodynamic examples/Maze_kcar.cfg
    # repeat 9 times on worker nodes:
    ./local_cluster.py --worker --nic_name=en0 --n_workers 10 --shared_directory /some/network/drive --type speed_kinodynamic examples/Maze_kcar.cfg

Find planner configuration that optimizes convergence rate of asymptotically (near-)optimal geometric planning:

    ./local_cluster.py --nic_name=en0 --n_workers 10 --shared_directory /some/network/drive --type opt examples/cubicles.cfg
    # repeat 9 times on worker nodes:
    ./local_cluster.py --worker --nic_name=en0 --n_workers 10 --shared_directory /some/network/drive --type  opt examples/cubicles.cfg

Type `./local_cluster.py --help` to see all options.
