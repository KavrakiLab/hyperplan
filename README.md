# Motion Planning Hyperparameter Optimization (MPHPO)

Tool for automatic selection of a motion planning algorithm and its parameters
that optimize some performance metric over a given set of problems.

It uses [HpBandSter](https://github.com/automl/HpBandSter) as the underlying
optimization package.

## Installation

Run the following commands:

    sudo apt-get install python3.7-dev python3.7-venv
    python3.7 -m venv venv
    source venv/bin/activate
    pip install --isolated -r requirements.txt

### OMPL.app benchmarking

Build a recent version of [OMPL.app](http://ompl.kavrakilab.org)
somewhere. Make sure that `ompl_benchmark` is somewhere in the $PATH.

### MoveIt benchmarking

TODO

### Robowflex benchmarking

1. [Install ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) and these build tools:

         sudo apt-get install python-wstool python-catkin-tools

2. Check out the code:

         export CATKIN_WS=~/ws_mphpo
         mkdir -p $CATKIN_WS/src
         cd $CATKIN_WS/src
         wstool init .
         git clone git@github.com:KavrakiLab/mphpo.git
         wstool merge $CATKIN_WS/src/mphpo/mphpo.rosinstall
         wstool update

3. Configure and build the code:

         cd $CATKIN_WS
         catkin config --extend /opt/ros/$ROS_DISTRO \
           --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
         rosdep install -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO
         catkin build

4. Source the workspace.

         source $CATKIN_WS/devel/setup.bash

## Running the optimization

Below are some examples on how to run the command line tool to optimize for different objectives using different backends.

### OMPL.app benchmarking

Find planner configuration that optimizes speed of geometric planning:

    ./scripts/mphpo_cmdline.py --n_workers 10 --shared_directory /some/network/drive examples/cubicles.cfg
    # repeat 9 times on worker nodes:
    ./scripts/mphpo_cmdline.py --worker --n_workers 10 --shared_directory /some/network/drive examples/cubicles.cfg

Find planner configuration that optimizes speed of kinodynamic planning:

    ./scripts/mphpo_cmdline.py --n_workers 10 --shared_directory /some/network/drive --opt speed_kinodynamic examples/Maze_kcar.cfg
    # repeat 9 times on worker nodes:
    ./scripts/mphpo_cmdline.py --worker --n_workers 10 --shared_directory /some/network/drive --opt speed_kinodynamic examples/Maze_kcar.cfg

Find planner configuration that optimizes convergence rate of asymptotically (near-)optimal geometric planning:

    ./scripts/mphpo_cmdline.py --n_workers 10 --shared_directory /some/network/drive --opt opt examples/cubicles.cfg
    # repeat 9 times on worker nodes:
    ./scripts/mphpo_cmdline.py --worker --n_workers 10 --shared_directory /some/network/drive --opt opt examples/cubicles.cfg

Type `./scripts/mphpo_cmdline.py --help` to see all options. See the shared directory for the result files. See the script `./scripts/analysis.py` for an example of how to perform some basic analysis of the results.

### MoveIt

TODO

### Robowflex benchmarking

Find planner configuration that optimizes speed of geometric planning for the Fetch arm and torso using 10 scenes and corresponding motion planning queries:

    # run the following line for every compute node where you have more than 1 worker running
    if [ `rostopic list |wc -l` == 0 ]; then roscore; fi &
    ./scripts/mphpo_cmdline.py --n_workers 10 --shared_directory /some/network/drive --backend robowflex examples/fetch &
    # repeat 9 times on worker nodes:
    ./scripts/mphpo_cmdline.py --worker --n_workers 10 --shared_directory /some/network/drive --backend robowflex examples/fetch

Note the format in the example directory:

- There is an equal number of scene and motion planning request YAML files. The scenes are all the files that match the glob pattern `*scene*.yaml`, while the requests are the files that `*request*.yaml`. The aggregate performance across all the planning problems is optimized.
- The `ompl_planning.yaml` file is a minimal template of the `ompl_planning.yaml` file that is normally found in the `<robot>_moveit_config` package for a robot. The only things you'd have to change for a different robot are the planning group and the projection.
