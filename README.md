# Motion Planning Hyperparameter Optimization (MPHPO)

Tool for automatic selection of a motion planning algorithm and its parameters
that optimize some performance metric over a given set of problems.

It uses [HpBandSter](https://github.com/automl/HpBandSter) as the underlying
optimization package.

## Installation

First, build a recent version of OMPL.app somewhere. Next run the following commands:

    python3 -m venv venv
    source venv/bin/activate
    pip install --isolated -r requirements.txt
    export PYTHONPATH=/home/mmoll/omplapp/ompl/py-bindings
