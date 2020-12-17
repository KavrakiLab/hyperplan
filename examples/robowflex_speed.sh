#/bin/bash
source ${HOME}/ws_hyperplan/devel/setup.bash
source ${HOME}/ws_hyperplan/venv/bin/activate
cd ${HOME}/ws_hyperplan/src/hyperplan

outdir=${HOME}/Bubox/archive/mark_moll/hyperplan/robowflex_speed
num_workers=${1:-8}
run_id=${2:-0}
worker=${3:-}

for i in `seq ${num_workers}`; do
    (nohup ./scripts/hyperplan_cmdline.py --run_id ${run_id} --n_workers ${num_workers} --shared_directory ${outdir} --n_iterations 7 --min_budget 3 --max_budget 2187 --backend robowflex --opt speed ${worker} examples/fetch/vicon >> robowflex_speed.log) &
    worker='--worker'
done
