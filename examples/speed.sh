#/bin/bash
export PATH=${HOME}/bin:${PATH}
source ${HOME}/ws_hyperplan/devel/setup.bash
source ${HOME}/ws_hyperplan/venv/bin/activate
cd ${HOME}/ws_hyperplan/src/hyperplan

outdir=${HOME}/Bubox/archive/mark_moll/hyperplan/speed
num_workers=${1:-8}
run_id=${2:-0}
worker=${3:-}

for i in `seq ${num_workers}`; do
    (nohup ./scripts/hyperplan_cmdline.py --run_id ${run_id} --n_workers ${num_workers} --shared_directory ${outdir} --n_iterations 7 --min_budget .33333333333 --max_budget 729 --backend omplapp --opt speed ${worker} examples/cubicles.cfg examples/Twistycool.cfg >> speed.log) &
    worker='--worker'
done
