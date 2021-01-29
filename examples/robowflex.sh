#/bin/bash
source ${HOME}/ws_hyperplan/devel/setup.bash
source ${HOME}/ws_hyperplan/venv/bin/activate
cd ${HOME}/ws_hyperplan/src/hyperplan

num_workers=${1:-8}
run_id=${2:-0}
opt=${3:-speed}
problem=${4:-small_pick}
#outdir=${HOME}/Bubox/archive/mark_moll/hyperplan/robowflex/${opt}/${problem}
outdir=/tmp/hyperplan/robowflex/${opt}/${problem}
max_budget=${5:-729}
num_iterations=${6:-7}
worker=${7:-}

for i in `seq ${num_workers}`; do
    mkdir -p ${outdir}
    (nohup ./scripts/hyperplan_cmdline.py \
    --run_id ${run_id} \
    --n_workers ${num_workers} \
    --shared_directory ${outdir} \
    --n_iterations ${num_iterations} \
    --max_budget ${max_budget} \
    --backend robowflex \
    --opt ${opt} \
    ${worker} examples/fetch/${problem} >> ${outdir}/hyperplan.log) &
    worker='--worker'
done
