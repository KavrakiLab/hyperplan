#/bin/bash
source ${HOME}/ws_mphpo/devel/setup.bash
source ${HOME}/ws_mphpo/venv/bin/activate
cd ${HOME}/ws_mphpo/src/mphpo

outdir=${HOME}/Bubox/archive/mark_moll/mphpo/speed_kinodynamic
num_workers=8
run_id=${1:-0}
worker=

for i in `seq ${num_workers}`; do
    (nohup ./scripts/mphpo_cmdline.py --run_id ${run_id} --n_workers ${num_workers} --shared_directory ${outdir} --n_iterations 7 --min_budget .33333333333 --max_budget 729 --backend omplapp --opt speed_kinodynamic ${worker} examples/Maze_kcar.cfg examples/narrow_passage_kcar.cfg >> speed_kinoddynamic.log) &
    worker='--worker'
done
