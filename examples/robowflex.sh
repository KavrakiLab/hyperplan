#/bin/bash
source ${HOME}/ws_hyperplan/devel/setup.bash
source ${HOME}/ws_hyperplan/venv/bin/activate
cd ${HOME}/ws_hyperplan/src/hyperplan

problem=${1:-box_pick}
opt=${2:-speed}
runid=${3:-1}
worker=${4:-}

(nohup ./scripts/hyperplan_cmdline.py --run_id=${runid} \
${worker} examples/fetch/${problem}-${opt}.yaml >> /tmp/hyperplan-${problem}-${opt}.log) &
