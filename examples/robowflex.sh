#/bin/bash
source ${HOME}/ws_hyperplan/devel/setup.bash
source ${HOME}/ws_hyperplan/venv/bin/activate
cd ${HOME}/ws_hyperplan/src/hyperplan

problem=${1:-box_pick}
opt=${2:-speed}
worker=${3:-}

(nohup ./scripts/hyperplan_cmdline.py \
${worker} examples/fetch/${problem}-${opt}.yaml >> /tmp/hyperplan-${problem}-${opt}.log) &
