#/bin/bash
source ${HOME}/ws_hyperplan/devel/setup.bash
source ${HOME}/ws_hyperplan/venv/bin/activate
cd ${HOME}/ws_hyperplan/src/hyperplan/examples/fetch

for f in *.yaml; do
    echo $f
    ../../scripts/hyperplan_eval.py $f
done
