#/bin/bash
robot=${1:-fetch}
problem=${2:-box_pick}
opt=${3:-speed}
testing=${4:-}
name=${problem}-${opt}
if [ "$testing" == "--test" ]; then
    suffix="-test"
fi

docker run \
   --name ${name}${suffix} \
   --mount type=bind,source=${HOME}/Bubox/archive/mark_moll,target=/ws_hyperplan/data \
   hyperplan ${testing} \
   /ws_hyperplan/data/hyperplan-${robot}/${problem}-${opt}.yaml >> /tmp/hyperplan-${problem}-${opt}.log
