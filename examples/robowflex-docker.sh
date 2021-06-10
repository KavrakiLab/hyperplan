#/bin/bash
robot=${1:-fetch}
shift
problem=${1:-box_pick}
shift
opt=${1:-speed}
shift
testing=${1:-}
shift
name=${problem}-${opt}
if [ "$testing" == "--test" ]; then
    suffix="-test"
fi

docker run \
   --name ${name}${suffix} \
   --mount type=bind,source=${HOME}/Bubox/archive/mark_moll,target=/ws_hyperplan/data \
   hyperplan ${testing} $@ \
   /ws_hyperplan/data/hyperplan-${robot}/${problem}-${opt}.yaml >> /tmp/hyperplan-${robot}-${problem}-${opt}${suffix}.log
