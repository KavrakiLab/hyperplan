#!/bin/bash
source /opt/ros/${ROS_DISTRO}/setup.bash
source ${CATKIN_WS}/devel/setup.bash
source ${CATKIN_WS}/venv/bin/activate
exec ${CATKIN_WS}/src/hyperplan/scripts/hyperplan_cmdline.py "$@"
