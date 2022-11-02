FROM ros:melodic
ENV CATKIN_WS=/ws_hyperplan
WORKDIR ${CATKIN_WS}/src
COPY . hyperplan

# Replace {token} with a token that has private repo access.
# See https://github.com/settings/tokens for details
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install python-catkin-tools python3.8-dev python3.8-venv && \
    wstool init . && \
    wstool merge hyperplan/hyperplan.rosinstall && \
    wstool update
WORKDIR ${CATKIN_WS}
RUN catkin config --extend /opt/ros/$ROS_DISTRO \
           --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
    rosdep install -y -r --from-paths src --ignore-src --rosdistro $ROS_DISTRO && \
    catkin build
RUN python3.8 -m venv venv && \
    . ./venv/bin/activate && \
    pip install --isolated wheel==0.38.0 setuptools==59.8.0 Cython==0.29.32 numpy==1.19.0 scipy==1.6.3 && \
    pip install --isolated -r src/hyperplan/requirements.txt
ENTRYPOINT ["./src/hyperplan/scripts/hyperplan_docker_entrypoint.sh"]
