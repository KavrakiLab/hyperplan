######################################################################
# Software License Agreement (BSD License)
#
#  Copyright (c) 2020, Rice University
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   * Neither the name of Rice University nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
######################################################################

# Author: Mark Moll

import platform
import subprocess
import numpy as np

def quantile_with_fallback(x, fallback, q=.7):
    return np.quantile(x, q) if len(x) > 0 else fallback
def nanquantile_with_fallback(x, fallback, q=.7):
    return np.nanquantile(x, q) if len(x) > 0 else fallback

def default_network_interface():
    operating_system = platform.system()
    network_interface = 'eth0'
    if operating_system == 'Linux':
        try:
            output = subprocess.run(
                'route | grep \'^default\' | grep -v wlx | grep -o \'[^ ]*$\'',
                shell=True, capture_output=True, check=True)
            network_interface = output.stdout.decode().strip()
        except subprocess.CalledProcessError:
            pass
    elif operating_system == 'Darwin':
        try:
            output = subprocess.run(
                'route -n get default | grep \'interface:\' | grep -o \'[^ ]*$\'',
                shell=True, capture_output=True, check=True)
            network_interface = output.stdout.decode().strip()
        except subprocess.CalledProcessError:
            pass
    return network_interface

def csv_dump(result, path):
    all_runs = result.get_all_runs()
    id2conf = result.get_id2config_mapping()
    with open(path, "w") as csvfile:
        print("id,budget,loss,planner,model_based", file=csvfile)
        for run in all_runs:
            config = id2conf[run.config_id]
            planner = config["config"]["planner"]
            model_based = int(config["config_info"]["model_based_pick"])
            print(
                f"\"{run.config_id}\",{run.budget},{run.loss},{planner},{model_based}",
                file=csvfile,
            )
