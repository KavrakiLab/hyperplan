/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2021, Rice University
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Rice University nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Author: Mark Moll */

#include <robowflex_library/builder.h>
#include <robowflex_library/detail/fetch.h>
#include <robowflex_library/io/broadcaster.h>
#include <robowflex_library/io/visualization.h>
#include <robowflex_library/planning.h>
#include <robowflex_library/benchmarking.h>
#include <robowflex_library/robot.h>
#include <robowflex_library/scene.h>
#include <robowflex_library/util.h>
#include <robowflex_ompl/ompl_interface.h>
#include <ompl/base/goals/GoalRegion.h>

using namespace robowflex;

static const std::string GROUP = "arm_with_torso";

struct GoalDistanceFunctor
{
    const Benchmarker::BenchmarkRequest &request;

    void operator()(planning_interface::MotionPlanResponse &run, Benchmarker::Results::Run &metrics)
    {
        auto planner = std::dynamic_pointer_cast<const OMPL::OMPLInterfacePlanner>(std::get<1>(request));
        if (planner == nullptr)
            ROS_FATAL("Unexpected planner!");
	auto pdef = planner->getLastSimpleSetup()->getProblemDefinition();
	double distance = pdef->getSolutionDifference();
	if (distance == -1)
	{
	    auto start = pdef->getStartState(0);
	    auto goal = std::dynamic_pointer_cast<ompl::base::GoalRegion>(pdef->getGoal());
	    if (goal == nullptr)
	        ROS_FATAL("Unexpected goal type!");
	    distance = goal->distanceGoal(start);
	}
        metrics.metrics["goal_distance"] = distance;
    }
};

Benchmarker::Results::ComputeMetricCallbackFn
callbackFnAllocator(const Benchmarker::BenchmarkRequest &request)
{
    return GoalDistanceFunctor{request};
}

int main(int argc, char **argv)
{
    ROS ros(argc, argv);
    if (argc < 5)
    {
        ROS_FATAL_STREAM("Command line syntax:\n\t" << argv[0]
                                                    << " scene.yaml request.yaml ompl_planning.yaml time "
                                                       "num_runs log_dir [simplify]");
        exit(-1);
    }
    std::string scene_file_name(argv[1]), request_file_name(argv[2]), planner_config_file_name(argv[3]);
    double planning_time = std::atof(argv[4]);
    bool rviz_only = argc < 7;
    auto robot = std::make_shared<FetchRobot>();
    robot->initialize();

    auto scene = std::make_shared<Scene>(robot);
    if (!scene->fromYAMLFile(scene_file_name))
    {
        ROS_FATAL_STREAM("Failed to read file " << scene_file_name << " for scene");
        exit(-1);
    }
    auto planner = std::make_shared<OMPL::OMPLInterfacePlanner>(robot, "default");
    OMPL::Settings settings;
    settings.simplify_solutions = rviz_only || (argc > 7);
    planner->initialize(planner_config_file_name, settings);
    auto request = std::make_shared<MotionRequestBuilder>(planner, GROUP);
    if (!request->fromYAMLFile(request_file_name))
    {
        ROS_FATAL_STREAM("Failed to read file " << request_file_name << " for request");
        exit(-1);
    }
    request->getRequest().planner_id = "planner";
    request->setAllowedPlanningTime(planning_time);
    request->setNumPlanningAttempts(1);

    if (rviz_only)
    {
        IO::RVIZHelper rviz(robot);
        IO::RobotBroadcaster bc(robot);
        bc.start();

        ROS_INFO("RViz Initialized! Press enter to continue (after your RViz is "
                 "setup)...");
        std::cin.get();
        rviz.updateScene(scene);
        rviz.updateMarkers();
        ROS_INFO("Scene displayed! Press enter to plan...");
        std::cin.get();
        while (true)
        {
            planning_interface::MotionPlanResponse res = planner->plan(scene, request->getRequest());
            if (res.error_code_.val == moveit_msgs::MoveItErrorCodes::SUCCESS)
                // Publish the trajectory to a topic to display in RViz
                rviz.updateTrajectory(res);

            ROS_INFO("Press enter to remove the scene.");
            std::cin.get();
        }
        rviz.removeScene();
    }
    else
    {
        unsigned int num_runs = std::atoi(argv[5]);
        std::string log_dir(argv[6]);
        Benchmarker benchmark;
        benchmark.setMetricCallbackFnAllocator(callbackFnAllocator);
        benchmark.addBenchmarkingRequest(scene_file_name, scene, planner, request);
        auto options = Benchmarker::Options(num_runs, Benchmarker::MetricOptions::LENGTH);
        options.progress_update_rate = .5;  // .5 seconds
        benchmark.benchmark({std::make_shared<OMPLBenchmarkOutputter>(log_dir)}, options);
    }

    return 0;
}
