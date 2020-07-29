/* Author: Mark Moll */

#include <cstdlib>
#include <robowflex_library/benchmarking.h>
#include <robowflex_library/builder.h>
#include <robowflex_library/detail/fetch.h>
#include <robowflex_library/io/visualization.h>
#include <robowflex_library/scene.h>
#include <robowflex_library/util.h>

using namespace robowflex;

static const std::string GROUP = "arm_with_torso";

int main(int argc, char **argv)
{
    ROS ros(argc, argv);
    if (argc < 5)
    {
        ROS_FATAL_STREAM("Command line syntax:\n\t" << argv[0]
                                                    << " scene.yaml request.yaml ompl_planning.yaml time "
                                                       "num_runs output.log");
        exit(-1);
    }
    bool rviz_only = argc < 7;
    auto fetch = std::make_shared<FetchRobot>();
    fetch->initialize();

    auto scene = std::make_shared<Scene>(fetch);
    if (!scene->fromYAMLFile(argv[1]))
    {
        ROS_FATAL("Failed to read file: %s for scene", argv[1]);
        exit(-1);
    }
    auto planner = std::make_shared<OMPL::FetchOMPLPipelinePlanner>(fetch);
    planner->initialize(OMPL::Settings(), argv[3]);
    auto request = std::make_shared<MotionRequestBuilder>(planner, GROUP);
    if (!request->fromYAMLFile(argv[2]))
    {
        ROS_FATAL("Failed to read file: %s for request", argv[2]);
        exit(-1);
    }
    request->getRequest().planner_id = "planner";
    request->setAllowedPlanningTime(std::atof(argv[4]));
    request->setNumPlanningAttempts(1);

    if (rviz_only)
    {
        IO::RVIZHelper rviz(fetch);
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
            if (res.error_code_.val != moveit_msgs::MoveItErrorCodes::SUCCESS)
                return 1;

            // Publish the trajectory to a topic to display in RViz
            rviz.updateTrajectory(res);

            ROS_INFO("Press enter to remove the scene.");
            std::cin.get();
        }
        rviz.removeScene();
    }
    else
    {
        Benchmarker benchmark;
        benchmark.addBenchmarkingRequest(argv[1], scene, planner, request);
        auto options = Benchmarker::Options(std::atoi(argv[5]));
        benchmark.benchmark({std::make_shared<OMPLBenchmarkOutputter>(argv[6])}, options);
    }
}
