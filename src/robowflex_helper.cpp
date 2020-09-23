/* Author: Mark Moll */

#include <cstdlib>
#include <robowflex_library/benchmarking.h>
#include <robowflex_library/builder.h>
#include <robowflex_library/detail/fetch.h>
#include <robowflex_library/io/visualization.h>
#include <robowflex_library/scene.h>
#include <robowflex_library/util.h>
#include <robowflex_ompl/ompl_interface.h>

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
        metrics.metrics["goal_distance"] = std::max(0., planner->getLastSimpleSetup()->getProblemDefinition()->getSolutionDifference());
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
                                                       "num_runs log_dir");
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
    settings.simplify_solutions = rviz_only;
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
