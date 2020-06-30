/* Author: Mark Moll */

#include <cstdlib>
#include <robowflex_library/util.h>
#include <robowflex_library/scene.h>
#include <robowflex_library/builder.h>
#include <robowflex_library/benchmarking.h>
#include <robowflex_library/detail/fetch.h>

using namespace robowflex;

static const std::string GROUP = "arm_with_torso";

int main(int argc, char **argv)
{
    ROS ros(argc, argv);
    if (argc < 6)
    {
        ROS_FATAL_STREAM("Command line syntax:\n\t" << argv[0]
            << " scene.yaml request.yaml ompl_planning.yaml num_runs time output.log");
        exit(-1);
    }
    auto fetch = std::make_shared<FetchRobot>();
    fetch->initialize();
    Benchmarker benchmark;
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
    request->setAllowedPlanningTime(std::atof(argv[5]));
    request->setNumPlanningAttempts(1);
    benchmark.addBenchmarkingRequest(argv[1], scene, planner, request);
    auto options = Benchmarker::Options(std::atoi(argv[4]));
    benchmark.benchmark({std::make_shared<OMPLBenchmarkOutputter>(argv[6])}, options);
}
