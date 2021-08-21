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
#include <robowflex_library/io.h>
#include <robowflex_library/io/visualization.h>
#include <robowflex_library/log.h>
#include <robowflex_library/planning.h>
#include <robowflex_library/benchmarking.h>
#include <robowflex_library/robot.h>
#include <robowflex_library/scene.h>
#include <robowflex_library/util.h>
#include <robowflex_ompl/ompl_interface.h>
#include <ompl/base/goals/GoalRegion.h>
#include <yaml-cpp/yaml.h>

using namespace robowflex;

using pallet_row_spec = std::tuple<unsigned int, unsigned int, double>;

namespace YAML
{
    template <>
    struct convert<pallet_row_spec>
    {
        static Node encode(const pallet_row_spec &rhs)
        {
            Node node;
            node.push_back(std::get<0>(rhs));
            node.push_back(std::get<1>(rhs));
            node.push_back(std::get<2>(rhs));
            return node;
        }

        static bool decode(const Node &node, pallet_row_spec &rhs)
        {
            if (!node.IsSequence() || node.size() != 3)
            {
                return false;
            }
            rhs =
                pallet_row_spec{node[0].as<unsigned int>(), node[1].as<unsigned int>(), node[2].as<double>()};
            return true;
        }
    };
}  // namespace YAML

class Manipulator : public Robot
{
public:
    Manipulator() : Robot("manipulator")
    {
    }

    bool initialize(const YAML::Node &robot_type)
    {
        setSRDFPostProcessAddFloatingJoint("base_joint");
        bool success = false;
        success = Robot::initialize(
            robot_type["urdf"].as<std::string>(), robot_type["srdf"].as<std::string>(),
            robot_type["joint_limits"].as<std::string>(), robot_type["kinematics"].as<std::string>());

        loadKinematics(robot_type["joint_group"]["name"].as<std::string>());

        return success;
    }
};

ScenePtr create_scene(const RobotConstPtr &robot, const std::vector<pallet_row_spec> &pallet_spec,
                      int pick_percentage)
{
    auto scene = std::make_shared<Scene>(robot);
    moveit_msgs::PlanningScene planning_scene;
    moveit_msgs::CollisionObject boxes;
    boxes.header.frame_id = "world";
    boxes.id = "boxes";
    double height = 0.;
    unsigned int total_num_boxes = 0, num_boxes, box_id = 0;
    geometry_msgs::Pose pose;
    pose.orientation.w = 1.0;
    shape_msgs::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);

    planning_scene.is_diff = true;

    for (const auto &layer : pallet_spec)
        total_num_boxes += std::get<0>(layer) * std::get<1>(layer);
    num_boxes = pick_percentage * total_num_boxes / 100;

    for (const auto &layer : pallet_spec)
    {
        auto rows = std::get<0>(layer), cols = std::get<1>(layer);
        std::vector<double> dims{1. / (double)rows, 1. / (double)cols, std::get<2>(layer)};
        for (unsigned int i = 0; i < 3; ++i)
            primitive.dimensions[i] = .99 * dims[i];
        pose.position.z = height;
        for (unsigned int row = 0; row < rows; ++row)
            for (unsigned int col = 0; col < cols && box_id < num_boxes; ++col, ++box_id)
            {
                pose.position.x = row * dims[0];
                pose.position.y = col * dims[1];
                boxes.primitives.push_back(primitive);
                boxes.primitive_poses.push_back(pose);
            }

        height += std::get<2>(layer);
    }
    planning_scene.world.collision_objects.push_back(boxes);
    scene->useMessage(planning_scene, true);

    return scene;
}

int main(int argc, char **argv)
{
    ROS ros(argc, argv);
    if (argc < 2)
    {
        ROS_FATAL_STREAM("Command line syntax:\n\t" << argv[0] << " config.yaml");
        exit(-1);
    }

    YAML::Node config = YAML::LoadFile(argv[1]);
    auto robot_type = config["robot_type"];
    auto tool_offset = config["tool_offset"].as<double>();
    auto pallet = config["pallet"].as<std::vector<pallet_row_spec>>();
    auto pallet_pick_percentage = config["pallet_pick_percentage"].as<int>();
    auto robot_pose = config["robot_pose"].as<std::vector<double>>();  // x,y,z,rotx,roty,rotz
    auto conveyor_pose = config["conveyor_pose"].as<std::vector<double>>();
    auto duration = config["duration"].as<double>();
    auto num_runs = config["num_runs"].as<unsigned int>();
    auto ompl_config = robot_type["ompl_planning"].as<std::string>();
    bool simplify = config["simplify"] ? config["simplify"].as<bool>() : false;

    // create robot model
    auto robot = std::make_shared<Manipulator>();
    robot->initialize(robot_type);
    robot->getScratchState()->setJointPositions(
        "base_joint", TF::createPoseXYZ(robot_pose[0], robot_pose[1], robot_pose[2], robot_pose[3],
                                        robot_pose[4], robot_pose[5]));
    robot->getScratchState()->update();

    // create scene
    auto scene = create_scene(robot, pallet, pallet_pick_percentage);

    // create planner
    auto planner = std::make_shared<OMPL::OMPLInterfacePlanner>(robot);
    OMPL::Settings settings;
    settings.simplify_solutions = config["rviz"] || simplify;
    planner->initialize(ompl_config, settings);

    // create motion planning request
    auto request = std::make_shared<MotionRequestBuilder>(
        planner, robot_type["joint_group"]["name"].as<std::string>(), "planner");
    request->setAllowedPlanningTime(duration);
    request->setNumPlanningAttempts(1);

    if (settings.simplify_solutions)
        ROS_INFO("Path simplification is enabled");
    else
        ROS_INFO("Path simplification is disabled");

    // show different solutions in RViz
    if (config["rviz"])
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
            if (res.error_code_.val == moveit_msgs::MoveItErrorCodes::SUCCESS)
                // Publish the trajectory to a topic to display in RViz
                rviz.updateTrajectory(res);

            ROS_INFO("Press enter to remove the scene.");
            std::cin.get();
        }
        rviz.removeScene();
    }
    // Profiler::Options options;
    // options.metrics = Profiler::LENGTH;
    // options.progress_update_rate = .5;  // .5 seconds
    // Experiment experiment("robowflex_worker", options, planning_time, num_runs);

    // experiment.getProfiler().addMetricCallback("goal_distance", goal_distance);
    // experiment.addQuery(scene_file_name, scene, planner, request->getRequest());
    // auto dataset = experiment.benchmark(1);
    // OMPLPlanDataSetOutputter output(log_dir);
    // output.dump(*dataset);
}
