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
#include <robowflex_library/trajectory.h>
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

class PickPlaceTask
{
public:
    PickPlaceTask(const YAML::Node &config)
    {
        auto robot_type = config["robot_type"];
        auto tool_offset = config["tool_offset"].as<double>();
        auto pallet = config["pallet"].as<std::vector<pallet_row_spec>>();
        auto pallet_pick_percentage = config["pallet_pick_percentage"].as<int>();
        auto robot_pose = config["robot_pose"].as<std::vector<double>>();  // x,y,z,rotx,roty,rotz
        auto conveyor_dimensions = config["conveyor_dimensions"].as<std::vector<double>>();
        auto conveyor_pose = config["conveyor_pose"].as<std::vector<double>>();
        auto ompl_config = robot_type["ompl_planning"].as<std::string>();
        bool simplify = config["simplify"] ? config["simplify"].as<bool>() : false;

        duration = config["duration"].as<double>();
        num_runs = config["num_runs"].as<unsigned int>();
        use_rviz = config["rviz"] ? config["rviz"] : false;

        // create robot model
        robot = std::make_shared<Manipulator>();
        auto &state = robot->getScratchState();
        auto tf = TF::createPoseXYZ(robot_pose[0], robot_pose[1], robot_pose[2], robot_pose[3], robot_pose[4],
                                    robot_pose[5]);

        robot->initialize(robot_type);
        state->setJointPositions("base_joint", tf);
        state->update();

        // create scene
        createScene(pallet, pallet_pick_percentage, conveyor_dimensions,
                    TF::createPoseXYZ(conveyor_pose[0], conveyor_pose[1], conveyor_pose[2], conveyor_pose[3],
                                      conveyor_pose[4], conveyor_pose[5]));
        scene->getScene()->setCurrentState(*state);

        // compute pick/place poses
        double box_height = std::get<2>(pallet.back());
        group = robot_type["joint_group"]["name"].as<std::string>();
        computePoses(box_height, tool_offset, conveyor_pose[2], group);

        // create planner
        planner = std::make_shared<OMPL::OMPLInterfacePlanner>(robot);
        OMPL::Settings settings;
        settings.simplify_solutions = use_rviz || simplify;
        planner->initialize(ompl_config, settings);

        if (settings.simplify_solutions)
            ROS_INFO("Path simplification is enabled");
        else
            ROS_INFO("Path simplification is disabled");
        
        if (use_rviz)
            rvizInitialization();
    }

    void runBenchmark()
    {
        unsigned int max_num_runs = num_runs > 0 ? num_runs : std::numeric_limits<unsigned int>::max();
        for (unsigned int run = 0; run < max_num_runs; ++run)
        {
            computePlan();
            if (num_runs == 0 && duration <= 0.)
                break;
        }
    }

protected:
    void rvizInitialization()
    {
        rviz = std::make_shared<IO::RVIZHelper>(robot);
        rviz->addTransformMarker("pick", "map", pick_pose, 2);
        rviz->addTransformMarker("place", "map", place_pose, 2);

        ROS_INFO("RViz Initialized! Press enter to continue (after your RViz is setup)...");
        std::cin.get();
        rviz->updateScene(scene);
        rviz->updateMarkers();
    }

    void rvizTrajectoryUpdate(const ScenePtr &scene, const planning_interface::MotionPlanResponse &response)
    {
        if (use_rviz)
        {
            rviz->updateScene(scene);
            rviz->updateTrajectory(response);
            ROS_INFO("Press enter to remove the scene.");
            std::cin.get();
        }
    }

    void createScene(const std::vector<pallet_row_spec> &pallet_spec, int pick_percentage,
                     const std::vector<double> &conveyor_dimensions, const RobotPose &conveyor_pose)
    {
        scene = std::make_shared<Scene>(robot);
        moveit_msgs::PlanningScene planning_scene;
        moveit_msgs::CollisionObject boxes;
        boxes.header.frame_id = "world";
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
        num_boxes = std::max(1, (int)std::nearbyint(pick_percentage * total_num_boxes / 100.));
        boxes.id = num_boxes > 1 ? "boxes" : "box";

        for (const auto &layer : pallet_spec)
        {
            auto rows = std::get<0>(layer), cols = std::get<1>(layer);
            std::vector<double> dims{1. / (double)rows, 1. / (double)cols, std::get<2>(layer)};
            for (unsigned int i = 0; i < 3; ++i)
                primitive.dimensions[i] = .99 * dims[i];
            pose.position.z = height + .5 * dims[2];
            for (unsigned int row = 0; row < rows; ++row)
                for (unsigned int col = 0; col < cols && box_id < num_boxes; ++col, ++box_id)
                {
                    pose.position.x = row * dims[0] + .5 * dims[0];
                    pose.position.y = col * dims[1] + .5 * dims[1];
                    boxes.primitives.push_back(primitive);
                    boxes.primitive_poses.push_back(pose);
                    // create a separate collision object for last box, which we will need to pick
                    if (box_id == num_boxes - 2)
                    {
                        planning_scene.world.collision_objects.push_back(boxes);
                        boxes.primitives.clear();
                        boxes.primitive_poses.clear();
                        boxes.id = "box";
                    }
                }

            height += std::get<2>(layer);
        }
        // add block that needs to be picked
        planning_scene.world.collision_objects.push_back(boxes);

        // add conveyor belt
        boxes.primitives.clear();
        boxes.primitive_poses.clear();
        boxes.id = "conveyor belt";
        primitive.dimensions = conveyor_dimensions;
        pose = TF::poseEigenToMsg(conveyor_pose);
        boxes.primitives.push_back(primitive);
        boxes.primitive_poses.push_back(pose);
        planning_scene.world.collision_objects.push_back(boxes);

        scene->useMessage(planning_scene, true);

        // give each collision object a different color
        std_msgs::ColorRGBA color;
        color.r = color.g = color.b = .2;
        color.a = 1.;
        if (num_boxes > 1)
            scene->getScene()->setObjectColor("boxes", color);
        color.r = 1;
        scene->getScene()->setObjectColor("box", color);
        color.r = .2;
        color.b = 1.;
        scene->getScene()->setObjectColor("conveyor belt", color);
    }

    void computePoses(double box_height, double tool_offset, double conveyor_height, const std::string &group)
    {
        pick_pose = scene->getScene()->getFrameTransform("box");
        place_pose = scene->getScene()->getFrameTransform("conveyor belt");
        Eigen::Matrix3d upside_down(Eigen::AngleAxisd(constants::pi, Eigen::Vector3d::UnitX()));

        std::cout << pick_pose.translation().z() << ' ' << place_pose.translation().z() << ' ' << box_height
                  << ' ' << tool_offset << std::endl;
        pick_pose.translation().z() += .5 * box_height + tool_offset + 1e-4;
        pick_pose.linear() = upside_down;
        place_pose.translation().z() += conveyor_height + box_height + tool_offset + 1e-3;
        place_pose.linear() = upside_down;
    }

    double goalDistance(const planning_interface::MotionPlanRequest &request)
    {
        const auto &pdef = planner->getLastSimpleSetup()->getProblemDefinition();
        double distance = pdef->getSolutionDifference();
        if (distance == -1)
        {
            const auto &start = pdef->getStartState(0);
            const auto &goal = std::dynamic_pointer_cast<ompl::base::GoalRegion>(pdef->getGoal());
            if (goal == nullptr)
                ROS_FATAL("Unexpected goal type!");
            distance = goal->distanceGoal(start);
        }
        return distance;
    }

    void updateCosts(const robot_trajectory::RobotTrajectoryPtr &traj, double &path_length,
                     double &trajectory_duration)
    {
        Trajectory trajectory(traj);
        if (trajectory.computeTimeParameterization())
            trajectory_duration += trajectory.getTrajectoryConst()->getDuration();
        else
            trajectory_duration = std::numeric_limits<double>::max();
        path_length += trajectory.getLength();
    }

    void computePlan()
    {
        auto scene = this->scene->deepCopy();
        double path_length{0.}, trajectory_duration{0.}, distance_to_goal{0.}, original_budget{duration};
        std::string tip{robot->getSolverTipFrames(group)[0]};
        std::string base{robot->getSolverBaseFrame(group)};
        MotionRequestBuilder request(planner, group, "planner");
        auto home = scene->getCurrentState();
        auto setGoal = [this, &request](RobotPose &eef_pose) {
            auto &state = robot->getScratchState();
            auto pose = Robot::IKQuery(group, eef_pose, 1e-4, {1e-4, 1e-4, constants::pi});
            if (!robot->setFromIK(pose, *state))
            {
                ROS_ERROR("Could not find IK solution for pose!");
                return false;
            }
            request.setGoalConfiguration(state);
            return true;
        };
        // auto setGoal = [&request, &tip, &base](RobotPose &eef_pose) {
        //     auto copy = eef_pose;
        //     Eigen::Quaterniond orientation(copy.rotation());
        //     copy.linear() = Eigen::Matrix3d::Identity();
        //     request.setGoalRegion(tip, base, copy, Geometry::makeSphere(1e-3), orientation, {1e-3, 1e-3, constants::pi});
        //     return true;
        // };

        request.setAllowedPlanningTime(duration);
        request.setNumPlanningAttempts(1);
        request.setStartConfiguration(home);
        if (!setGoal(pick_pose))
            return;

        // move from home to pick
        auto res = planner->plan(scene, request.getRequest());
        if (res.error_code_.val == moveit_msgs::MoveItErrorCodes::SUCCESS)
        {
            rvizTrajectoryUpdate(scene, res);
            updateCosts(res.trajectory_, path_length, trajectory_duration);
            duration -= res.planning_time_;

            // move from pick to place
            request.setAllowedPlanningTime(duration);
            scene->getScene()->setCurrentState(res.trajectory_->getLastWayPoint());
            scene->attachObject("box");
            request.useSceneStateAsStart(scene);
            if (!setGoal(place_pose))
                return;
            res = planner->plan(scene, request.getRequest());
            if (res.error_code_.val == moveit_msgs::MoveItErrorCodes::SUCCESS)
            {
                rvizTrajectoryUpdate(scene, res);
                updateCosts(res.trajectory_, path_length, trajectory_duration);
                duration -= res.planning_time_;

                // move from place to home
                request.setAllowedPlanningTime(duration);
                scene->getScene()->setCurrentState(res.trajectory_->getLastWayPoint());
                scene->detachObject("box");
                request.useSceneStateAsStart(scene);
                request.setGoalConfiguration(home);
                res = planner->plan(scene, request.getRequest());
                if (res.error_code_.val == moveit_msgs::MoveItErrorCodes::SUCCESS)
                {
                    rvizTrajectoryUpdate(scene, res);
                    updateCosts(res.trajectory_, path_length, trajectory_duration);
                    duration -= res.planning_time_;
                }
                else
                {
                    duration = 0.;
                    distance_to_goal = goalDistance(request.getRequest());
                }
            }
            else
            {
                duration = 0.;
                // TODO: add heuristic cost between place&home
                distance_to_goal = goalDistance(request.getRequest());
            }
        }
        else
        {
            duration = 0.;
            // TODO: add heuristic cost between pick&place and place&home
            distance_to_goal = goalDistance(request.getRequest());
        }

        std::cout << "HYPERPLAN " << (original_budget - duration) << " " << path_length << " "
                  << trajectory_duration << " " << distance_to_goal << std::endl;
    }

    std::shared_ptr<Manipulator> robot;
    std::string group;
    ScenePtr scene;
    RobotPose pick_pose;
    RobotPose place_pose;
    std::shared_ptr<OMPL::OMPLInterfacePlanner> planner;
    bool use_rviz;
    std::shared_ptr<IO::RVIZHelper> rviz;
    unsigned int num_runs;
    double duration;
};

int main(int argc, char **argv)
{
    ROS ros(argc, argv);
    if (argc < 2)
    {
        ROS_FATAL_STREAM("Command line syntax:\n\t" << argv[0] << " config.yaml");
        exit(-1);
    }

    PickPlaceTask task(YAML::LoadFile(argv[1]));
    task.runBenchmark();
}
