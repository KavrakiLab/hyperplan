import os
import configparser
from pathlib import Path

import ConfigSpace as CS
from hpbandster.core.worker import Worker

from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc
from ompl import app as oa

class BaseWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'problems' in kwargs:
            self.problems = []
            for problem in kwargs['problems']:
                if isinstance(problem, str):
                    self.problems.append(self.configureAppProblem(problem))
                elif isinstance(problem, og.SimpleSetup) or isinstance(problem, oc.SimpleSetup):
                    self.problems.append(problem)
                else:
                    raise TypeError(f'Unknown problem type: {type(problem)}')
        else:
            self.problems = \
                [self.configureAppProblem(os.environ['HOME'] + '/omplapp/resources/3D/cubicles.cfg')]

    @staticmethod
    def configureAppProblem(fname):
        config = configparser.ConfigParser(strict=False)
        config.read([fname])
        is3d = False
        if config.has_option("problem", "start.z"):
            is3d = True
            if config.has_option("problem", "control"):
                ctype = config.get("problem", "control")
                if ctype == "blimp":
                    setup = oa.BlimpPlanning()
                elif ctype == "quadrotor":
                    setup = oa.QuadrotorPlanning()
                else:
                    setup = oa.SE3RigidBodyPlanning()
            else:
                setup = oa.SE3RigidBodyPlanning()
        else:
            if config.has_option("problem", "control"):
                ctype = config.get("problem", "control")
                if ctype == "kinematic_car":
                    setup = oa.KinematicCarPlanning()
                elif ctype == "dynamic_car":
                    setup = oa.DynamicCarPlanning()
                else:
                    setup = oa.SE2RigidBodyPlanning()
            else:
                setup = oa.SE2RigidBodyPlanning()
        cfg_dir = Path(fname).parent
        setup.setEnvironmentMesh(str(cfg_dir / config.get("problem", "world")))
        setup.setRobotMesh(str(cfg_dir / config.get("problem", "robot")))
        setup.getSpaceInformation().setStateValidityCheckingResolution(.01)
        start = ob.State(setup.getGeometricComponentStateSpace())
        goal = ob.State(setup.getGeometricComponentStateSpace())
        if is3d:
            start().setX(config.getfloat("problem", "start.x"))
            start().setY(config.getfloat("problem", "start.y"))
            start().setZ(config.getfloat("problem", "start.z"))
            start().rotation().setAxisAngle(config.getfloat("problem", "start.axis.x"),
                                            config.getfloat("problem", "start.axis.y"),
                                            config.getfloat("problem", "start.axis.z"),
                                            config.getfloat("problem", "start.theta"))
            goal().setX(config.getfloat("problem", "goal.x"))
            goal().setY(config.getfloat("problem", "goal.y"))
            goal().setZ(config.getfloat("problem", "goal.z"))
            goal().rotation().setAxisAngle(config.getfloat("problem", "goal.axis.x"),
                                           config.getfloat("problem", "goal.axis.y"),
                                           config.getfloat("problem", "goal.axis.z"),
                                           config.getfloat("problem", "goal.theta"))
        else:
            start().setX(config.getfloat("problem", "start.x"))
            start().setY(config.getfloat("problem", "start.y"))
            start().setYaw(config.getfloat("problem", "start.theta"))
            goal().setX(config.getfloat("problem", "goal.x"))
            goal().setY(config.getfloat("problem", "goal.y"))
            goal().setYaw(config.getfloat("problem", "goal.theta"))
        setup.setStartAndGoalStates(start, goal)
        if is3d:
            if config.has_option("problem", "volume.min.x") and \
               config.has_option("problem", "volume.min.y") and \
               config.has_option("problem", "volume.min.z") and \
               config.has_option("problem", "volume.max.x") and \
               config.has_option("problem", "volume.max.y") and \
               config.has_option("problem", "volume.max.z"):
                bounds = ob.RealVectorBounds(3)
                bounds.low[0] = config.getfloat("problem", "volume.min.x")
                bounds.low[1] = config.getfloat("problem", "volume.min.y")
                bounds.low[2] = config.getfloat("problem", "volume.min.z")
                bounds.high[0] = config.getfloat("problem", "volume.max.x")
                bounds.high[1] = config.getfloat("problem", "volume.max.y")
                bounds.high[2] = config.getfloat("problem", "volume.max.z")
                setup.getGeometricComponentStateSpace().setBounds(bounds)
        else:
            if config.has_option("problem", "volume.min.x") and \
               config.has_option("problem", "volume.min.y") and \
               config.has_option("problem", "volume.max.x") and \
               config.has_option("problem", "volume.max.y"):
                bounds = ob.RealVectorBounds(2)
                bounds.low[0] = config.getfloat("problem", "volume.min.x")
                bounds.low[1] = config.getfloat("problem", "volume.min.y")
                bounds.high[0] = config.getfloat("problem", "volume.max.x")
                bounds.high[1] = config.getfloat("problem", "volume.max.y")
                setup.getGeometricComponentStateSpace().setBounds(bounds)
        return setup

    def configurePlanner(self, config):
        for problem in self.problems:
            si = problem.getSpaceInformation()
            planner = eval('og.%s(si)' % config['planner'])
            params = planner.params()
            for param, value in config.items():
                if params.hasParam(param):
                    params.setParam(param, str(value))
            problem.setPlanner(planner)

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))
        return config_space

# def pathlength(**kwargs):
#     setup = configureProblem(str(ompl_resources_dir / '3D/cubicles.cfg'))
#     configurePlanner(setup, **kwargs)
#     if setup.solve(kwargs['budget']):
#         return setup.getSolutionPath().cost().value()
