from time import process_time
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from .base_worker import BaseWorker

class SpeedWorker(BaseWorker):
    def compute(self, config_id, config, budget, **kwargs):
        self.configurePlanner(config)
        all_durations = []
        all_path_lengths = []
        loss = budget
        while budget > 0:
            durations = []
            path_lengths = []
            last_solve_time = 0
            for problem in self.problems:
                problem.clear()
                start = process_time()
                if problem.solve(budget):
                    last_solve_time = process_time() - start
                    durations.append(last_solve_time)
                    path_lengths.append(problem.getSolutionPath().length())
                else:
                    budget = 0
                    break
                budget -= last_solve_time
            if len(durations) == len(self.problems):
                all_durations.append(durations)
                all_path_lengths.append(path_lengths)

        if all_durations:
            print('sums: ', np.sum(all_durations, axis=1))
            loss = np.quantile(np.sum(all_durations, axis=1), .7)
        return {'loss': loss,
                'info': {'path_lengths': all_path_lengths, 'durations': all_durations}}

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        planner = CSH.CategoricalHyperparameter(
            name='planner',
            choices=[
                'PRM',
                'LazyPRM',
                'RRT',
                'RRTConnect',
                'EST',
                'BiEST',
                'SBL',
                'KPIECE1',
                'BKPIECE1',
                'LBKPIECE1',
                'PDST'])
        max_nearest_neighbors = CSH.UniformIntegerHyperparameter(
            'max_nearest_neighbors', lower=1, upper=20, default_value=8)
        rnge = CSH.UniformFloatHyperparameter(
            'range', lower=0.001, upper=10000, log=True)
        intermediate_states = CSH.UniformIntegerHyperparameter(
            'intermediate_states', lower=0, upper=1, default_value=0)
        goal_bias = CSH.UniformFloatHyperparameter(
            'goal_bias', lower=0., upper=1., default_value=.05)
        cs.add_hyperparameters([planner, max_nearest_neighbors, rnge, intermediate_states, \
                               goal_bias])

        cond1 = CS.OrConjunction(CS.EqualsCondition(max_nearest_neighbors, planner, 'PRM'),
                                 CS.EqualsCondition(max_nearest_neighbors, planner, 'LazyPRM'))
        cond2 = CS.OrConjunction(CS.EqualsCondition(rnge, planner, 'LazyPRM'),
                                 CS.EqualsCondition(rnge, planner, 'RRT'),
                                 CS.EqualsCondition(rnge, planner, 'RRTConnect'),
                                 CS.EqualsCondition(rnge, planner, 'EST'),
                                 CS.EqualsCondition(rnge, planner, 'BiEST'),
                                 CS.EqualsCondition(rnge, planner, 'SBL'),
                                 CS.EqualsCondition(rnge, planner, 'KPIECE1'),
                                 CS.EqualsCondition(rnge, planner, 'BKPIECE1'),
                                 CS.EqualsCondition(rnge, planner, 'LBKPIECE1'),
                                 CS.EqualsCondition(rnge, planner, 'PDST'))
        cond3 = CS.OrConjunction(CS.EqualsCondition(intermediate_states, planner, 'RRT'),
                                 CS.EqualsCondition(intermediate_states, planner, 'RRTConnect'))
        cond4 = CS.OrConjunction(CS.EqualsCondition(goal_bias, planner, 'RRT'),
                                 CS.EqualsCondition(goal_bias, planner, 'EST'),
                                 CS.EqualsCondition(goal_bias, planner, 'KPIECE1'))
        cs.add_conditions([cond1, cond2, cond3, cond4])
        return cs
