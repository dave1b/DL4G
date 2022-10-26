from jass.game.game_rule import GameRule
from jass.game.game_sim import GameSim

from determination_monte_carlo_tree_search.ucb import UCB
from node import Node
from jass.game.game_util import *
from jass.game.game_observation import *
import numpy as np


class DMCTS:
    def __init__(self, game_obs: GameObservation):
        self.game_obs = game_obs
        self.root = Node(parent=None, action=None, player_nr=None, next_player=self.game_obs.player)
        self.rule = GameRule()
        self.game_sim = GameSim(self.rule)
        self.game_sim.init_from_state(self.game_obs)

        promising_node, depth = self.select_promising_node()
        valid_cards = np.flatnonzero(promising_node.round.get_valid_cards())


    def select_promising_node(self):
        node = self.root
        depth = 0
        while len(node.children) != 0:
            ucb = UCB()
            node = ucb.find_best_node_ucb(node)
            depth += 1
        return node, depth

        # def run_simulations(self, iterations: int):
        #     for _ in range(iterations):
        #         self.run_simulation()
        game_sim = GameSim(self.rule)
        game_sim.init_from_state(self.state)

        node = self.node_selection.tree_policy(self.root, game_sim)

        rewards = self.reward_calc.calculate_reward(node, game_sim)

        while not node.is_root():
            node.propagate(rewards)
