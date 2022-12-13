from collections import defaultdict
import numpy as np
from threading import Timer
from jass.game import game_state_util
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber
from numpy import ndarray


class DMCTSNode:

    def __init__(self, game_state: GameState, player_number, parent=None, parent_action=None):
        self.game_state = game_state
        self.player_number = player_number
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.visits_count = 0
        self.results = defaultdict(int)
        self.win_score = 0
        self.rule = RuleSchieber()
        self.game_sim = GameSim(RuleSchieber())
        self.untried_actions = None
        self.__get_untried_actions()
        self.is_running = True

    def __get_untried_actions(self) -> ndarray:
        valid_cards = self.rule.get_valid_actions_from_obs(
            game_state_util.observation_from_state(self.game_state, self.player_number))
        self.untried_actions = np.flatnonzero(valid_cards)
        return self.untried_actions

    def best_action(self, time_budget: int) -> int:
        Timer(time_budget, self.__stop_exploration).start()
        while self.is_running:
            v = self.__tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        return self.__best_child().parent_action

    def __stop_exploration(self):
        self.is_running = False

    def __tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.__is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.__best_child()
        return current_node

    def rollout(self) -> int:
        self.game_sim.init_from_state(self.game_state)
        while not self.game_sim.is_done():
            # possible_moves = ruleSchieber.get_valid_actions_from_obs(gameSim.get_observation())
            possible_moves = self.rule.get_valid_cards(self.game_sim.get_observation().hand,
                                                       self.game_sim.get_observation().current_trick,
                                                       self.game_sim.get_observation().nr_cards_in_trick,
                                                       self.game_sim.get_observation().trump)
            action = self.__rollout_policy(possible_moves)
            self.game_sim.action_play_card(action)

        points = self.game_sim.state.points
        return 1 if points[0] > points[1] else 0

    def difference_win_loss(self):
        wins = self.results[1]
        loses = self.results[-1]
        return wins - loses

    def expand(self):
        action, self.untried_actions = self.untried_actions[-1], self.untried_actions[:-1]
        # Initilize GameSim with state
        self.game_sim.init_from_state(self.game_state)
        self.game_sim.action_play_card(action)
        child_node = DMCTSNode(self.game_state, self.player_number, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    @staticmethod
    def __rollout_policy(possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def __increment_visits(self):
        self.visits_count += 1

    def is_terminal_node(self):
        return self.game_state.nr_played_cards == 36

    def backpropagate(self, result):
        self.visits_count += 1
        self.results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

    def __is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def __best_child(self, c_param=0.5):
        choices_weights = [(c.difference_win_loss() / c.visits_count) + c_param * np.sqrt(
            (np.log(self.visits_count) / c.visits_count)) for c in self.children]
        return self.children[np.argmax(choices_weights)]
