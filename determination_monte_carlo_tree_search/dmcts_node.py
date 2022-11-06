from collections import defaultdict

import numpy as np
from jass.game import game_state_util
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber
from queue import Queue


class DMCTSNode:

    def __init__(self, game_state: GameState, player_number, parent=None, parent_action=None, best_child_queue=None):
        self.game_state = game_state
        self.player_number = player_number
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.visits_count = 0
        self.results = defaultdict(int)
        self.win_score = 0
        self.win_count = 0
        self.rule = RuleSchieber()
        self.game_sim = GameSim(RuleSchieber())
        self.untried_actions = None
        self.get_untried_actions()
        self.best_child_queue = best_child_queue

    def get_untried_actions(self):
        valid_cards = self.rule.get_valid_actions_from_obs(
            game_state_util.observation_from_state(self.game_state, self.player_number))
        # valid_cards = self.rule.get_valid_cards(self.game_state.hands[self.game_state.player, :],
        #                                         self.game_state.current_trick, self.game_state.nr_cards_in_trick,
        #                                         self.game_state.trump)
        self.untried_actions = np.flatnonzero(valid_cards)
        return self.untried_actions

    def best_action(self, thread_running_queue: Queue):
        while thread_running_queue.empty():
            v = self.tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        x = self.best_child().parent_action
        self.best_child_queue.put(x)

    def tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def rollout(self):
        self.game_sim.init_from_state(self.game_state)

        while not self.game_sim.is_done():
            # possible_moves = ruleSchieber.get_valid_actions_from_obs(gameSim.get_observation())
            possible_moves = self.rule.get_valid_cards(self.game_sim.get_observation().hand,
                                                       self.game_sim.get_observation().current_trick,
                                                       self.game_sim.get_observation().nr_cards_in_trick,
                                                       self.game_sim.get_observation().trump)
            action = self.rollout_policy(possible_moves)
            self.game_sim.action_play_card(action)

        points = self.game_sim.state.points
        return 1 if points[0] > points[1] else 0

    def difference_win_loss(self):
        wins = self.results[1]
        loses = self.results[-1]
        return wins - loses

    def expand(self):
        action, self.untried_actions = self.untried_actions[-1], self.untried_actions[:-1]
        # print("action: ", action)
        # print("untried_actions: ", self.untried_actions)
        # Initilize GameSim with state
        self.game_sim.init_from_state(self.game_state)
        self.game_sim.action_play_card(action)
        child_node = DMCTSNode(self.game_state, self.player_number, parent=self, parent_action=action,
                               best_child_queue=self.best_child_queue)
        self.children.append(child_node)
        return child_node

    @staticmethod
    def rollout_policy(possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def increment_visits(self):
        self.visits_count += 1

    def get_child_with_max_score(self):
        best_child = self.children[0]
        for child in self.children:
            if child.win_score > best_child:
                best_child = child
        return best_child

    def get_child_with_max_visit_count(self):
        max_visit_child = self.children[0]
        for child in self.children:
            if child.visit_count > max_visit_child.visit_count:
                max_visit_child = child
        return max_visit_child

    def get_child_cards(self):
        child_cards = []
        for child in self.children:
            child_cards.append(child.card)
        return child_cards

    def is_root(self):
        return self.parent is None

    def is_terminal_node(self):
        return self.game_state.nr_played_cards == 36

    def backpropagate(self, result):
        self.visits_count += 1
        self.results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=0.5):
        choices_weights = [(c.difference_win_loss() / c.visits_count) + c_param * np.sqrt(
            (np.log(self.visits_count) / c.visits_count)) for c in self.children]
        return self.children[np.argmax(choices_weights)]
