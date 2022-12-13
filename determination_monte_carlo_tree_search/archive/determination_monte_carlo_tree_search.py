from jass.game import game_state_util
from jass.game.game_rule import GameRule
from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber

from determination_monte_carlo_tree_search.ucb import UCB
from dmcts_node import Node
from jass.game.game_util import *
from jass.game.game_observation import *
import numpy as np


class DMCTS:
    def __init__(self, game_obs: GameObservation):
        self.game_obs = game_obs
        self.root = Node(parent=None, action=None, player_nr=None, next_player=self.game_obs.player)
        self.rule = RuleSchieber()
        self.game_sim = GameSim(self.rule)
        # self.game_sim.init_from_state(self.game_obs)
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

    def run_simulations(self, iterations: int):
        for _ in range(iterations):
            self.run_simulation()

    def generate_random_hands(self, player_hand):
        cards = np.arange(0, 36, dtype=np.int32)
        player_hand_int = convert_one_hot_encoded_cards_to_int_encoded_list(player_hand)
        np.random.shuffle(cards)
        for card in cards:
            if card in player_hand_int:
                cards = np.delete(cards, np.where(cards == card))

        hands = np.zeros(shape=[4, 36], dtype=np.int32)
        i = 0
        for x in range(0, 4):
            if x == self.game_obs.player_view:
                hands[x,] = player_hand
            else:
                if i == 0:
                    hands[x, cards[0:9]] = 1
                if i == 1:
                    hands[x, cards[9:18]] = 1
                if i == 2:
                    hands[x, cards[18:27]] = 1
                i += 1
        return hands

    def run_simulation(self):
        hand = self.game_obs.hand
        game_sim = GameSim(self.rule)
        random_hands_players = self.generate_random_hands(hand)
        state = game_state_util.state_from_observation(self.game_obs, random_hands_players)
        game_sim.init_from_state(state)
        possible_moves = RuleSchieber.get_valid_actions_from_obs(self.game_obs)
        action = self.rollout_policy(possible_moves)
        game_sim.action_play_card(action)

        # node = self.node_selection.tree_policy(self.root, game_sim)
        # rewards = self.reward_calc.calculate_reward(node, game_sim)
        # while not node.is_root():
        #     node.propagate(rewards)

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]
