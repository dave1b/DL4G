from node import Node
from jass.game.game_util import *
import numpy as np


class ISMCTS:
    def __init__(self, cards_played_before, current_trick, player_hand, player_number, parent_node, current_trump):
        self.cards_played_before = cards_played_before
        self.current_trick = current_trick
        self.player_hand = player_hand
        self.player_number = player_number
        self.current_trump = current_trump
        self.parent_node = parent_node
        self.child_nodes = []
        self.trick_points = None
        self.root_node = Node(self.cards_played_before, self.current_trick, None, current_trump, None)
        self.all_cards = np.ones(36)
        self.unplayed_cards_by_enemy_one_hot = self.unplayed_cards_by_enemy(self)
        self.unplayed_cards_by_enemy_as_int = convert_one_hot_encoded_cards_to_int_encoded_list(self.unplayed_cards_by_enemy_one_hot)

    def unplayed_cards_by_enemy(self):
        return self.all_cards - self.cards_played_before - self.player_hand - self.current_trick

    def start(self, time_in_seconds):
        for unplayed_card in self.unplayed_cards_by_enemy_one_hot:
            pass


