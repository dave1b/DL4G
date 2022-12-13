from jass.game import game_state_util
from jass.game.game_rule import GameRule
from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber

from jass.game.game_util import *
from jass.game.game_observation import *
import numpy as np

player = 0


def generate_random_hands(player_hand):
    cards = np.arange(0, 36, dtype=np.int32)
    player_hand_int = convert_one_hot_encoded_cards_to_int_encoded_list(player_hand)
    np.random.shuffle(cards)
    print(player_hand_int)
    for card in cards:
        if card in player_hand_int:
            cards = np.delete(cards, np.where(cards == card))

    print(len(cards))
    hands = np.zeros(shape=[4, 36], dtype=np.int32)
    print(hands)
    i = 0
    for x in range(0, 4):
        print(x)
        if x == player:
            print("is gleich")
            hands[x,] = player_hand
        else:
            if i == 0:  # convert to one hot encoded
                hands[x, cards[0:9]] = 1
            if i == 1:
                hands[x, cards[9:18]] = 1
            if i == 2:
                hands[x, cards[18:27]] = 1
            i += 1
    return hands


one_hot = [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
value = generate_random_hands(one_hot)
print("result")
print(value)
