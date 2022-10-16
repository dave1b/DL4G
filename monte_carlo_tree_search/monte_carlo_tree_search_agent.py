from jass.game.game_util import *
from jass.game.game_state import GameState
from jass.agents.agent_cheating import AgentCheating
from jass.game.game_state_util import *
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent

# score if the color is trump
trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
# score if the color is not trump
no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]


def havePuurWithFour(hand: np.ndarray) -> np.ndarray:
    encoded_hand = convert_one_hot_encoded_cards_to_str_encoded_list(hand)
    colors_count = count_colors(hand)
    result = np.zeros(4, dtype=int)
    if "DJ" in encoded_hand:
        print("Hat diamond Buur")
        if colors_count[0] >= 4:
            result[0] = 1

    if "HJ" in encoded_hand:
        print("Hat heart Buur")
        if colors_count[1] >= 4:
            result[1] = 1

    if "SJ" in encoded_hand:
        print("Hat spade Buur")
        if colors_count[2] >= 4:
            result[2] = 1

    if "CJ" in encoded_hand:
        print("Hat club Buur")
        if colors_count[3] >= 4:
            result[3] = 1
    return result


def calculate_trump_selection_score(_cards, _trump: int) -> int:
    # print("hand: ", _cards)
    # print("trump: ", _trump)
    _score = 0
    for card in _cards:
        # print("trump: 0 - card: ", card)
        if 0 <= card <= 8:
            if _trump == 0:
                _score += trump_score[card]
            else:
                _score += no_trump_score[card]

    for card in _cards:
        card -= 9
        # print("trump: 1 - card: ", card)
        if 0 <= card <= 8:
            if _trump == 1:
                _score += trump_score[card]
            else:
                _score += no_trump_score[card]

    for card in _cards:
        card -= 18
        # print("trump: 2 - card: ", card)
        if 0 <= card <= 8:
            if _trump == 2:
                _score += trump_score[card]
            else:
                _score += no_trump_score[card]

    for card in _cards:
        card -= 27
        # print("trump: 3 - card: ", card)
        if 0 <= card <= 8:
            if _trump == 3:
                _score += trump_score[card]
            else:
                _score += no_trump_score[card]

    # print("score: ", _score)
    return _score


class MonteCarloTreeSearchAgent(AgentCheating):
    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()
        self.minTrumpValue = 68
        self.round = 0

    def action_trump(self, game_state: GameState) -> int:
        obs = observation_from_state(game_state)
        # print(obs.forehand)
        _trump_scores = np.zeros(4, dtype=int)

        for _trumpIndex in range(3):
            _int_encoded_hand = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
            _trump_scores[_trumpIndex] = calculate_trump_selection_score(_int_encoded_hand, _trumpIndex)

        _best_trump_result, = np.where(_trump_scores == _trump_scores.max())
        _best_trump = _best_trump_result[0]
        print("bester trumpf 1: ", _best_trump)
        print("bester trumpf 2: ", _best_trump, " trump score: ", _trump_scores.max())
        # print("largest points ", _trump_scores.max(), " with trump", _best_trump)

        if game_state.forehand == -1:
            if _trump_scores.max() >= self.minTrumpValue:
                return _best_trump
            else:
                return PUSH
        if game_state.forehand == 0:
            return _best_trump

    def action_play_card(self, game_State: GameState) -> int:

        print(game_State.trick_winner)
        print(game_State.trick_points)
        print(game_State.hands)
        print(convert_one_hot_encoded_cards_to_int_encoded_list(game_State.hands[0]))
        print(convert_one_hot_encoded_cards_to_int_encoded_list(game_State.hands[1]))
        print(convert_one_hot_encoded_cards_to_int_encoded_list(game_State.hands[2]))
        print(convert_one_hot_encoded_cards_to_int_encoded_list(game_State.hands[3]))
        print(game_State.nr_cards_in_trick)
        print(game_State.nr_played_cards)
        print(game_State.current_trick)
        print(game_State.trick_first_player)
        print(game_State.dealer)
        print(game_State.player)


## Use implementation
from jass.arena.arena import Arena
from jass.agents.agent_cheating_random_schieber import AgentCheatingRandomSchieber

# Jass Arena for trying Agents
arena = Arena(nr_games_to_play=100, cheating_mode=True)
arena.set_players(MonteCarloTreeSearchAgent(), AgentCheatingRandomSchieber(), MonteCarloTreeSearchAgent(), AgentCheatingRandomSchieber())
arena.play_all_games()
# arena.play_game(dealer=NORTH)
print(arena.points_team_0.sum(), arena.points_team_1.sum())
