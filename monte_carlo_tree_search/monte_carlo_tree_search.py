from jass.game.game_util import *
from jass.game.game_state import GameState
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


class IntroBasicAgent(Agent):
    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()
        self.minTrumpValue = 68
        self.round = 0

    def action_trump(self, game_State: GameState) -> int:
        obs = observation_from_state(game_State)
        # print(obs.forehand)
        _trump_scores = np.zeros(4, dtype=int)

        for _trumpIndex in range(3):
            _int_encoded_hand = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
            _trump_scores[_trumpIndex] = calculate_trump_selection_score(_int_encoded_hand, _trumpIndex)

        _best_trump_result, = np.where(_trump_scores == _trump_scores.max())
        _best_trump = _best_trump_result[0]
        print("bester trumpf 1: ", _best_trump )
        print("bester trumpf 2: ", _best_trump , " trump score: ", _trump_scores.max())
        # print("largest points ", _trump_scores.max(), " with trump", _best_trump)

        if game_State.forehand == -1:
            if _trump_scores.max() >= self.minTrumpValue:
                return _best_trump
            else:
                return PUSH
        if game_State.forehand == 0:
            return _best_trump

    def action_play_card(self, game_State: GameState) -> int:
        obs = observation_from_state(game_State)
        print("Spielzug: ", self.round)
        self.round += 1
        self.round = self.round % 9
        _encoded_valid_cards_ = convert_one_hot_encoded_cards_to_int_encoded_list(
            self._rule.get_valid_cards_from_obs(obs))
        _valid_cards = (self._rule.get_valid_cards_from_obs(obs))
        _trump = obs.trump

        # wenn player selber angesagt hat und am Zug ist
        if obs.dealer - 1 == obs.player_view:

            condition = (_encoded_valid_cards_ >= 9 * _trump) & (_encoded_valid_cards_ < 9 * (_trump + 1))

            _player_trump_cards = np.extract(condition, _encoded_valid_cards_)
            print("karten in hand: ", np.array(_encoded_valid_cards_))
            print("trump: ", _trump)
            print("trümpfe in hand: ", _player_trump_cards)

            _player_trump_cards_normalized = np.array(_player_trump_cards).__add__(-_trump * 9)
            print("trümpfe in hand normalisiert: ", _player_trump_cards_normalized)

            player_trump_cards_in_trump_score = np.zeros(9, dtype=int)

            if len(_player_trump_cards_normalized) > 0:
                for _card in _player_trump_cards_normalized:
                    print("trump_cards_nomalized: ", np.array(_card))
                    player_trump_cards_in_trump_score[_card] = np.array(trump_score[_card.astype(int)])
                print("die übergemappten trump-punkte sind :", player_trump_cards_in_trump_score)
                print("die maximale karte ist: ", player_trump_cards_in_trump_score.max())
                _card_to_play = player_trump_cards_in_trump_score.argmax() + _trump * 9
                print("die zu spielende karte ist: ", _card_to_play)
                return _card_to_play

        # wenn Spieler nicht angesagt hat
        # we use the global random number generator here
        return np.random.choice(np.flatnonzero(_valid_cards))


## Use implementation
from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber
# Jass Arena for trying Agents
arena = Arena(nr_games_to_play=100)
arena.set_players(IntroBasicAgent(), AgentRandomSchieber(), IntroBasicAgent(), AgentRandomSchieber())
arena.play_all_games()
# arena.play_game(dealer=NORTH)
print(arena.points_team_0.sum(), arena.points_team_1.sum())