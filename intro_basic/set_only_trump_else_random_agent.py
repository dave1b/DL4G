from jass.game.game_util import *
from jass.game.game_observation import GameObservation
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


class SetOnlyTrumpElseRandomAgent(Agent):
    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()
        self.minTrumpValue = 68
        self.round = 0

    def action_trump(self, obs: GameObservation) -> int:
        im_loch = obs.dealer - 1 == obs.player_view
        print("im loch: ", im_loch)
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

        if obs.forehand == -1:
            if _trump_scores.max() >= self.minTrumpValue:
                return _best_trump
            else:
                return PUSH
        if obs.forehand == 0:
            return int(_best_trump)

    def action_play_card(self, obs: GameObservation) -> int:
        print("Spielzug: ", self.round)
        self.round += 1
        self.round = self.round % 9
        _encoded_valid_cards_ = convert_one_hot_encoded_cards_to_int_encoded_list(
            self._rule.get_valid_cards_from_obs(obs))
        _valid_cards = (self._rule.get_valid_cards_from_obs(obs))
        _trump = obs.trump

        # wenn player selber angesagt hat und am Zug ist
        if obs.dealer - 1 == obs.player_view:

            condition = (_encoded_valid_cards_ > 9 * _trump) & (_encoded_valid_cards_ < 9 * (_trump + 1))

            _player_trump_cards = np.extract(_encoded_valid_cards_)
            print("karten in hand: ", np.array(_encoded_valid_cards_))
            print("trump: ", _trump)
            print("trumpfe in hand: ", _player_trump_cards)

            _player_trump_cards_normalized = np.array(_player_trump_cards).__add__(-_trump * 9)
            print("trumpfe in hand normalisiert: ", _player_trump_cards_normalized)

            player_trump_cards_in_trump_score = np.zeros(9, dtype=int)

            if len(_player_trump_cards_normalized) > 0:
                for _card in _player_trump_cards_normalized:
                    print("trump_cards_nomalized: ", np.array(_card))
                    player_trump_cards_in_trump_score[_card] = np.array(trump_score[_card.astype(int)])
                print("die ubergemappten trump-punkte sind :", player_trump_cards_in_trump_score)
                print("die maximale karte ist: ", player_trump_cards_in_trump_score.max())
                _card_to_play = player_trump_cards_in_trump_score.argmax() + _trump * 9
                print("die zu spielende karte ist: ", _card_to_play)
                return _card_to_play

        # wenn Spieler nicht angesagt hat
        # we use the global random number generator here
        return int(np.random.choice(np.flatnonzero(_valid_cards)))


def main():
    from jass.arena.arena import Arena
    from jass.agents.agent_random_schieber import AgentRandomSchieber

    # setup the arena
    arena = Arena(nr_games_to_play=5)
    player = AgentRandomSchieber()
    my_player = SetOnlyTrumpElseRandomAgent()

    arena.set_players(my_player, player, my_player, player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
