import numpy as np
import logging
from jass.game.game_util import *
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
import pickle


class MLPOnlyTrumpAgent(Agent):
    def __init__(self):
        self._logger = logging.getLogger("MLPOnlyTrumpAgent")
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()
        self.minTrumpValue = 68
        self.round = 0
        self.feature_columns = np.array(
            ['DA', 'DK', 'DQ', 'DJ', 'D10', 'D9', 'D8', 'D7', 'D6', 'HA', 'HK', 'HQ', 'HJ', 'H10',
             'H9', 'H8', 'H7', 'H6', 'SA', 'SK', 'SQ', 'SJ', 'S10', 'S9', 'S8', 'S7', 'S6', 'CA',
             'CK', 'CQ', 'CJ', 'C10', 'C9', 'C8', 'C7', 'C6', 'FH'])
        self.trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]

    def action_trump(self, obs: GameObservation) -> int:
        self._logger.info('Trump request mlp')
        print("trump request mlp")
        im_loch = obs.dealer - 1 == obs.player_view
        _valid_cards = np.array(self._rule.get_valid_cards_from_obs(obs))
        _valid_cards = np.append(_valid_cards, int(im_loch))
        data = _valid_cards.reshape(1, -1)

        pkl_filename = 'C:/Users/Dave/Documents/GitHub/DL4G/ml/models/mlp_model.pkl'
        with open(pkl_filename, 'rb') as file:
            mlp_model = pickle.load(file)

        trump = mlp_model.predict(data)[0]
        print(" trumpf ist: ", trump)
        return int(trump)

    def action_play_card(self, obs: GameObservation) -> int:
        print("Spielzug: ", self.round)
        self.round += 1
        self.round = self.round % 9
        _encoded_valid_cards_ = convert_one_hot_encoded_cards_to_int_encoded_list(
            self._rule.get_valid_cards_from_obs(obs))
        _valid_cards = (self._rule.get_valid_cards_from_obs(obs))
        return np.random.choice(np.flatnonzero(_valid_cards))


def main():
    from jass.arena.arena import Arena
    from jass.agents.agent_random_schieber import AgentRandomSchieber

    # setup the arena
    arena = Arena(nr_games_to_play=1000)
    player = AgentRandomSchieber()
    my_player = MLPOnlyTrumpAgent()

    arena.set_players(my_player, player, my_player, player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
