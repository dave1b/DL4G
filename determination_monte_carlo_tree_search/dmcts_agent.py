from jass.game.game_util import *
from jass.game.game_state import GameState
from jass.agents.agent_cheating import AgentCheating
from jass.game.game_state_util import *
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from dmcts_node import DMCTSNode
import logging
import pickle


class DMCTSAgent(Agent):
    def __init__(self):
        self._logger = logging.getLogger("DMTSAgent")
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()
        self.round = 0

        self._rng = np.random.default_rng()

    def action_trump(self, game_observation: GameObservation) -> int:
        # self._logger.info('Trump request mlp')
        # print("trump request mlp")
        # im_loch = game_observation.dealer - 1 == game_observation.player_view
        # _valid_cards = np.array(self._rule.get_valid_cards_from_obs(game_observation))
        # _valid_cards = np.append(_valid_cards, int(im_loch))
        # data = _valid_cards.reshape(1, -1)
        #
        # pkl_filename = 'C:/Users/Dave/Documents/GitHub/DL4G/ml/models/mlp_model.pkl'
        # with open(pkl_filename, 'rb') as file:
        #     mlp_model = pickle.load(file)
        #
        # trump = mlp_model.predict(data)[0]
        # print(" trumpf ist: ", trump)
        # return int(trump)
        self._logger.info('Trump request')
        if game_observation.forehand == -1:
            # if forehand is not yet set, we are the forehand player and can select trump or push
            if self._rng.choice([True, False]):
                self._logger.info('Result: {}'.format(PUSH))
                return PUSH
        # if not push or forehand, select a trump
        result = int(self._rng.integers(low=0, high=MAX_TRUMP, endpoint=True))
        self._logger.info('Result: {}'.format(result))
        return result

    def action_play_card(self, game_observation: GameObservation) -> int:
        self.round += 1
        self.round = self.round % 9

        # threads und timelimit
        # globale timelimit
        # hands mehrmals genereieren
        # return ist nicht action sonder alle nodes
        # most visited

        hands = self.calculate_random_hands(game_observation)
        game_state = state_from_observation(game_observation, hands)
        dmcts_node = DMCTSNode(game_state, game_observation.player_view)
        best_action = dmcts_node.best_action().parent_action
        print(card_strings[best_action])
        print("-------------------------------------------------")
        return best_action

    def calculate_random_hands(self, game_obs: GameObservation):
        cards = np.arange(0, 36, dtype=np.int32)
        player_hand_int = convert_one_hot_encoded_cards_to_int_encoded_list(game_obs.hand)
        np.random.shuffle(cards)
        for card in cards:
            if card in player_hand_int:
                cards = np.delete(cards, np.where(cards == card))

        hands = np.zeros(shape=[4, 36], dtype=np.int32)
        i = 0
        for x in range(0, 4):
            if x == game_obs.player_view:
                hands[x,] = game_obs.hand
            else:
                if i == 0:
                    hands[x, cards[0:9]] = 1
                if i == 1:
                    hands[x, cards[9:18]] = 1
                if i == 2:
                    hands[x, cards[18:27]] = 1
                i += 1
        return hands


## Use implementation
from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber

# Jass Arena for trying Agents
arena = Arena(nr_games_to_play=50, cheating_mode=False)
arena.set_players(DMCTSAgent(), AgentRandomSchieber(),
                  DMCTSAgent(), AgentRandomSchieber())
arena.play_all_games()
# arena.play_game(dealer=NORTH)
print(arena.points_team_0.sum(), arena.points_team_1.sum())
