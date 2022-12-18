import pandas as pd
from jass.game.game_util import *
from jass.game.game_state_util import *
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from numpy import ndarray
from determination_monte_carlo_tree_search.dmcts_node import DMCTSNode
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from tensorflow import keras
import logging


class DMCTSAgent(Agent):
    def __init__(self):
        super().__init__()
        logging.basicConfig(level=logging.DEBUG)
        self.model_name = "mlp_model_v2"
        # self.model_name = "./determination_monte_carlo_tree_search/mlp_model_v2"
        self._rule = RuleSchieber()
        self.round = 0
        self.thread_count = 22
        # self.thread_count = (min(32, (os.cpu_count() or 1) + 4))
        self.time_budget_for_algorythm = 3
        self.__dmcts_iteration_count = 3
        self.feature_columns = [
            # Diamonds
            'DA', 'DK', 'DQ', 'DJ', 'D10', 'D9', 'D8', 'D7', 'D6',
            # Hearts
            'HA', 'HK', 'HQ', 'HJ', 'H10', 'H9', 'H8', 'H7', 'H6',
            # Spades
            'SA', 'SK', 'SQ', 'SJ', 'S10', 'S9', 'S8', 'S7', 'S6',
            # Clubs
            'CA', 'CK', 'CQ', 'CJ', 'C10', 'C9', 'C8', 'C7', 'C6',
        ]
        logging.info("Thread count: %s", self.thread_count)

    def action_trump(self, game_observation: GameObservation) -> int:
        logging.info('Trump request mlp')
        # calculate if forehand
        forehand = game_observation.player_view == (game_observation.dealer + 3) % 4
        hand = self._rule.get_valid_cards_from_obs(game_observation)
        data = self.__convert_hand_to_predict_form(hand, int(forehand))
        # load model
        mlp_model = keras.models.load_model(self.model_name)
        prediction = mlp_model.predict(data)[0]
        logging.info("Prediction: %s", prediction)
        trump = prediction.argmax(axis=0)
        if (trump == 6):
            if (forehand):
                return 10
            if (not forehand):
                prediction[6] = 0
                trump = prediction.argmax(axis=0)
        logging.info("Chosen trump: %s", trump)
        return int(trump)

    def action_play_card(self, game_observation: GameObservation) -> int:
        self.round += 1
        logging.info("Round: %s ", self.round)
        self.round = self.round % 9

        best_actions_list = []
        # run dmcts
        for i in range(0, self.__dmcts_iteration_count):
            future_objects = []
            process_pool_executor = ProcessPoolExecutor(max_workers=self.thread_count)
            for x in range(0, self.thread_count):
                node = self.__construct_root_node(game_observation)
                future = process_pool_executor.submit(node.best_action, self.time_budget_for_algorythm)
                future_objects.append(future)
            logging.info("%s threads running...", self.thread_count)
            best_actions_list += (list(map(lambda future_object: future_object.result(), future_objects)))

        # count best action
        counts = Counter(best_actions_list)
        best_action_overall = (max(counts.most_common(), key=lambda key_value: key_value[1]))[0]
        logging.info("Best actions: %s", best_actions_list)
        logging.info("Best overall action: %s", best_action_overall)
        logging.info("Best card to play: %s", card_strings[best_action_overall])
        logging.info("-------------------------------------------------")
        return best_action_overall

    def __convert_hand_to_predict_form(self, hand, forehand: int):
        np.transpose(hand)
        forehand = pd.DataFrame(np.array([forehand]), columns=["FH"])
        data = pd.DataFrame([hand], columns=self.feature_columns)
        data = pd.concat([data, forehand], axis=1)
        for color in 'DHSC':
            # Jack and nine combination
            new_col = '{}_J9'.format(color)
            data[new_col] = data['{}J'.format(color)] & data['{}9'.format(color)]
            # Ace King Queen combination
            new_col = '{}_AKQ'.format(color)
            data[new_col] = data['{}A'.format(color)] & data['{}K'.format(color)] & data['{}Q'.format(color)]
            # Six seven eight combination
            new_col = '{}_678'.format(color)
            data[new_col] = data['{}6'.format(color)] & data['{}7'.format(color)] & data['{}8'.format(color)]
        return data

    @staticmethod
    def __calculate_random_hands(game_obs: GameObservation) -> ndarray:
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

    def __construct_root_node(self, game_observation: GameObservation) -> DMCTSNode:
        hands = self.__calculate_random_hands(game_observation)
        game_state = state_from_observation(game_observation, hands)
        dmcts_node = DMCTSNode(game_state, game_observation.player_view)
        return dmcts_node


from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber


def main():
    # Jass Arena for testing Agents
    arena = Arena(nr_games_to_play=1, cheating_mode=False, print_every_x_games=1)
    arena.set_players(DMCTSAgent(), DMCTSAgent(),
                      DMCTSAgent(), DMCTSAgent())
    arena.play_all_games()
    print(arena.points_team_0.sum(), arena.points_team_1.sum())


if __name__ == '__main__':
    main()
