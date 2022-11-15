from jass.game.game_util import *
from jass.game.game_state_util import *
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from determination_monte_carlo_tree_search.dmcts_node import DMCTSNode
from concurrent.futures import ProcessPoolExecutor
import logging
from collections import Counter
import pickle


class DMCTSAgent(Agent):
    def __init__(self):
        self._logger = logging.getLogger("DMTSAgent")
        super().__init__()
        self.pkl_filename = 'mlp_model.pkl'
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()
        self.round = 0
        self.thread_count = 16
        # self.thread_count = (min(32, (os.cpu_count() or 1) + 4))
        self.thread_pool_executor = ProcessPoolExecutor(self.thread_count)
        self._rng = np.random.default_rng()
        self.time_budget_for_algorythm = 9
        self.threads_running = False
        print("Thread count: ", self.thread_count)

    def action_trump(self, game_observation: GameObservation) -> int:
        self._logger.info('Trump request mlp')
        print("trump request mlp")
        im_loch = game_observation.dealer - 1 == game_observation.player_view
        _valid_cards = np.array(self._rule.get_valid_cards_from_obs(game_observation))
        _valid_cards = np.append(_valid_cards, int(im_loch))
        data = _valid_cards.reshape(1, -1)

        with open(self.pkl_filename, 'rb') as file:
            mlp_model = pickle.load(file)

        trump = mlp_model.predict(data)[0]
        print(" trumpf ist: ", trump)
        return int(trump)

    def action_play_card(self, game_observation: GameObservation) -> int:
        self.round += 1
        print("round: ", self.round)
        self.round = self.round % 9
        future_objects = []
        for x in range(0, self.thread_count):
            node = self.construct_root_node_threaded(game_observation)
            future = self.thread_pool_executor.submit(node.best_action, self.time_budget_for_algorythm)
            future_objects.append(future)
        print("threads running...")

        best_actions_list = list(map(lambda future_object: future_object.result(), future_objects))

        counts = Counter(best_actions_list)
        print(counts)

        best_action_overall = (max(counts.most_common(), key=lambda key_value: key_value[1]))[0]
        print("all best actions: ", best_actions_list)
        print("Best overall action: ")
        print(best_action_overall)
        print(card_strings[best_action_overall])
        print("-------------------------------------------------")
        return best_action_overall

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

    def construct_root_node_threaded(self, game_observation: GameObservation):
        hands = self.calculate_random_hands(game_observation)
        game_state = state_from_observation(game_observation, hands)
        dmcts_node = DMCTSNode(game_state, game_observation.player_view)
        return dmcts_node


from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber


def main():
    # Jass Arena for trying Agents
    arena = Arena(nr_games_to_play=1, cheating_mode=False, print_every_x_games=1)
    arena.set_players(DMCTSAgent(), DMCTSAgent(),
                      DMCTSAgent(), DMCTSAgent())
    arena.play_all_games()
    # arena.play_game(dealer=NORTH)
    print(arena.points_team_0.sum(), arena.points_team_1.sum())


if __name__ == '__main__':
    main()
