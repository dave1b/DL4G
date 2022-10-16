import time
from monte_carlo_tree_search.node import Node

class MCTS:
    def monte_carlo_tree_search(self, run_time_seconds, current_player_number, all_hands):
        end_time = time.time() + run_time_seconds
        root_node = Node(0, 0, all_hands)
        while time.time() < end_time:
            pass

    def simulate_round(self):
        pass
