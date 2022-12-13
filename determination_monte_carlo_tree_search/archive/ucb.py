import math
from dmcts_node import Node


class UCB:
    def __init__(self, c=1):
        self.c = c

    def ucb_value(self, total_visits: int, node_win_count: float, node_visits: int) -> float:
        if node_visits == 0:
            return 2147483647
        ucb = (node_win_count / node_visits) + self.c * (math.sqrt((math.log(total_visits) / node_visits)))
        return ucb

    def find_best_node_ucb(self, node: Node):
        parent_visits = node.visits_count
        best_child = None
        best_score = -1
        for child in node.children:
            score = self.ucb_value(parent_visits, child.win_count, child.visits_count)
            if score > best_score:
                best_child = child
                best_score = score

        if best_child is None:
            print("No best children found")
        return best_child
