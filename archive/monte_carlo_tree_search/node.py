class Node:
    def __init__(self, node_depth, node_index, hands):
        self.children = []
        self.hands = hands
        self.node_index = [node_depth, node_index]
        self.accumulated_payoffs_w = [0, 0]
        self.number_of_sub_simulations_n = 0
        self.is_leaf = False

    def go_down(self):
        pass