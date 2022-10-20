class Node:
    def __init__(self, cards_played_before, cards_played_in_current_trick,parent_node, current_trump,
                 root_node,
                 is_root=False):
        self.cards_played_before = cards_played_before
        self.cards_played_in_current_trick = cards_played_in_current_trick
        # self.player_number = player_number
        self.current_trump = current_trump
        self.root_node = root_node
        self.parent_node = parent_node
        self.child_nodes = []
        self.is_root = is_root
        self.trick_points = None


    def backward_induct(self):
        pass
