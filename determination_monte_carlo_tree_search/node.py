import numpy as np


class Node:

    def __init__(self, parent: 'Node' or None, action: int or None, player_nr: int or None, next_player: int or None,
                 nr_players: int = 4):
        self.parent = parent
        self.action = action
        self.player_nr = player_nr
        self.next_player = next_player
        self.nr_players = nr_players
        self.children = []
        self.win_score = 0
        self.win_count = 0
        self.visits_count = 0
        self.card = None

        def increment_visits(self):
            self.visits_count += 1

        def get_random_child(self):
            return np.random.choice(self.childs)

        def add_child(self, node: 'Node'):
            self.childs.append(node)

        def get_child_with_max_score(self):
            best_child = self.children[0]
            for child in self.children:
                if child.win_score > best_child:
                    best_child = child
            return best_child

        def get_child_with_max_visit_count(self):
            max_visit_child = self.children[0]
            for child in self.children:
                if child.visit_count > max_visit_child.visit_count:
                    max_visit_child = child
            return max_visit_child

        def get_child_cards(self):
            child_cards = []
            for child in self.children:
                child_cards.append(child.card)
            return child_cards

        def is_root(self):
            return self.parent is None

        def is_terminal(self):
            return self.next_player is None
