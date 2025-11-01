import numpy as np

from board_helper import BoardHelper

class Node:

    EXPLORE_FACTOR = 1
    DEPTH_WEIGHT = 0.5

    def __init__(self, move=None, root=False):
        self.move = move
        self.historical = True if root else False
        self.visited = False
        self.cached = False
        self.visit_count = 0
        self.total_score = 0
        self.position_value = 0
        self.policy = 0
        self.win_rate = None
        self.score = 0
        self.outcome = None
        self.endings_found = 0
        self.parent = None
        self.children = {}
    
    def expand_node(self, board, value, policy):
        self.position_value = value
        legal_moves = list(board.legal_moves)
        encoded_moves = [BoardHelper.move_to_index(move if board.turn else BoardHelper.flip_move(move)) for move in legal_moves]
        valid_indices = np.array(encoded_moves, dtype=np.uint32)
        self.update_scores(1 - (value + 1) / 2)
        move_weights = policy[valid_indices]
        move_weights /= sum(move_weights)
        for move, weight in zip(legal_moves, move_weights):
            child = Node(move)
            child.parent = self
            child.policy = weight
            self.children[move.uci()] = child
            self.visited = True
    
    def set_terminal(self, board):
        self.outcome = board.outcome()
        self.visited = True
        self.update_scores(1)
    
    def update_scores(self, value):
        self.total_score += value
        self.visit_count += 1
        self.win_rate = self.total_score / self.visit_count
        if self.parent and not self.historical:
            self.parent.update_scores(1 - value)
        
    def pick_best(self):
        def calculate_score(child):
            if child.win_rate is None:
                child.win_rate = 1 - self.win_rate
            exploration = Node.EXPLORE_FACTOR * child.policy * (self.visit_count ** 0.5) / (child.visit_count + 1)
            child.score = child.win_rate + exploration
            return child.score
        return max(self.children.values(), key=calculate_score)
    
    def find_leaf(self):
        if not self.visited or self.outcome:
            return self
        else:
            return self.pick_best().find_leaf()
    
    def pick_popular(self):
        return max(self.children.values(), key=lambda c: c.visit_count)
    
    def list_siblings(self):
        if self.parent is None:
            return []
        return [child for child in self.parent.children.values() if child is not self]
        

    def find_all_leaves(self):
        if not self.visited and not self.historical:
            return [self]
        leaves = []
        for child in self.children.values():
            leaves += child.find_all_leaves()
        return leaves
    
    def format_tree(self, recurse=False, visitedOnly=False, depth=0):
        move_name = self.move.uci() if self.move else 'root'
        parent_visits = self.parent.visit_count if self.parent else 0
        result = f'{move_name} - visits: {self.visit_count} - policy: {self.policy} - parent visits: {parent_visits} - valuation: {self.win_rate} - searchability: {self.score}'
        if (recurse or depth < 1) and len(self.children):
            for child in self.children.values():
                if child.visited or not visitedOnly:
                    indent = '   ' * depth
                    result += f'\n{indent}|——{child.format_tree(recurse, visitedOnly, depth + 1)}'
        return result
    
    def build_path(start, end):
        path = []
        current = end
        while current != start and current.parent:
            path.append(current.move)
            current = current.parent
        return list(reversed(path))
    