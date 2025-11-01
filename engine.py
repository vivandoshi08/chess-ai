import numpy as np
import chess

from board_helper import BoardHelper
from node import Node
from evaluator import Evaluator
from tracker import Tracker

class Engine:

    BATCH_SIZE = 1
    SIMS_PER_TURN = 400

    def __init__(self):
        self.evaluator = Evaluator(load_saved=True)
        self.board = chess.Board()
        self.tree = Node(root=True)
        self.cache_hits = 0
        self.cache_misses = 0
    
    def sync_moves(self, moves):
        print(self.board)
        for move in moves:
            self.board.push_san(move)
            if self.tree.children and move in self.tree.children:
                self.tree = self.tree.children[move]
                self.tree.historical = True
            else:
                self.tree = Node(root=True)
    
    def push_moves(self, moves):
        for move in moves:
            self.board.push(move)

    def pop_moves(self, moves):
        for move in moves:
            self.board.pop()

    def pick_move(self):
        Tracker.start('run_concurrent')
        for i in range(Engine.SIMS_PER_TURN):
            if i % 20 == 0:
                print('thinking:', i)
            self.run_batch_search(self.tree, i)
        selected_move = self.tree.pick_popular().move.uci()
        win_rate = round(self.tree.children[selected_move].win_rate * 100, 2)
        print(selected_move, '(', win_rate, '% win)')
        Tracker.stop('run_concurrent')
        return selected_move
    
    def run_batch_search(self, tree, sim_num):
        best_leaf = tree.find_leaf()
        if best_leaf.outcome:
            best_leaf.update_scores(1)
            tree.endings_found += 1
            return
        move_chain = Node.build_path(tree, best_leaf)
        self.push_moves(move_chain)
        if self.board.is_game_over():
            best_leaf.set_terminal(self.board)
            self.pop_moves(move_chain)
            return
        board_state = self.board if self.board.turn else self.board.mirror()
        best_leaf_key = Storage.queue_prediction(board_state)
        if not Storage.is_cached(best_leaf_key):
            self.pop_moves(move_chain)
            Tracker.start('searching_to_cache')
            if Engine.BATCH_SIZE > 1:
                self.cache_siblings(best_leaf, Engine.BATCH_SIZE)
            Tracker.stop('searching_to_cache')
            Tracker.start('predicting')
            Storage.predict_queued(self.evaluator)
            Tracker.stop('predicting')
            self.push_moves(move_chain)
            self.cache_misses += 1
        else:
            self.cache_hits += 1
        value, policy = Storage.fetch(best_leaf_key)
        best_leaf.expand_node(self.board, value, policy)
        self.pop_moves(move_chain)
    
    def cache_siblings(self, best_leaf, num_to_cache):
        potential = []
        current = self.tree
        move_path = Node.build_path(self.tree, best_leaf)
        for move in move_path:
            current = current.children[move.uci()]
            for sibling in current.list_siblings():
                if not sibling.visited and not sibling.cached:
                    sibling.cache_value = abs(current.score - sibling.score)
                    potential.append(sibling)
        potential.sort(key=lambda n: n.cache_value)
        for sibling in potential[:num_to_cache-1]:
            sibling.cached = True
            path = Node.build_path(self.tree, sibling)
            self.push_moves(path)
            board_state = self.board if self.board.turn else self.board.mirror()
            Storage.queue_prediction(board_state)
            self.pop_moves(path)

    def run_single_search(self, tree):
        if not tree.historical:
            self.board.push(tree.move)
        if not tree.visited:
            if self.board.is_game_over():
                tree.set_terminal(self.board)
            else:
                board_state = self.board if self.board.turn else self.board.mirror()
                value, policy = self.evaluator.predict(board_state)
                tree.expand_node(self.board, value[0][0], policy[0])
        elif tree.outcome:
            tree.update_scores(1)
        else:
            self.run_single_search(tree.pick_best())
        if tree.move is not None and not tree.historical:
            self.board.pop()
    
    def __str__(self):
        return f"\n{self.board}\n"


class Storage:

    cache = {}
    pending = {}

    def queue_prediction(board):
        key = board.fen()
        if key not in Storage.pending and key not in Storage.cache:
            Storage.pending[key] = BoardHelper.encode_board(board)
        return key

    def predict_queued(evaluator):
        if len(Storage.pending) == 0:
            return
        boards = np.array(list(Storage.pending.values()))
        values, policies = evaluator.batch_predict(boards)
        for i, key in enumerate(Storage.pending):
            Storage.cache[key] = (values[i][0], policies[i])
        Storage.pending = {}
    
    def fetch(key):
        return Storage.cache.get(key)

    def is_cached(key):
        return key in Storage.cache