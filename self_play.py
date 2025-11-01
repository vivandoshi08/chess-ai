import numpy as np
import chess
import random
from collections import deque

from board_helper import BoardHelper
from node import Node
from engine import Engine

class SelfPlay:

    TEMP_THRESHOLD = 15
    INITIAL_TEMP = 1.0
    FINAL_TEMP = 0.1
    SIMS_PER_MOVE = 200

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.game_history = []
    
    def play_game(self):
        board = chess.Board()
        game_data = []
        move_count = 0
        
        while not board.is_game_over():
            move_count += 1
            temperature = self.get_temperature(move_count)
            
            root = Node(root=True)
            
            for i in range(SelfPlay.SIMS_PER_MOVE):
                self.run_simulation(root, board.copy())
            
            move_probs = self.get_move_probabilities(root, temperature)
            board_state = BoardHelper.encode_board(board if board.turn else board.mirror())
            
            game_data.append({
                'board': board_state,
                'policy': move_probs,
                'player': board.turn
            })
            
            selected_move = self.select_move(root, move_probs)
            board.push(selected_move)
        
        result = self.get_game_result(board)
        training_samples = self.process_game_data(game_data, result)
        
        return training_samples, result
    
    def run_simulation(self, root, board):
        node = root
        search_path = [node]
        
        while node.visited and not node.outcome:
            node = node.pick_best()
            search_path.append(node)
            if node.move:
                board.push(node.move)
        
        if board.is_game_over():
            node.set_terminal(board)
            value = self.get_terminal_value(board)
        else:
            board_state = board if board.turn else board.mirror()
            encoded = BoardHelper.encode_board(board_state)
            value, policy = self.evaluator.predict(encoded)
            value = value[0][0]
            policy = policy[0]
            node.expand_node(board, value, policy)
        
        self.backpropagate(search_path, value, board.turn)
    
    def backpropagate(self, path, value, turn):
        for node in reversed(path):
            node.update_scores(value if node.move is None or board.turn == turn else 1 - value)
    
    def get_temperature(self, move_count):
        if move_count < SelfPlay.TEMP_THRESHOLD:
            return SelfPlay.INITIAL_TEMP
        return SelfPlay.FINAL_TEMP
    
    def get_move_probabilities(self, root, temperature):
        visits = np.array([child.visit_count for child in root.children.values()])
        
        if temperature == 0:
            probs = np.zeros(len(visits))
            probs[np.argmax(visits)] = 1.0
        else:
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / np.sum(visits_temp)
        
        policy_vector = np.zeros(BoardHelper.MOVE_SPACE, dtype=np.float32)
        for i, (move_uci, child) in enumerate(root.children.items()):
            move = chess.Move.from_uci(move_uci)
            index = BoardHelper.move_to_index(move)
            policy_vector[index] = probs[i]
        
        return policy_vector
    
    def select_move(self, root, move_probs):
        moves = list(root.children.keys())
        probs = [move_probs[BoardHelper.move_to_index(chess.Move.from_uci(m))] for m in moves]
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        
        selected_uci = np.random.choice(moves, p=probs)
        return chess.Move.from_uci(selected_uci)
    
    def get_game_result(self, board):
        outcome = board.outcome()
        if outcome is None:
            return 0.0
        
        if outcome.winner is None:
            return 0.0
        elif outcome.winner == chess.WHITE:
            return 1.0
        else:
            return -1.0
    
    def get_terminal_value(self, board):
        result = self.get_game_result(board)
        return (result + 1.0) / 2.0
    
    def process_game_data(self, game_data, result):
        samples = []
        
        for data in game_data:
            board_state = data['board']
            policy = data['policy']
            player = data['player']
            
            if player == chess.WHITE:
                value = result
            else:
                value = -result
            
            samples.append({
                'input': board_state,
                'value': value,
                'policy': policy
            })
        
        return samples


class DataBuffer:

    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add_game(self, samples):
        for sample in samples:
            self.buffer.append(sample)
    
    def sample_batch(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]
        
        inputs = np.array([s['input'] for s in batch], dtype=np.float32)
        values = np.array([s['value'] for s in batch], dtype=np.float32)
        policies = np.array([s['policy'] for s in batch], dtype=np.float32)
        
        return inputs, values, policies
    
    def size(self):
        return len(self.buffer)
    
    def is_ready(self, min_size):
        return len(self.buffer) >= min_size


class SelfPlayTrainer:

    GAMES_PER_ITERATION = 100
    TRAINING_STEPS = 1000
    BATCH_SIZE = 256
    MIN_BUFFER_SIZE = 5000

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.buffer = DataBuffer()
        self.iteration = 0
    
    def train_iteration(self):
        self.iteration += 1
        print('iteration', self.iteration)
        print('generating', SelfPlayTrainer.GAMES_PER_ITERATION, 'games')
        self.generate_games()
        
        if self.buffer.is_ready(SelfPlayTrainer.MIN_BUFFER_SIZE):
            print('training on', self.buffer.size(), 'samples')
            self.train_network()
        else:
            print('buffer:', self.buffer.size(), '/', SelfPlayTrainer.MIN_BUFFER_SIZE, 'skipping')
    
    def generate_games(self):
        self_play = SelfPlay(self.evaluator)
        
        wins = 0
        draws = 0
        losses = 0
        
        for game_num in range(SelfPlayTrainer.GAMES_PER_ITERATION):
            samples, result = self_play.play_game()
            self.buffer.add_game(samples)
            
            if result > 0.5:
                wins += 1
            elif result < -0.5:
                losses += 1
            else:
                draws += 1
            
            if (game_num + 1) % 10 == 0:
                print(' ', game_num + 1, '/', SelfPlayTrainer.GAMES_PER_ITERATION, '|', wins, 'w', draws, 'd', losses, 'l')
        
        print('done:', wins, 'w', draws, 'd', losses, 'l')
    
    def train_network(self):
        for step in range(SelfPlayTrainer.TRAINING_STEPS):
            inputs, values, policies = self.buffer.sample_batch(SelfPlayTrainer.BATCH_SIZE)
            
            loss = self.evaluator.model.train_on_batch(inputs, [values, policies])
            
            if (step + 1) % 100 == 0:
                print('  step', step + 1, '/', SelfPlayTrainer.TRAINING_STEPS, '| loss:', round(loss[0], 4))
        
        self.save_checkpoint()
    
    def save_checkpoint(self):
        import sys
        checkpoint_path = f'{sys.path[0]}/network/weights/iteration_{self.iteration}'
        self.evaluator.model.save(checkpoint_path)
        print('saved iteration', self.iteration)
    
    def run(self, num_iterations):
        for i in range(num_iterations):
            self.train_iteration()
        print('training complete')
