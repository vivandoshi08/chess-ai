import random
import sys

import numpy as np
import chess
import chess.pgn
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers

from board_helper import BoardHelper

class Evaluator:

    TRAINING_BATCH = 1024
    MEMORY_LIMIT = 100000

    def __init__(self, load_saved=True):
        self.model = None
        self.interpreter = None
        self.prev_batch_size = None
        if load_saved:
            self.load_weights()

    def load_games():
        pgn = open("/Users/vivandoshi/Documents/vdzero/big.pgn")
        
        while True:
            pointer = pgn.tell()
            headers = chess.pgn.read_headers(pgn)
            if headers is None:
                pgn.seek(0)
                print('restarting pgn')
                continue
            
            if Evaluator.is_valid_game(headers):
                pgn.seek(pointer)
                yield chess.pgn.read_game(pgn)
    
    def is_valid_game(headers):
        if 'Bullet' in headers['Event']:
            return False
        if headers['Result'] not in ['1-0', '0-1']:
            return False
        if headers['Termination'] != 'Normal':
            return False
        
        white_elo = int(headers['WhiteElo'])
        black_elo = int(headers['BlackElo'])
        
        if white_elo < 1800 or black_elo < 1800:
            return False
        if abs(white_elo - black_elo) >= 100:
            return False
        
        return True
    
    def prepare_batches():
        samples = []
        game_gen = Evaluator.load_games()

        while True:
            while len(samples) < Evaluator.MEMORY_LIMIT:
                game = next(game_gen)
                board = game.board()
                moves = list(game.mainline_moves())
                total_moves = len(moves)
                
                for i, move in enumerate(moves):
                    board_state = board if board.turn else board.mirror()
                    inputs = BoardHelper.encode_board(board_state)
                    
                    moves_left = total_moves - i - 1
                    value = BoardHelper.compute_value(game, moves_left, total_moves)
                    
                    move_to_encode = move if board.turn else BoardHelper.flip_move(move)
                    policy = BoardHelper.encode_move(move_to_encode)
                    
                    final_value = value if board.turn else -value
                    samples.append([inputs, final_value, policy])
                    board.push(move)
            
            random.shuffle(samples)
            batch_data = samples[:Evaluator.TRAINING_BATCH]
            samples = samples[Evaluator.TRAINING_BATCH:]
            
            inputs_chunk = np.array([x[0] for x in batch_data], dtype=np.float32)
            value_chunk = np.array([x[1] for x in batch_data], dtype=np.float32)
            policy_chunk = np.array([x[2] for x in batch_data], dtype=np.float32)
            
            yield (inputs_chunk, [value_chunk, policy_chunk])

    def create_network(self):
        weight_decay = 0.00001
        seed = 1
        weight_init = keras.initializers.glorot_uniform(seed=seed)
        weight_init_dense = keras.initializers.glorot_uniform(seed=seed)

        input_layer = layers.Input((12, 8, 8))
        x = layers.Conv2D(256, (5, 5), kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=weight_init, padding='same', data_format='channels_first', use_bias=False)(input_layer)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        x = self.add_residual_blocks(x, 7, weight_decay, weight_init)

        value_output = self.build_value_head(x, weight_decay, weight_init, weight_init_dense)
        policy_output = self.build_policy_head(x, weight_decay, weight_init, weight_init_dense)
        
        self.model = keras.Model(input_layer, [value_output, policy_output])
        print(self.model.summary())

        self.model.compile(
            loss=['mean_squared_error', 'categorical_crossentropy'],
            optimizer=keras.optimizers.Adam(),
            metrics={
                'value': keras.metrics.mean_absolute_error,
                'policy': keras.metrics.categorical_accuracy,
            }
        )
    
    def add_residual_blocks(self, layer, count, weight_decay, weight_init):
        for i in range(count):
            layer = self.single_residual_block(layer, weight_decay, weight_init)
        return layer
    
    def single_residual_block(self, layer, weight_decay, weight_init):
        x = layers.Conv2D(256, (3, 3), kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=weight_init, padding='same', data_format='channels_first', use_bias=False)(layer)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=weight_init, padding='same', data_format='channels_first', use_bias=False)(layer)
        x = layers.Add()([x, layer])
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)
        return x
    
    def build_value_head(self, layer, weight_decay, weight_init, weight_init_dense):
        x = layers.Conv2D(4, (8, 8), kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=weight_init, padding='same', data_format='channels_first', use_bias=False)(layer)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(768, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=weight_init_dense)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(512, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=weight_init_dense)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(256, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=weight_init_dense)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(1, kernel_initializer=weight_init_dense)(x)
        return layers.Activation('tanh', name='value')(x)
    
    def build_policy_head(self, layer, weight_decay, weight_init, weight_init_dense):
        x = layers.Conv2D(8, (8, 8), kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=weight_init, padding='same', data_format='channels_first', use_bias=False)(layer)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=weight_init_dense)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(4096, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=weight_init_dense)(x)
        return layers.Activation('softmax', name='policy')(x)

    def load_weights(self, low_precision=True, path=None):
        if path is None:
            path = f'{sys.path[0]}/network/weights/model'
        self.model = keras.models.load_model(path)
        if low_precision:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            self.interpreter = tf.lite.Interpreter(model_content=converter.convert())

    def train(self, resume=True):
        if resume:
            self.load_weights(low_precision=False)
        else:
            self.create_network()
        self.model.fit(
            Evaluator.prepare_batches(),
            epochs=1000,
            steps_per_epoch=1000,
            callbacks=[ModelSaver()]
        )

    def predict(self, board_or_array):
        if isinstance(board_or_array, np.ndarray):
            input_data = np.array([board_or_array])
        else:
            input_data = np.array([BoardHelper.encode_board(board_or_array)])
        return self.model.predict(input_data, verbose=0)
    
    def batch_predict(self, boards):
        if self.interpreter:
            input_index = self.interpreter.get_input_details()[0]['index']
            target_shape = np.shape(boards)
            if target_shape[0] != self.prev_batch_size:
                print(target_shape[0])
                self.interpreter.resize_tensor_input(input_index, target_shape)
                self.interpreter.allocate_tensors()
                self.prev_batch_size = target_shape[0]
            
            output_details = self.interpreter.get_output_details()
            value_index = output_details[0]['index']
            policy_index = output_details[1]['index']
            
            self.interpreter.set_tensor(input_index, boards)
            self.interpreter.invoke()
            
            value = self.interpreter.get_tensor(value_index)
            policy = self.interpreter.get_tensor(policy_index)
            return value, policy
        return self.model.predict(boards, verbose=0)
    
    def train_self_play(self, num_iterations=10):
        from self_play import SelfPlayTrainer
        
        if self.model is None:
            print('creating network')
            self.create_network()
        
        trainer = SelfPlayTrainer(self)
        trainer.run(num_iterations)

class ModelSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save(f'{sys.path[0]}/network/weights/model')