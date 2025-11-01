import numpy as np
import chess

class BoardHelper:

    PENALTY_WEIGHT = 0.1
    PENALTY_CURVE = 0.25
    MOVE_SPACE = 4096

    def encode_board(board):
        piece_types = [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]
        colors = [chess.WHITE, chess.BLACK]
        
        masks = []
        for piece in piece_types:
            for color in colors:
                masks.append(board.pieces(piece, color).mask)
        
        masks_array = np.array(masks, dtype=np.uint64)
        bits = np.unpackbits(masks_array.view(np.uint8))
        return bits.reshape(12, 8, 8).astype(np.float32)
    
    def compute_value(game, moves_left, total_moves):
        penalty = BoardHelper.PENALTY_WEIGHT * (moves_left / total_moves) ** BoardHelper.PENALTY_CURVE
        
        if game.headers['Result'] == '1-0':
            return np.array(1.0 - penalty, dtype=np.float32)
        else:
            return np.array(-1.0 + penalty, dtype=np.float32)

    def encode_move(move):
        result = np.zeros(BoardHelper.MOVE_SPACE, dtype=np.float32)
        result[BoardHelper.move_to_index(move)] = 1.0
        return result
    
    def move_to_index(move):
        return move.from_square * 64 + move.to_square
    
    def flip_move(move):
        from_square = chess.square_mirror(move.from_square)
        to_square = chess.square_mirror(move.to_square)
        return chess.Move(from_square, to_square)