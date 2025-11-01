import sys

from connector import Connector
from engine import Engine
from evaluator import Evaluator

def run_online():
    Connector()

def run_training():
    resume = '-n' not in sys.argv
    evaluator = Evaluator(load_saved=False)
    evaluator.train(resume=resume)

def run_self_play():
    iterations = 10
    if len(sys.argv) > 2:
        try:
            iterations = int(sys.argv[2])
        except ValueError:
            pass
    
    print('starting self-play for', iterations, 'iterations')
    evaluator = Evaluator(load_saved=False)
    evaluator.train_self_play(num_iterations=iterations)

def run_debug():
    engine = Engine()
    engine.board.set_fen('rnbq1rk1/pp2ppbp/3p1np1/2p1P3/2PP1B2/5N2/PP1N1PPP/R2QKB1R b KQ - 0 7')
    move = engine.pick_move()
    print('selected move:', move)

if __name__ == '__main__':
    commands = {
        'run': run_online,
        'learn': run_training,
        'selfplay': run_self_play,
        'debug': run_debug
    }
    
    if len(sys.argv) <= 1:
        command = 'run'
    else:
        command = sys.argv[1]
    
    handler = commands.get(command)
    if handler:
        handler()