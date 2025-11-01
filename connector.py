import sys
import threading

import berserk
from dotenv import dotenv_values

from engine import Engine


class Connector:

    def __init__(self):
        token = self.load_token()
        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(session=self.session)
        self.listen_events()
    
    def load_token(self):
        try:
            return dotenv_values('.env')['LICHESS_TOKEN']
        except KeyError:
            raise ValueError('missing LICHESS_TOKEN in environment')
    
    def listen_events(self):
        print('waiting for challenges')
        for event in self.client.bots.stream_incoming_events():
            handlers = {
                'challenge': self.on_challenge,
                'gameStart': self.on_game_start
            }
            handler = handlers.get(event['type'])
            if handler:
                handler(event)
    
    def on_challenge(self, event):
        challenge_id = event['challenge']['id']
        if Connector.is_valid_challenge(event):
            self.client.bots.accept_challenge(challenge_id)
        else:
            self.client.bots.decline_challenge(challenge_id)
    
    def on_game_start(self, event):
        match = Match(self.client, event['game']['id'])
        match.start()
    
    def is_valid_challenge(event):
        variant = event['challenge']['variant']['key']
        time_control = event['challenge']['timeControl']['type']
        return variant == 'standard' and time_control == 'unlimited'


class Match(threading.Thread):

    def __init__(self, client, game_id, **kwargs):
        super().__init__(**kwargs)
        self.game_id = game_id
        self.client = client
        self.stream = client.bots.stream_game_state(game_id)
        self.engine = Engine()
        self.is_white = True

    def run(self):
        print(f'game started: {self.game_id}')
        for event in self.stream:
            handlers = {
                'gameStart': self.on_start,
                'gameState': self.on_state,
                'gameFull': self.on_full,
                'chatLine': self.on_chat
            }
            handler = handlers.get(event['type'])
            if handler:
                handler(event)
    
    def on_start(self, event):
        pass

    def on_state(self, event):
        if event['status'] in ['resign', 'aborted']:
            print('game ended:', event['status'])
            sys.exit()
        if event['moves']:
            self.engine.sync_moves([event['moves'].split(' ')[-1]])
        self.send_move()

    def on_full(self, event):
        if event['state']['moves']:
            self.engine.sync_moves(event['state']['moves'].split(' '))
        self.is_white = event['white']['id'] == 'vdzero'
        self.send_move()

    def on_chat(self, event):
        pass

    def send_move(self):
        if self.engine.board.turn == self.is_white:
            move = self.engine.pick_move()
            self.client.bots.make_move(self.game_id, move)