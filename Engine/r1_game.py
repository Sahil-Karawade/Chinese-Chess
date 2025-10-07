# rl_game.py

import copy
from .game import Game
import torch
import numpy as np
import random
from .device_config import device

class XiangqiGame:
    def __init__(self):
        self.engine = Game()
        self.board_shape = (10, 9)
        self.action_size = 90 * 90  # 8100 possible move indices

    def reset(self):
        self.engine = Game()
    
    def print_board(self):
        """Visualizes the board using your piece objects"""
        piece_symbols = {
            ('black', 'Rook'): '♜', ('black', 'Knight'): '♞', ('black', 'Elephant'): '♝',
            ('black', 'Advisor'): '♛', ('black', 'General'): '♚', ('black', 'Cannon'): '♟',
            ('black', 'Soldier'): 'B',
            ('red', 'Rook'): '♖', ('red', 'Knight'): '♘', ('red', 'Elephant'): '♗',
            ('red', 'Advisor'): '♕', ('red', 'General'): '♔', ('red', 'Cannon'): '♙',
            ('red', 'Soldier'): 'R'
        }

        print("\n  " + " ".join(str(i) for i in range(9)))  # File numbers
        for rank in range(10):
            line = str(rank) + " "
            for file in range(9):
                piece = self.engine.board[rank][file]
                if piece:
                    symbol = piece_symbols.get((piece.color, piece.__class__.__name__), '?')
                    line += symbol + " "
                else:
                    line += "· "
            print(line)
    
        print(f"\nCurrent player: {self.engine.current_turn}")
        print(f"Game status: {'OVER' if self.is_game_over() else 'ONGOING'}")
        print(f"Legal moves: {len(self.engine.get_all_legal_moves(self.engine.current_turn))}")

    def get_current_player(self):
        return self.engine.current_turn

    def is_game_over(self):
        return self.engine.game_over

    def get_winner(self):
        return self.engine.winner if self.engine.game_over else None
        

    def get_legal_actions(self):
        legal_moves = self.engine.get_all_legal_moves(self.engine.current_turn)
        return [self.move_to_index(start, end) for (start, end) in legal_moves]

    def move_to_index(self, start, end):
        s_idx = start[0] * 9 + start[1]
        e_idx = end[0] * 9 + end[1]
        return s_idx * 90 + e_idx

    def index_to_move(self, index):
        s = index // 90
        e = index % 90
        start = (s // 9, s % 9)
        end = (e // 9, e % 9)
        return start, end

    def step(self, action_index, verbose=False):
        if self.is_game_over():
            if verbose:
                print("Attempted to step after game over. Ignoring.")
        
        move = self.index_to_move(action_index)
        if verbose:
            print(f"Applying move {move}")
        success = self.engine.make_move(*move, verbose=verbose)
        if not success:
            print(f"Illegal move attempted: {move} (index {action_index})")
            legal_moves = self.get_legal_actions()
            if legal_moves:
                fallback = random.choice(legal_moves)
                fallback_move = self.index_to_move(fallback)
                print(f"Fallback: replacing {move} with {fallback_move}")
                self.engine.make_move(*fallback_move, verbose=verbose)
            else:
                print("No legal fallback moves available. Ending game.")
                self.engine.game_over = True
                return self.clone(), -1, True  # You could treat this as a terminal state

        reward = 0
        done = self.engine.game_over

        if done:
            winner = self.get_winner()
            if winner == self.engine.current_turn:
                reward = 1
            elif winner == 'draw':
                reward = 0
            else:
                reward = -1

        return self.clone(), reward, done

    def get_state_tensor(self):
        #Create a base tensor of zeros with shape (16, 10, 9)
        tensor = np.zeros((16, 10, 9), dtype=np.float32)

        piece_to_channel = {
            'General': 0,
            'Advisor': 1,
            'Elephant': 2,
            'Knight': 3,
            'Rook': 4,
            'Cannon': 5,
            'Soldier': 6
        }

        board = self.engine.board
        current_color = self.engine.current_turn

        for x in range(10):
            for y in range(9):
                piece = board[x][y]
                if piece is None:
                    continue
                ch_base = 0 if piece.color == 'red' else 7
                ch_offset = piece_to_channel.get(piece.name, -1)
                if ch_offset >= 0:
                    tensor[ch_base + ch_offset, x, y] = 1

        # Turn indicator planes
        if current_color == 'red':
            tensor[14, :, :] = 1
        else:
            tensor[15, :, :] = 1

        # Normalize to red's perspective
        if current_color == 'black':
            tensor = self._flip_tensor_perspective(tensor)

        return torch.tensor(tensor)

    def _flip_tensor_perspective(self, tensor):
        """Flip vertically and swap red/black planes."""
        flipped = np.zeros_like(tensor)
        # Swap piece planes and flip vertically
        for i in range(7):
            flipped[i] = np.flip(tensor[i + 7], axis=0)  # red ← black
            flipped[i + 7] = np.flip(tensor[i], axis=0)  # black ← red
        flipped[14] = tensor[15]  # Red's turn plane
        flipped[15] = tensor[14]  # Black's turn plane
        return flipped

    def clone(self):
        return copy.deepcopy(self)
