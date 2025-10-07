# self_play_modal.py
import glob
import os
import pickle
import modal
import numpy as np
import torch
from datetime import datetime, timezone
import uuid
from .model import XiangqiNet
from .mcts import MCTS
from .evaluation import Evaluator
from .r1_game import XiangqiGame
from .device_config import device
from .modal_config import app, torch_image, vol
from .import self_play
import random

# Modal config

def softmax_temperature(logits, temperature=1.0):
    x = np.array(logits, dtype=np.float32)
    x = x - np.max(x)  # for numerical stability
    x = x / temperature
    e_x = np.exp(x)
    return e_x / np.sum(e_x)

def get_dynamic_sims_per_move(turn_number, base=100, max_sims=700, scale_turn=30):
    """
    Linearly increase sims_per_move from `base` to `max_sims` over `scale_turn` turns.
    """
    ratio = min(turn_number / scale_turn, 1.0)
    return int(base + ratio * (max_sims - base))

def should_adjudicate(game, evaluator, move_count):
        """
        Returns 'red' or 'black' if that side is clearly winning and game is stalling.
        """
        if move_count < 80:
            return None, 0.0, 0, 0  # Don't adjudicate too early
        
        if move_count < 100:
            major_piece_gap = 2
            score_threshold = 2.0
        
        if move_count < 200:
            major_piece_gap = 1
            score_threshold = 1.5
        else:
            return 'draw', 0.0, None, None

        score = evaluator.evaluate(game)
        
        # Convert relative score to absolute perspective
        red_is_ahead = score > score_threshold
        black_is_ahead = score < -score_threshold
        red_majors = 0
        black_majors = 0

        if red_is_ahead or black_is_ahead:
            # Count major pieces: Rook, Cannon, Knight
            
            for row in game.engine.board:
                for piece in row:
                    if piece and piece.name in ['Rook', 'Cannon', 'Knight']:
                        if piece.color == 'red':
                            red_majors += 1
                        else:
                            black_majors += 1
            gap = red_majors - black_majors

            if red_is_ahead and gap >= major_piece_gap:
                return 'red', score, red_majors, black_majors
            elif black_is_ahead and gap <= -major_piece_gap:
                return 'black', score, red_majors, black_majors

        return None, score, red_majors, black_majors

def assign_game_result(game_history, winner):
    game_data = []
    for (state, pi, player) in game_history:
        if winner == 'draw':
            z = 0
        else:
            z = 1 if player == winner else -1
        game_data.append((state, pi, z))
    return game_data

@app.function(image= torch_image, volumes={"/vol": vol}, cpu=1.0, timeout= 24*60*60)
def run_self_play_modal(num_games=10, sims_per_move = 750, temperature=2.0):
    print(f"Starting GPU-accelerated self-play on {device}")

    model = XiangqiNet().to(device)
    #checkpoint_path = "/vol/final_xiangqi_model_state_dict.pt"  # path to your trained weights
    #model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # set to evaluation mode for inference

    DATA_DIR = "/vol/self_play_data"
    os.makedirs(DATA_DIR, exist_ok=True)
    # Determine where to resume
    existing_files = glob.glob(os.path.join(DATA_DIR, "game_*.pkl"))
    start_index = len(existing_files)
    print(f"Found {start_index} existing games. Resuming from game {start_index + 1}")
    

    def run_game():
        timestamp = datetime.now(timezone.utc).isoformat()
        game_id = str(uuid.uuid4())
        game = XiangqiGame()
        mcts = MCTS(model, sims_per_move=750)
        evaluator = Evaluator()       
        game_history = []
        print("==== NEW GAME ====")
        moves=[]
        game_history=[] # [(state_tensor, pi)]
        game.print_board()    

        while not game.is_game_over():
        # Subtract 1 because turn_number is incremented before game-over is detected
            winner, score, red_majors, black_majors = should_adjudicate(game, evaluator, game.engine.turn_number)
            if winner:
                print(f"Game adjudicated in favor of {winner} | Eval score = {score:.2f}, Red majors = {red_majors}, Black majors = {black_majors}")
                game_data = assign_game_result(game_history, winner)
                metadata = {
                    'game_id': game_id,
                    'timestamp': timestamp,
                    'winner': winner,
                    'termination_type': 'adjudication',
                    'final_score': score,  # from should_adjudicate or evaluator
                    'total_moves': game.engine.turn_number - 1,
                    'model_version': getattr(model, 'version', 'unknown')
                }
                return game_data, metadata

            
            #Dynamic sims_per_move based on turn number
            #sims_per_move = get_dynamic_sims_per_move(game.engine.turn_number)
            #mcts = MCTS(model, sims_per_move)
            print(f"[Turn {game.engine.turn_number}] Using {sims_per_move} simulations")
            print(f"Player to move: {game.engine.current_turn}")
            print("\nCurrent board:")
            

            root_game = game.clone()
            best_move, root = mcts.run(root_game)
            # NEW: Debug MCTS visits
            
            
            print(f"Root visits: {sum(root.N.values())}")  # Should be ~sims_per_move
            print(f"Top moves: {sorted(root.N.items(), key=lambda x: -x[1])[:3]}")  # Top 3 moves
            
            
            # Must implement this
            

            if best_move is None:
                print("MCTS returned no move — likely game is over.")

            legal_moves = game.get_legal_actions()
            if best_move not in legal_moves:
                print(f"Warning: MCTS returned illegal move {best_move}")
                print(f"Legal moves were: {legal_moves}")
                best_move = random.choice(legal_moves)
                print(f"Fallback to random legal move: {best_move}")


            # Generate pi vector from root node visit counts
            pi = np.zeros(8100, dtype=np.float32)
            for move, count in root.N.items():
                pi[move] = count
            pi = softmax_temperature(pi, temperature)
            #if np.sum(pi) == 0:
            #        pi += 1e-10 # Avoid division by zero
            #pi = np.exp(pi / temperature)
            #pi /= np.sum(pi)
    
            #state_tensor = game.get_state_tensor().unsqueeze(0)  # [1, 32, 10, 9]
            state_tensor = game.get_state_tensor().cpu().numpy()
            game_history.append((state_tensor, pi, game.get_current_player()))

            # Play the move
            moves.append(best_move)
            game.step(best_move)
            game.print_board()
            #turn += 1
            #print(f"\n===Turn {turn}===")
            #Add extra safety here
            
        print(f"Game ended — winner: {game.get_winner()}")
        winner = game.get_winner()
        score = evaluator.evaluate(game)
        game_data = assign_game_result(game_history, winner)
        # Subtract 1 because turn_number is incremented before game-over is detected
        metadata = {
            'game_id': game_id,
            'timestamp': timestamp,
            'winner': winner,
            'termination_type': 'checkmate' if winner else 'draw',
            'final_score': score,  
            'total_moves': game.engine.turn_number - 1, 
            'model_version': getattr(model, 'version', 'unknown')
        }
        return game_data, metadata

    # Data path
    for i in range(start_index + 1, start_index + 1 + num_games):
        print(f"Game {i - start_index}/{num_games}")
        game_data = run_game()

        file_path = os.path.join(DATA_DIR, f"game_{i}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(game_data, f)

        print(f"Saved {len(game_data)} samples to {file_path}")


        # Load existing data or create empty
        #try:
        #    with open(DATA_PATH, "rb") as f:
        #        existing_data = pickle.load(f)
        #except FileNotFoundError:
        #    existing_data = []

        # Append new game data and save immediately
        #with open(DATA_PATH, "wb") as f:
        #    pickle.dump(existing_data + game_data, f)

        #print(f"Saved {len(game_data)} samples after game {i+1}")




