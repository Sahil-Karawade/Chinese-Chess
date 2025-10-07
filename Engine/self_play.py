# self_play.py
import os
import pickle
import glob
import torch
import numpy as np
from .model import XiangqiNet
from .mcts import MCTS
from .r1_game import XiangqiGame
from .device_config import device
from collections import deque
from .evaluation import Evaluator
import random
from datetime import datetime, timezone
import uuid


def softmax_temperature(logits, temperature=1.0):
    x = np.array(logits, dtype=np.float32)
    x = x - np.max(x)  # for numerical stability
    x = x / temperature
    e_x = np.exp(x)
    return e_x / np.sum(e_x)

def get_dynamic_sims_per_move(turn_number, base=100, max_sims=750, scale_turn=30):
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
        else:
            major_piece_gap = 1
            score_threshold = 1.5

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


def run_self_play_game(model, sims_per_move = 750, temperature=1.5):
    timestamp = datetime.now(timezone.utc).isoformat()
    game_id = str(uuid.uuid4())
    game = XiangqiGame()
    mcts = MCTS(model, sims_per_move)
    evaluator = Evaluator()
    print("==== NEW GAME ====")
    turn = 0
    moves=[]
    game_history = []  # [(state_tensor, pi)]
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

        


        #if len(moves) > 60:  # Arbitrary cutoff
            # Randomly assign winner based on material/position
        #    if random.random() < 0.5:
         #       return moves, 'red_wins'  # Force outcome
          #  else:
           #     return moves, 'black_wins'

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
        turn += 1
        print(f"\n===Turn {turn}===")
        #Add extra safety here
        
    print(f"🏁 Game ended — winner: {game.get_winner()}")
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



if __name__ == "__main__":

    model = XiangqiNet().to(device)
    #checkpoint_path = "models/exports/final_xiangqi_model_state_dict.pt"  # path to your trained weights
    #model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # set to evaluation mode for inference

    num_games = 10
    #sims_per_move = 750

    save_dir = "self_play_data"
    os.makedirs(save_dir, exist_ok=True)

    # === Resumption: detect existing games ===
    existing_files = glob.glob(os.path.join(save_dir, "game_*.pkl"))
    existing_game_indices = {
        int(os.path.basename(f).split("_")[1].split(".")[0]) for f in existing_files
    }
    print(f"Found {len(existing_files)} previously saved games.")

    start_index = 1
    while start_index in existing_game_indices:
        start_index += 1

    for i in range(start_index, start_index + num_games):
        print(f"\n Starting self-play game {i}...")
        try:
            game_data, metadata = run_self_play_game(model, sims_per_move=750)
        except Exception as e:
            print(f"Game {i} failed: {e}")
            continue
        save_obj = {
            'metadata': metadata,
            'game_data': game_data
        }
        save_path = os.path.join(save_dir, f"game_{i}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(save_obj, f)
        print(f"Saved {len(game_data)} samples to {save_path}")


    #all_data=[]
    #for i in range(num_games):
    #    print(f"Playing self-play game {i+1}...")
    #    data = run_self_play_game(model)
    #    all_data.extend(data)

    #print(f"Collected {len(all_data)} training samples.")
   # You can save to disk or return all_data to your trainer
    #save_dir = "self_play_data"
    #os.makedirs(save_dir, exist_ok=True)  # Create dir if it doesn't exist

    # Save as a pickle file (human-readable)
    #save_path = os.path.join(save_dir, "self_play_data.pkl")
    #with open(save_path, "wb") as f:
    #    pickle.dump(all_data, f)
    #print(f"Data saved to {save_path}")