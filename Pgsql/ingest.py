import pickle
import os
from datetime import datetime
import numpy as np
from .sql_db import db, app
from .model import Game, Sample
from scipy.stats import entropy as calc_entropy

# Assuming your Flask app and models are imported
# from your_app import app, db, Game, Sample

SELF_PLAY_DIR = "self_play_data"

def calc_entropy(pi_vec):
    """Calculate entropy of probability distribution"""
    pi_vec = np.array(pi_vec)
    pi_vec = pi_vec + 1e-10  # Add small epsilon to avoid log(0)
    pi_vec = pi_vec / np.sum(pi_vec)  # Normalize
    return -np.sum(pi_vec * np.log(pi_vec))

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def infer_winner_from_game_data(moves):
    """
    Robust winner inference from game data.
    
    Logic:
    1. Red always starts (move 0)
    2. If game has odd number of moves, last move was by Red
    3. If game has even number of moves, last move was by Black
    4. Look at z-values to determine who actually won
    """
    if not moves:
        return None, "no_moves"
    
    num_moves = len(moves)
    last_z = moves[-1][2]
    
    print(f"  Debug: {num_moves} moves, last z-value: {last_z}")
    
    # Method 1: Check if the last move resulted in a win (z=1)
    if last_z == 1:
        # The player who made the last move won
        if num_moves % 2 == 1:  # Odd number of moves = Red made last move
            return "red", "last_move_win"
        else:  # Even number of moves = Black made last move
            return "black", "last_move_win"
    
    # Method 2: Check if it's a draw (z=0)
    if last_z == 0:
        return "draw", "explicit_draw"
    
    # Method 3: Last z=-1, need to analyze the pattern
    # Look for the most recent z=1 to see who was winning
    for i in range(len(moves) - 1, -1, -1):
        z_val = moves[i][2]
        if z_val == 1:
            # Found a winning position
            if i % 2 == 0:  # Red's move
                return "red", "retroactive_win"
            else:  # Black's move
                return "black", "retroactive_win"
    
    # Method 4: Count positive z-values for each player
    red_positive_z = 0
    black_positive_z = 0
    red_total_z = 0
    black_total_z = 0
    
    for i, (_, _, z) in enumerate(moves):
        if i % 2 == 0:  # Red's move
            if z > 0:
                red_positive_z += 1
            red_total_z += z
        else:  # Black's move
            if z > 0:
                black_positive_z += 1
            black_total_z += z
    
    print(f"  Debug: Red +z: {red_positive_z}, total z: {red_total_z:.2f}")
    print(f"  Debug: Black +z: {black_positive_z}, total z: {black_total_z:.2f}")
    
    # If one player has significantly more positive positions
    if red_positive_z > black_positive_z:
        return "red", "positive_count"
    elif black_positive_z > red_positive_z:
        return "black", "positive_count"
    elif red_total_z > black_total_z:
        return "red", "total_z"
    elif black_total_z > red_total_z:
        return "black", "total_z"
    
    # Method 5: Fallback - assume the player who would move next lost
    # If last z=-1, the player about to move is in a bad position
    if num_moves % 2 == 1:  # Next move would be Black's
        return "red", "fallback_next_player_losing"
    else:  # Next move would be Red's
        return "black", "fallback_next_player_losing"

def extract_metadata_winner(metadata):
    """Extract winner from metadata with various possible keys"""
    if not metadata:
        return None
    
    # Try different possible keys
    winner_keys = ['winner', 'result', 'game_result', 'final_result']
    for key in winner_keys:
        if key in metadata and metadata[key]:
            winner = metadata[key]
            # Normalize the winner string
            if isinstance(winner, str):
                winner = winner.lower().strip()
                if winner in ['red', 'r']:
                    return 'red'
                elif winner in ['black', 'b']:
                    return 'black'
                elif winner in ['draw', 'd', 'tie']:
                    return 'draw'
            return winner
    
    return None

def ingest_file(pickle_path):
    try:
        filename = os.path.basename(pickle_path)
        if not filename.endswith(".pkl"):
            return
        
        print(f"\n Processing {filename}")
        
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        
        moves = None
        metadata = {}
        winner = None
        winner_source = None
        
        # Handle different data formats
        if isinstance(data, dict):
            # New format with metadata
            metadata = data.get("metadata", {})
            moves = data.get("game_data", [])
            
            print(f"  Found metadata: {list(metadata.keys()) if metadata else 'None'}")
            
            # Try to get winner from metadata first
            winner = extract_metadata_winner(metadata)
            if winner:
                winner_source = "metadata"
                print(f"  Using metadata winner: {winner}")
            
        elif isinstance(data, list):
            # Old format - just moves
            moves = data
            print(f"  Old format detected - {len(moves)} moves")
        
        else:
            print(f"Unknown data format: {type(data)}")
            return
        
        if not moves:
            print(f"No moves found in {filename}")
            return
        
        # If no winner from metadata, infer from game data
        if not winner:
            winner, winner_source = infer_winner_from_game_data(moves)
            print(f"  Inferred winner: {winner} (method: {winner_source})")
        
        # Validate moves format
        try:
            sample_move = moves[0]
            if len(sample_move) != 3:
                print(f"Unexpected move format: {len(sample_move)} elements")
                return
        except:
            print(f"Cannot validate move format")
            return
        
        # Insert into database
        with app.app_context():
            # Create Game entry - only use columns that exist in your schema
            game = Game(
                date=datetime.now(),
                result=winner,
                source_file=filename,
                z_col=moves[-1][2] if moves else None
            )
            
            db.session.add(game)
            db.session.commit()
            
            print(f"  Created Game ID {game.id} with winner: {winner}")
            
            # Add samples
            samples_added = 0
            samples_failed = 0
            
            for i, sample_tuple in enumerate(moves):
                try:
                    state_data, pi_vec, z_val = sample_tuple
                    
                    # Convert pi to numpy array for calculations
                    pi_np = np.array(pi_vec, dtype=np.float32)
                    
                    # Calculate entropy and best move probability
                    entropy_val = float(calc_entropy(pi_np))
                    best_prob = float(np.max(pi_np))
                    
                    sample = Sample(
                        game_id=game.id,
                        state=pickle.dumps(state_data),
                        pi=pickle.dumps(pi_vec),
                        z=float(z_val),
                        entropy=entropy_val,
                        best_move_prob=best_prob
                    )
                    
                    db.session.add(sample)
                    samples_added += 1
                    
                except Exception as inner_e:
                    print(f"Failed to parse sample {i}: {inner_e}")
                    samples_failed += 1
            
            db.session.commit()
            print(f"Added {samples_added} samples ({samples_failed} failed)")
            
            # Print summary statistics
            z_values = [move[2] for move in moves]
            z_pos = sum(1 for z in z_values if z > 0)
            z_neg = sum(1 for z in z_values if z < 0)
            z_zero = sum(1 for z in z_values if z == 0)
            
            print(f"Z-value distribution: +1: {z_pos}, -1: {z_neg}, 0: {z_zero}")
            print(f"Winner source: {winner_source}")
            
    except Exception as e:
        print(f"Failed to ingest {pickle_path}: {e}")
        import traceback
        traceback.print_exc()

def clear_db():
    """Clear all entries from the database"""
    with app.app_context():
        deleted_samples = db.session.query(Sample).count()
        deleted_games = db.session.query(Game).count()
        
        db.session.query(Sample).delete()
        db.session.query(Game).delete()
        db.session.commit()
        
        print(f"🧹 Cleared {deleted_samples} samples and {deleted_games} games from database.")

def ingest_all():
    """Ingest all pickle files from the self-play directory"""
    if not os.path.exists(SELF_PLAY_DIR):
        print(f"Directory {SELF_PLAY_DIR} does not exist!")
        return
    
    pkl_files = [f for f in os.listdir(SELF_PLAY_DIR) if f.endswith('.pkl')]
    print(f"Found {len(pkl_files)} .pkl files in {SELF_PLAY_DIR}")
    
    if not pkl_files:
        print("No pickle files to process.")
        return
    
    # Sort files to process them in a consistent order
    pkl_files.sort()
    
    successful = 0
    failed = 0
    
    for filename in pkl_files:
        path = os.path.join(SELF_PLAY_DIR, filename)
        try:
            ingest_file(path)
            successful += 1
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            failed += 1
    
    print(f"\n Processing complete: {successful} successful, {failed} failed")

def validate_database():
    """Validate the ingested data"""
    with app.app_context():
        total_games = db.session.query(Game).count()
        total_samples = db.session.query(Sample).count()
        
        print(f"\n Database Summary:")
        print(f"Total games: {total_games}")
        print(f"Total samples: {total_samples}")
        
        # Count winners
        red_wins = db.session.query(Game).filter(Game.result == 'red').count()
        black_wins = db.session.query(Game).filter(Game.result == 'black').count()
        draws = db.session.query(Game).filter(Game.result == 'draw').count()
        nulls = db.session.query(Game).filter(Game.result.is_(None)).count()
        
        print(f"Red wins: {red_wins}")
        print(f"Black wins: {black_wins}")
        print(f"Draws: {draws}")
        print(f"NULL results: {nulls}")
        
        if nulls > 0:
            print(f"{nulls} games have NULL results - investigate these!")
        
        # Sample z-value distribution
        avg_entropy = db.session.query(db.func.avg(Sample.entropy)).scalar()
        avg_best_prob = db.session.query(db.func.avg(Sample.best_move_prob)).scalar()
        
        print(f"Average entropy: {avg_entropy:.3f}")
        print(f"Average best move prob: {avg_best_prob:.3f}")

if __name__ == '__main__':
    print("Starting data ingestion process...")
    
    # Clear existing data (comment out if you want to append)
    clear_db()
    
    # Ingest all files
    ingest_all()
    
    # Validate results
    validate_database()
    
    print("\n Data ingestion complete!")