import math
import numpy as np
import torch
from copy import deepcopy
from .device_config import device
from .r1_game import XiangqiGame
from .evaluation import Evaluator


class MCTSNode:
    def __init__(self, game_state, parent=None):
        assert isinstance(game_state, XiangqiGame), "game_state must be XiangqiGame instance"
        self.game = game_state  # XiangqiGame instance
        self.parent = parent
        self.children = {}  # move -> child node
        
        self.N = {}  # Visit count per move
        self.W = {}  # Total value per move
        self.Q = {}  # Mean value per move
        self.P = {}  # Prior probability per move

        self.is_expanded = False

    def is_leaf(self):
        return not self.is_expanded


class MCTS:
    def __init__(self, model, sims_per_move=50, c_puct=0.5, temperature=1.0):
        assert temperature>0
        self.model = model.to(device)  # Force GPU placement
        self.sims_per_move = sims_per_move
        self.c_puct = c_puct
        self.temperature = max(temperature, 0.01)
        self.device = device
        self.evaluate = Evaluator()

    


    def is_capture_move(self, action_index, game):
        """Returns True if the move captures an opponent piece."""
        start, end = game.index_to_move(action_index)
        board = game.engine.board

        from_piece = board[start[0]][start[1]]
        to_piece = board[end[0]][end[1]]

        # Ensure both are valid pieces
        if from_piece is None:
            return False
        if to_piece is None:
            return False

        return from_piece.color != to_piece.color
    
    def is_mate_in_one(self, action, game):
        # Quick temporary deepcopy to simulate the move
        simulated_game = deepcopy(game)
        start, end = simulated_game.index_to_move(action)
        current_player = game.get_current_player()
        if simulated_game.engine.make_move(start, end):
            if simulated_game.is_game_over():
                winner = simulated_game.get_winner()
            # Check if current player (before move) wins after this move
                if winner == current_player:
                    return True
        return False

    def get_captured_piece(self, action_index, game):
        try:
            _, to_pos = game.index_to_move(action_index)
            x, y = to_pos
            return game.engine.board[x][y]
        except Exception as e:
            raise ValueError(f"Invalid action index: {action_index}, error: {e}")
    
    def is_major_back_rank_mobilization(self, move_index, game):
        pos, _ = game.index_to_move(move_index)
        from_row, from_col =  pos
        piece = game.engine.board[from_row][from_col]

        if not piece:
            return False

        # Only major pieces
        if piece.name not in ['Knight', 'Elephant', 'General', 'Advisor']:
            return False

        red_back_rank = 9
        black_back_rank = 0

        if (piece.color == 'red' and from_row == red_back_rank) or (piece.color == 'black' and from_row == black_back_rank):
        # Moved from back rank
            return True

        return False

    def decode_move_index(self, index):
        """Decodes a move index into (from_row, from_col, to_row, to_col)"""
        from_square = index // 90
        to_square = index % 90
        return divmod(from_square, 9) + divmod(to_square, 9)

    def run(self, root_game):
        
        if root_game.is_game_over():
            print("🚫 MCTS received a terminal game state — skipping search.")
            return None, None

        """Run MCTS from root position and return best action"""
        root = MCTSNode(deepcopy(root_game))

        for sim in range(self.sims_per_move):
            node = root
            path = []

            debug_this_sim = (sim == self.sims_per_move - 1)

             # SELECTION PHASE
            while not node.is_leaf() and not node.game.is_game_over():
                move, child = self.select_child(node, debug = debug_this_sim)
                path.append((node, move))
                node = child

            # EXPANSION PHASE
            if not node.game.is_game_over():
                value = self.expand(node)
            else:
                value = self.get_terminal_value(node.game)

            # BACKPROPAGATION PHASE
            for parent_node, move in reversed(path):  # Reverse for efficiency
                #old_q = parent_node.Q.get(move, 0)  # Current Q-value (or 0 if never visited)
                parent_node.N[move] = parent_node.N.get(move, 0) + 1
                parent_node.W[move] = parent_node.W.get(move, 0) + value
                parent_node.Q[move] = parent_node.W[move] / parent_node.N[move]
                #print(f"📈 Backprop move {move}: old_Q={old_q:.3f}, new_Q={parent_node.Q[move]:.3f}, value={value:.3f}")
                value = -value  # Flip for opponent's perspective
        print("\n" + "="*50)
        print("🏁 FINAL MOVE SELECTION")
        print("="*50)
        action = self.select_action(root)
        return action, root

    def select_child(self, node, debug =  False):
        """Select child node using PUCT algorithm"""
        total_visits = sum(node.N.values())  # Avoid division by zero
        best_score = -math.inf
        best_move = None
        best_child = None
        if debug:
            print(f"\n🔍 MCTS Selection Debug (total visits: {total_visits})")

        #Store all scores for comparisons
        move_scores = []
        for move in node.P.keys():
            # PUCT formula: Q + U
            Q = node.Q.get(move, 0)
            U = self.c_puct * node.P[move] * math.sqrt(total_visits) / (2 + node.N.get(move, 0)) 
            score = Q + U
            if debug:
                move_scores.append({
                'move': move,
                'Q': Q,
                'U': U, 
                'P': node.P[move],
                'visits': node.N.get(move, 0),
                'total_score': score,
                'is_mate': self.is_mate_in_one(move, node.game),
                'is_capture': self.is_capture_move(move, node.game)
                })

            if score > best_score:
                candidate_child = node.children.get(move)
                if candidate_child is not None:
                    best_score = score
                    best_move = move
                    best_child = candidate_child
                    
        if best_child is None and best_move is not None:
            if debug:
                print(f"🚫 Move {best_move} has no child — removing and retrying")
            del node.P[best_move]
            return self.select_child(node, debug)

        #if debug:
            #print("  Move breakdown:")
            #for data in sorted(move_scores, key=lambda x: x['total_score'], reverse=True):
            #    mate_flag = "🏆" if data['is_mate'] else ""
            #    capture_flag = "⚔️" if data['is_capture'] else ""
            #    print(f"{mate_flag}{capture_flag} Move {data['move']}: "
            #        f"Q={data['Q']:.3f}, U={data['U']:.3f}, P={data['P']:.3f}, "
            #        f"visits={data['visits']}, TOTAL={data['total_score']:.3f}")
        
            #print(f"  🎯 Selected: {best_move} (score: {best_score:.3f})")
        

        return best_move, best_child

    def expand(self, node):
        """Expand leaf node using neural network predictions"""
        legal_actions = node.game.get_legal_actions()
        state_tensor = node.game.get_state_tensor().unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            #policy = torch.softmax(policy_logits[0] / self.temperature, dim=0).cpu().numpy()
            logits = policy_logits[0].cpu().numpy()
        #print(f"🧠 NN evaluation: {value.item():.3f} (positive = good for current player)")
        #print(f"🧠 Current player: {node.game.get_current_player()}")
        #Extra logits for legal moves only
        legal_logits = {a: logits[a] for a in legal_actions}

        #Optional: numerical stability - subtract max logit
        max_logit = max(legal_logits.values())
        exp_logits = {a: np.exp((logits[a]-max_logit) / self.temperature) for a in legal_actions}
        
        #Adding validation after getting policy
        #policy = np.nan_to_num(policy, nan=1.0/len(legal_actions))
        #policy = np.maximum(policy, 1e-10)
        #policy = policy/policy.sum()
        total = sum(exp_logits.values()) + 1e-10
        adjusted_policy = {a: exp_logits[a] / total for a in legal_actions}

        #Bias added to remove moves that don't take game toward conclusion for the initial phase
        # Bias to promote capturing and mobilizing major back-rank pieces early
        # Adjusted policy using evaluator-based scaling
        #adjusted_policy = {}

        early_game_limit = 20
        turn = node.game.engine.turn_number

        for action in legal_actions:
            score = exp_logits[action]

            #Mate in 1 heuristics
            if self.is_mate_in_one(action, node.game):
                score *= 2000.0  # very large bonus to prioritize mate moves

            # Capture bonus scaled using Evaluator's PIECE_VALUES
            #if node.game.get_current_player() == 'red':
            if self.is_capture_move(action, node.game):
                captured_piece = self.get_captured_piece(action, node.game)
                norm_value = self.evaluate.PIECE_VALUES[captured_piece.name] 
                    
                    #norm_value = raw_value / 1000.0  # Normalize (e.g., Chariot = 1.0)
                score *= (1.5 + 0.25*norm_value)  # Tunable

        # Mobilization bias (only in early game)
            if turn < early_game_limit and self.is_major_back_rank_mobilization(action, node.game):
                progress = turn / early_game_limit
                mobilization_scale = 1.0 + (1.0 - progress) * 0.5  # decays over time
                score *= mobilization_scale

        # Bonus for advancing soldier across the river
            if self.evaluate._soldier_advancement(node.game): #and node.game.get_current_player() == 'red':
                score *= 1.0  # small boost

            #if node.game.get_current_player() == 'red':
            #    score *= 2.5
            adjusted_policy[action] = score

        #Normalize to get valid probability distribution
        total = sum(adjusted_policy.values()) + 1e-10
        adjusted_policy= {a: adjusted_policy[a] / total for a in legal_actions}

        # Add Dirichlet noise to root node for exploration
        if node.parent is None:
            legal_actions = list(adjusted_policy.keys())
            alpha = 0.3  # or try 0.15–0.5
            epsilon = 0.25
            noise = np.random.dirichlet([alpha] * len(legal_actions))
            for i, a in enumerate(legal_actions):
                adjusted_policy[a] = (1 - epsilon) * adjusted_policy[a] + epsilon * noise[i]
        # Store priors and initialize child nodes
        for action in legal_actions:
            node.P[action] = adjusted_policy[action]
            node.N[action] = 0
            node.W[action] = 0
            node.Q[action] = 0
            
            child_game = deepcopy(node.game)
            success = child_game.engine.make_move(*child_game.index_to_move(action), verbose=False)

            if not success or self.is_illegal_general_face(child_game):
                #print(f"❌ Skipping illegal move: {action}")
                # Clean up P dictionary to avoid using it in selection
                if action in node.P:
                    del node.P[action]
                continue

            #child_game.step(action, verbose=False)
            node.children[action] = MCTSNode(child_game, parent=node)

        #print(f"Predicted value: {value.item()}")  # In evaluate() or expand()

        node.is_expanded = True
        #print(f"🧠 Neural network evaluation: {value.item():.3f}")
        scaled_value =  value.item()*5
        #print(f"🧠 NN raw: {value.item():.3f}, scaled: {scaled_value:.3f}")
        return scaled_value

    def get_terminal_value(self, game):
        """Handle terminal game states"""
        winner = game.get_winner()
        if winner == 'draw':
            return 0.0
        return 1.0 if winner == 'red' else -1.0

    def evaluate(self, game):
        """Evaluate non-terminal state"""
        with torch.no_grad():
            state_tensor = game.get_state_tensor().unsqueeze(0).to(self.device)
            _, value = self.model(state_tensor)
        return value.item()

    #def select_action(self, root):
       # """Select final action based on visit counts"""
        #if not root.N:
        #    return np.random.choice(root.game.get_legal_actions())

        # Temperature-adjusted visit counts
        #visits = np.array(list(root.N.values()), dtype=np.float64)
        #if np.all(visits==0):
        #    return np.random.choice(list(root.N.keys()))
        #log_visits =np.log(visits + 1e-10)/self.temperature
        #probs = np.exp(log_visits - log_visits.max())
        #probs /= probs.sum()

        #assert not np.any(np.isnan(probs))
        #assert np.all(probs>=0)
        #return np.random.choice(list(root.N.keys()), p=probs)
    def select_action(self, root, debug = False):
        """Select the move with the highest visit count (greedy strategy)"""
        if not root.N:
            print("⚠️ No available moves in root.N, picking random legal move.")
            legal = root.game.get_legal_actions()
            return np.random.choice(legal) if legal else None
        if debug:
            print("\n📊 Final Visit Count Summary:")
            sorted_moves = sorted(root.N.items(), key=lambda x: x[1], reverse=True)
        
            for i, (move, visits) in enumerate(sorted_moves[:5]):
                mate_flag = "🏆" if self.is_mate_in_one(move, root.game) else ""
                capture_flag = "⚔️" if self.is_capture_move(move, root.game) else ""
                q_value = root.Q.get(move, 0)
                p_value = root.P.get(move, 0)
            
                print(f"  {i+1}. {mate_flag}{capture_flag} Move {move}: "
                    f"{visits} visits, Q={q_value:.3f}, P={p_value:.3f}")

        # Return the move (key) with the highest visit count (value)
        best_move= max(root.N.items(), key=lambda item: item[1])
        if debug:
            print(f"🎯 Choosing move {best_move[0]} with visit count {best_move[1]}")
        return best_move[0]

    def move_to_index(self, move):
        """Convert board move to action index (maintained from your original)"""
        (sx, sy), (ex, ey) = move
        return sx * 9 * 90 + sy * 90 + ex * 9 + ey
    
    def is_illegal_general_face(self, game):
        """Returns True if generals face each other directly with no blocking piece."""
        red_pos = game.engine.general_pos['red']
        black_pos = game.engine.general_pos['black']
    
        if red_pos[1] != black_pos[1]:
            return False  # Not in same column

        col = red_pos[1]
        min_row = min(red_pos[0], black_pos[0]) + 1
        max_row = max(red_pos[0], black_pos[0])

        for r in range(min_row, max_row):
            if game.engine.board[r][col] is not None:
                return False  # Something is blocking

        return True  # No blocker and aligned ⇒ illegal


#if __name__ == "__main__":
     # Adjust import if needed
    
  #  xiangqi_game = XiangqiGame()  # This wraps the Game class
   # engine_board = xiangqi_game.engine.board

    #print(f"✅ Type of xiangqi_game: {type(xiangqi_game)}")
    #print(f"✅ Type of xiangqi_game.engine: {type(xiangqi_game.engine)}")
    #print(f"✅ Type of xiangqi_game.engine.board: {type(engine_board)}")
    
    #piece = engine_board[0][0]
    #if piece:
    #    print(f"📍 Piece at (0,0): {piece.name} ({piece.color})")
    #else:
     #   print("📍 No piece at (0,0)")

#
#print(f"Board type:{type(game.engine.board)}")