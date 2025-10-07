from .board import initialize_board
from Engine.pieces import Rook, Knight, Cannon, General, Advisor, Elephant, Soldier


class Game:
    def __init__(self):
        self.board = initialize_board()
        self.current_turn = 'red'  # Red moves first in Xiangqi
        self.game_over = False
        self.move_history = []  # To track game history
        self.captured_pieces = {'red': [], 'black': []}
        self.general_pos = {'red': None, 'black': None}
        self._update_general_positions()
        self.check_history = [] #stores positions where check occurred
        self.turn_number = 0
        self.piece_move_count = {}
        self.board_state_counts = {}
        self.winner = None
    
    def _update_general_positions(self):
        """Efficiently scan and cache general positions"""
        # Reset positions
        self.general_pos = {'red': None, 'black': None}
        
        # Only check palace rows (where generals can be)
        palace_rows = {'red': (7, 8, 9), 'black': (0, 1, 2)}
        for color, rows in palace_rows.items():
            for x in rows:
                for y in range(3, 6):  # Palace columns
                    piece = self.board[x][y]
                    if piece and piece.name == 'General':
                        self.general_pos[color] = (x, y)
                        break  # Each general has only one position

    def switch_turn(self):
        """Switch turn to the other player"""
        self.current_turn = 'black' if self.current_turn == 'red' else 'red'

    def is_valid_move(self, start, end):
        """Validate a move, including check prevention and flying general rule."""
        sx, sy = start
        ex, ey = end
    
    # Basic validations (boundaries, turn, etc.)
        if not (0 <= sx < 10 and 0 <= sy < 9 and 0 <= ex < 10 and 0 <= ey < 9):
            return False
    
        piece = self.board[sx][sy]
        target = self.board[ex][ey]
    
        if not piece or piece.color != self.current_turn:
            return False
        if target and target.color == self.current_turn:
            return False
    
    # Check if the move follows piece rules (e.g., General moves one space)
        if not piece.is_valid_move(start, end, self.board):
            return False
    
    # Check if the move causes flying general violation
        if self.flying_general_violation(start, end):
            return False
    
    # Simulate the move to check for self-check
        original_piece = target
        original_gen_pos = self.general_pos[piece.color] if piece.name == 'General' else None
    
    # Temporarily update the board
        self.board[ex][ey] = piece
        self.board[sx][sy] = None
        if piece.name == 'General':
            self.general_pos[piece.color] = end
    
    # Check if this move leaves the General in check
        in_check = self.is_in_check(piece.color)
    
    # Revert the board state
        self.board[sx][sy] = piece
        self.board[ex][ey] = original_piece
        if piece.name == 'General':
            self.general_pos[piece.color] = original_gen_pos
    
        return not in_check  # Move is valid only if it doesn't result in check
    
    def is_in_check(self, color):
    #Check if the current player's General is under attack by reusing piece movement logic."""
        general_pos = self.general_pos[color]
        opponent_color = 'black' if color == 'red' else 'red'
    
    # Check all opponent pieces to see if they can attack the General
        for x in range(10):
            for y in range(9):
                piece = self.board[x][y]
                if piece and piece.color == opponent_color:
                    if piece.is_valid_move((x, y), general_pos, self.board):
                        return True
        return False

    def flying_general_violation(self, start, end):
        """Check if moving a piece causes the generals to face each other unobstructed."""
        red_gen = self.general_pos['red']
        black_gen = self.general_pos['black']
    
        if not red_gen or not black_gen:
            return False  # One general is already captured
    
        piece = self.board[start[0]][start[1]]
    
    # Simulate the move (temporarily update positions)
        if piece.name == 'General':
            if piece.color == 'red':
                red_gen = end  # New position for red general
            else:
                black_gen = end  # New position for black general
    
    # Check if they are in the same file (column)
        if red_gen[1] == black_gen[1]:
            min_x, max_x = sorted([red_gen[0], black_gen[0]])
        # Check if there are any pieces between them
            for x in range(min_x + 1, max_x):
                if self.board[x][red_gen[1]] is not None:
                    return False  # There's a blocking piece
            return True  # Generals face each other with no pieces in between
    
        return False  # Generals are not aligned
        
    
    def make_move(self, start, end, verbose=False):
        #Temperory track for debug
        """Execute a move after validation, with optimized General tracking."""
        if not self.is_valid_move(start, end):  # <- Uses the improved validator
            if verbose:
                print(f"Invalid move attempted by {self.current_turn}: {start} -> {end}")
            return False
        if verbose:
            print(f"\n--- {self.current_turn}'s turn ---")
            print(f"Attempting move: {start} -> {end}")
            print(f"Piece at start: {self.board[start[0]][start[1]]}")
            print(f"Target at end: {self.board[end[0]][end[1]]}")

        sx, sy = start
        ex, ey = end
        piece = self.board[sx][sy]
        target = self.board[ex][ey]
        
        
        # Handle capture (including General capture)
        if target is not None:
            self.captured_pieces[target.color].append(target)
           # print(f"{piece.color}'s {piece.name} captures {target.color}'s {target.name}!")
            if target.name == 'General':  # Game ends immediately on General capture
                self.general_pos[target.color] = None
                self.game_over = True
                self.winner = self.current_turn
                
            # Reset repetition history on capture
            self.board_state_counts.clear()  

        # Execute move
        self.board[ex][ey] = piece
        self.board[sx][sy] = None
        self.move_history.append((start, end))

    # Update General position *before* moving (for check detection)
        if piece.name == 'General':
            self.general_pos[piece.color] = (ex, ey)  # Immediate update

    
    
    #Checking perpetual
        # Move has already been made. Now check if opponent is in check.
        opponent_color = 'black' if self.current_turn == 'red' else 'red'  # Define opponent here
        if self.is_in_check(opponent_color):
            self._log_check()
            if self._is_perpetual_check():
                print(f"{self.current_turn} loses by perpetual check!")
                self.game_over = True
                 #ensures the checker is marked as loser
                return True #move stands, but results in game over

        if target is None:
            board_hash = self._get_board_hash()
            self.board_state_counts[board_hash] = self.board_state_counts.get(board_hash, 0) + 1

            # Check for threefold repetition draw
            if self.board_state_counts[board_hash] >= 3:
                #print("Draw by threefold repetition.")
                self.game_over = True

            
        self.turn_number  += 1

        self.switch_turn()
        #print(f"After move: {self.current_turn}'s turn")

        self.check_win_conditions()

        #Temperory debug:
        #if self.game_over:
        #    print("GAME OVER STATE REACHED")
        #else:
        
        #print(f"Next turn: {self.current_turn}")
        
        #self.display_board()

        return True
        

    # No need to update General position again here—already handled above
    # Check win conditions (e.g., checkmate/stalemate)
         
    def _get_board_hash(self):
        hash_val = hash(self.current_turn)
        for x in range(10):
            for y in range(9):
                if piece := self.board[x][y]:
                    hash_val ^= hash((x, y, piece.name, piece.color))
        return hash_val
    
    def _get_check_details(self):
        """Return positions of checking piece and checked king"""
        checked_color = 'black' if self.current_turn == 'red' else 'red'
        king_pos = self.general_pos[checked_color]
        
        # Find all attacking pieces
        attackers = []
        for x in range(10):
            for y in range(9):
                piece = self.board[x][y]
                if piece and piece.color == self.current_turn:
                    if piece.is_valid_move((x,y), king_pos, self.board):
                        attackers.append(((x,y), piece.name))
        
        return attackers #(Takes into account double check) If want more faster version then remove  piece.name aboveand include return attackers[0] if attackers else None"""
    
    def _log_check(self):
        """Record check details to history"""
        attacker_pos = self._get_check_details()
        if attacker_pos:
            checked_color = 'black' if self.current_turn == 'red' else 'red'
            self.check_history.append((
                self.current_turn,                # Checking player
                self._get_board_hash(),           # Full board state
                attacker_pos,                     # (x,y) of attacking piece
                self.general_pos[checked_color]   # (x,y) of checked king
            ))

    def _is_perpetual_check(self):
        """True if current player made 3+ identical consecutive checks"""
        if len(self.check_history) < 3:
            return False

        # Get last 3 checks by current player
        last_checks = [h for h in self.check_history[-3:] if h[0] == self.current_turn]
        if len(last_checks) < 3:
            return False

        # Unpack last 3 checks
        (p1, h1, attacker1, king1), (p2, h2, attacker2, king2), (p3, h3, attacker3, king3) = last_checks[-3:]

        # All must be: same player, same board hash, same attacker/king positions
        return (h1 == h2 == h3 and 
                attacker1 == attacker2 == attacker3 and
                king1 == king2 == king3)

    def check_win_conditions(self):
        """Check for checkmate or General capture."""
    # Case 1: General capture already handled in make_move()
        #if self.game_over:
        #    return
    
        current_color = self.current_turn
        if not self.general_pos.get(current_color):
            if not self.winner:
                self.winner = 'black' if current_color == 'red' else 'red'
            self.game_over = True
            return
        
        in_check = self.is_in_check(current_color)
        legal_moves = self.get_all_legal_moves(current_color)

    # Case 2: Checkmate (in check + no moves)
        if in_check and not legal_moves:
            self.winner = 'black' if current_color == 'red' else 'red'
            self.game_over = True
            
            

    # Case 3: Stalemate (not in check + no moves)
        elif not in_check and not legal_moves:
            self.winner = 'draw'
            self.game_over = True
            #print("Stalemate! The game is a draw.")
            
            

    def display_board(self):
        """Print the current board state"""
        print("\n   " + " ".join(str(i) for i in range(9)))
        for i, row in enumerate(self.board):
            row_display = []
            for piece in row:
                if piece is None:
                    # Show river boundary
                    if i == 4 or i == 5:
                        row_display.append("~~")
                    else:
                        row_display.append("· ")
                else:
                    row_display.append(str(piece))
            print(f"{i} [" + " ".join(row_display) + "]")
        print(f"\n{self.current_turn}'s turn\n")

    def get_all_legal_moves(self, color):
        """Brute-force check all possible moves for a color.
        Returns: List[((start_x, start_y), (end_x, end_y))]"""
        """Rules-complete: Gets all valid moves for a color (with check prevention)"""
        legal_moves = []
    
        # Check every possible start and end position
        for sx in range(10):
            for sy in range(9):
            # Only look at pieces of our color
                piece = self.board[sx][sy]
                if not piece or piece.color != color:
                    continue
                
                # Check every possible destination
                for ex in range(10):
                    for ey in range(9):
                        if self.is_valid_move((sx, sy), (ex, ey)):
                            # Simulate move to check if it leaves us in check
                            if not self._move_leaves_in_check((sx, sy), (ex, ey)):
                                legal_moves.append(((sx, sy), (ex, ey)))
    
        return legal_moves

    def _move_leaves_in_check(self, start, end):
        """Returns True if making this move leaves us in check"""
        sx, sy = start
        ex, ey = end
        piece = self.board[sx][sy]
        target = self.board[ex][ey]
    
    # Simulate the move
        self.board[ex][ey] = piece
        self.board[sx][sy] = None
        original_gen_pos = None
    
    # Update General position if moving General
        if piece.name == 'General':
            original_gen_pos = self.general_pos[piece.color]
            self.general_pos[piece.color] = (ex, ey)
    
    # Check if we're in check after the move
        in_check = self.is_in_check(piece.color)
    
    # Undo the move
        self.board[sx][sy] = piece
        self.board[ex][ey] = target
        if piece.name == 'General':
            self.general_pos[piece.color] = original_gen_pos
    
        return in_check

    def get_possible_moves(self, position):
        """Get all valid moves for a piece at given position"""
        """Quick UI helper: Gets raw moves for one piece (no check validation)"""
        """WARNING: Returns POTENTIAL moves without check validation. 
           For rules-compliant moves, use get_all_legal_moves()."""
        # Useful for AI or move hinting
        if not (0 <= position[0] < 10 and 0 <= position[1] < 9):
            return []
            
        piece = self.board[position[0]][position[1]]
        if piece is None or piece.color != self.current_turn:
            return []
            
        possible_moves = []
        for x in range(10):
            for y in range(9):
                if self.is_valid_move(position, (x, y)):
                    possible_moves.append((x, y))
        return possible_moves

    def get_board(self):
        # For webapp (frontend+backend)
        """Return board as JSON-serializable structure for frontend use."""
        board_state = []

        for row in self.board:
            row_state = []
            for piece in row:
                if piece is None:
                    row_state.append(None)
                else:
                    row_state.append({
                        "type": piece.name,
                        "color": piece.color
                    })
            board_state.append(row_state)

        return {
            "board": board_state,
            "turn": self.current_turn
        }
    
    def load_board(self, board_state, turn, game_over = False, winner = None):
        """
        Restore the internal board from a JSON-serializable 2D array.
        Each cell in board_state is either None or a dict: {"type": "Rook", "color": "black"}.
        """
        new_board = []

        # Rebuild each row
        for row in board_state:
            new_row = []
            for cell in row:
                if cell is None:
                    new_row.append(None)
                else:
                    piece_type = cell["type"]
                    piece_color = cell["color"]

                    # Recreate the correct piece object
                    if piece_type == "Rook":
                        new_row.append(Rook(piece_color))
                    elif piece_type == "Knight":
                        new_row.append(Knight(piece_color))
                    elif piece_type == "Cannon":
                        new_row.append(Cannon(piece_color))
                    elif piece_type == "Soldier":
                        new_row.append(Soldier(piece_color))
                    elif piece_type == "Advisor":
                        new_row.append(Advisor(piece_color))
                    elif piece_type == "Elephant":
                        new_row.append(Elephant(piece_color))
                    elif piece_type == "General":
                        new_row.append(General(piece_color))
                    else:
                        raise ValueError(f"Unknown piece type: {piece_type}")

            new_board.append(new_row)

        # Replace the board and current turn
        self.board = new_board
        self.current_turn = turn
        self.winner = winner
        self.game_over = game_over        # Rebuild general positions for check detection (only for optimization purpose)
        self.general_pos = {"red": None, "black": None}
        for x, row in enumerate(new_board):
            for y, piece in enumerate(row):
                if piece is not None and piece.name == "General":
                    self.general_pos[piece.color] = (x, y)

    def get_winner(self):
        return self.winner if self.game_over else None
    
    
                       
#g1 = Game()
# Red moves first
#g1.make_move((2,3), (1,3))  # Red Soldier moves
#print(f"Winner: {g1.winner}")
#g.make_move((3,0), (4,0))  # Black Soldier moves
#g.make_move((7,1), (5,1))  # Red Cannon moves
#g.make_move((2,1), (4,1))  # Black Cannon moves
#g.make_move((5,0), (4,0))
#print("Red captured:", [p.name for p in g.captured_pieces['red']])
#print("Black captured:", [p.name for p in g.captured_pieces['black']])
#print("Game over:", g.game_over)

#g1.make_move((6,0), (5,0))  # Red
#g1.make_move((3,0), (4,0))  # Black
#g1.make_move((5,0), (4,0))  # Red captures
#print(g1.board[4][0])  # Should be RSoldier
#print(g1.board[5][0])  # Should be None
#print(g1.captured_pieces['black'])  # Should show [Soldier]