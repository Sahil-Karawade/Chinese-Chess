class Evaluator:
    # Piece values (adjust based on strength- 10 max and 0 min)
    PIECE_VALUES = {
        'General': 10.0,  # Essentially infinite (checkmate trumps all)
        'Rook': 2.5,
        'Cannon': 1.8,
        'Knight': 1.0,
        'Elephant': 0.4,
        'Advisor': 0.15,
        'Soldier': 0.7  # Base + progressive bonus
    }

    def evaluate(self, game):
        """Main evaluation function"""
        score = 0
        
        # 1. Material count
        score += self._material_balance(game)
        
        # 2. Piece activity
        #score += self._piece_activity(game)
        
        # 3. King safety
        #score += self._king_safety(game)
        
        # 4. Pawn structure
        #score += self._soldier_advancement(game)
        
        return score if game.engine.current_turn == 'red' else -score

    def _material_balance(self, game):
        """Count total piece values"""
        balance = 0
        for x in range(10):
            for y in range(9):
                if piece := game.engine.board[x][y]:
                    value = self.PIECE_VALUES[piece.name]
                    balance += value if piece.color == 'red' else -value
        return balance * 1.0  # Material weight

    def _piece_activity(self, game):
        """Reward pieces controlling central squares"""
        #activity = 0
        central_squares = [(x,y) for x in range(3,7) for y in range(3,6)]
        
        for x in range(10):
            for y in range(9):
                if piece := game.engine.board[x][y]:
                    # Bonus for central control
                    if (x,y) in central_squares:
                        activity += 2.0 if piece.color == 'red' else -2.0
                    
                    # Mobility bonus
                    moves = len(game.engine.get_possible_moves((x,y)))
                    activity += moves * (5 if piece.color == 'red' else -5)
        return activity * 0.3  # Activity weight

    def _king_safety(self, game):
        """Penalize exposed kings"""
        safety = 0
        for color in ['red', 'black']:
            king_pos = game.general_pos[color]
            if not king_pos:
                continue
                
            # Penalize open files toward king
            x, y = king_pos
            safety += -2.0 if color == 'red' else 2.0  # Base penalty
            
            # Additional penalty if cannon/chariot aligns with king
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                while 0 <= nx < 10 and 0 <= ny < 9:
                    if piece := game.engine.board[nx][ny]:
                        if piece.color != color and piece.name in ['Rook', 'Cannon']:
                            safety += -30 if color == 'red' else 30
                        break
                    nx += dx
                    ny += dy
        return safety * 0.5

    def _soldier_advancement(self, game):
        """Reward advanced soldiers (crossed river)"""
        #bonus = 0
        for x in range(10):
            for y in range(9):
                if (piece := game.engine.board[x][y]) and piece.name == 'Soldier':
                    # Crossed river bonus
                    if piece.color == 'red' and x >= 3:
                        return True
                        #bonus += 1.0
                    elif piece.color == 'black' and x <= 6:
                        return True
                        #bonus += -1.0
        return False
    #bonus * 0.2