class Piece:
    def __init__(self, name, color):
        self.name = name  # e.g., 'General', 'Rook', 'Knight', 'Cannon', etc.
        self.color = color  # 'red' or 'black'

    def __repr__(self):
        return f"{self.color[0].upper()}{self.name}"
    
class Soldier(Piece):
    def __init__(self, color):
        super().__init__('Soldier', color)


    def is_valid_move(self, start, end, board):
        sx, sy = start  # (row, col)
        ex, ey = end    # (row, col)
        dx, dy = ex - sx, ey - sy

        # Soldiers move forward (Black: +1 row; Red: -1 row)
        forward = 1 if self.color == 'black' else -1

        # Before crossing the river: only forward
        if (self.color == 'black' and sx < 5) or (self.color == 'red' and sx > 4):
            return (dx, dy) == (forward, 0)
        # After crossing the river: forward or sideways
        else:
            return (dx, dy) in [(forward, 0), (0, 1), (0, -1)]
  
class Cannon(Piece):
    def __init__(self, color):
        super().__init__('Cannon', color)

    def is_valid_move(self, start, end, board):
        sx, sy = start  # (row, col)
        ex, ey = end    # (row, col)
        dx, dy = ex - sx, ey - sy

        # Cannons move orthogonally (row or column changes, not both)
        if not (dx == 0 or dy == 0):
            return False

        # Check if target is occupied by a friendly piece
        if board[ex][ey] is not None and board[ex][ey].color == self.color:
            return False

        # Determine direction (step for row/col)
        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

        # Check path for screens (capture) or clear path (normal move)
        is_capture = board[ex][ey] is not None  # Target has an enemy piece
        screen_count = 0

        x, y = sx + step_x, sy + step_y
        while (x != ex or y != ey) and 0 <= x < 10 and 0 <= y < 9:
            if board[x][y] is not None:  # Found a piece in the path
                screen_count += 1
            x += step_x
            y += step_y

        # Validate move
        if is_capture:
            return screen_count == 1  # Must jump exactly one screen to capture
        else:
            return screen_count == 0  # No pieces in path for normal moves

class Rook(Piece):
    def __init__(self, color):
        super().__init__('Rook', color)


    def is_valid_move(self, start, end, board):
        sx, sy = start  # (row, col)
        ex, ey = end    # (row, col)
        dx, dy = ex - sx, ey - sy

        # Rook move orthogonally (row or column changes, not both)
        if not (dx == 0 or dy == 0):
            return False

        # Check if target is occupied by a friendly piece
        if board[ex][ey] is not None and board[ex][ey].color == self.color:
            return False

        # Determine direction (step for row/col)
        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

        # Check if path is clear
        x, y = sx + step_x, sy + step_y
        while (x != ex or y != ey) and 0 <= x < 10 and 0 <= y < 9:
            if board[x][y] is not None:
                return False  # Path blocked
            x += step_x
            y += step_y
        return True
    
class General(Piece):
    def __init__(self, color):
        super().__init__('General', color)


    def is_valid_move(self, start, end, board):
        sx, sy = start  # (row, col)
        ex, ey = end    # (row, col)
        dx, dy = ex - sx, ey - sy

        # Generals move one step orthogonally within the palace
        if not ((abs(dx) == 1 and dy == 0) or (dx == 0 and abs(dy) == 1)):
            return False

        # Palace boundaries (columns 3-5, rows 0-2 for Black; rows 7-9 for Red)
        if self.color == 'black':
            valid_palace = (0 <= ex <= 2) and (3 <= ey <= 5)
        else:  # Red
            valid_palace = (7 <= ex <= 9) and (3 <= ey <= 5)

        if not valid_palace:
            return False
        # Added this check to General's is_valid_move
        target = board[ex][ey]
        if target is not None and target.color == self.color:
            return False
        # Optional: Check for "Flying General" rule (Generals cannot face each other directly)
        # (Implement this separately in game logic)
        return True

class Knight(Piece):
    def __init__(self, color):
        super().__init__('Knight', color)

    def is_valid_move(self, start, end, board):
        x1, y1 = start  # Current position (row, col)
        x2, y2 = end    # Target position (row, col)
        dx, dy = x2 - x1, y2 - y1

        # All possible Horse moves (dx, dy)
        possible_moves = [
            (1, 2), (1, -2), (-1, 2), (-1, -2),
            (2, 1), (2, -1), (-2, 1), (-2, -1)
        ]

        # Check if the move is one of the allowed L-shapes
        if (dx, dy) not in possible_moves:
            return False

        # Check hobbling (blocking "foot" position)
        if abs(dx) == 1:  # Horizontal step first (dx=±1, dy=±2)
            hobble_x, hobble_y = x1, y1 + (dy // 2)
        else:  # Vertical step first (dx=±2, dy=±1)
            hobble_x, hobble_y = x1 + (dx // 2), y1

        # Ensure hobble square is unblocked
        if not (0 <= hobble_x < 10 and 0 <= hobble_y < 9):
            return False  # Hobble square out of bounds
        if board[hobble_x][hobble_y] is not None:
            return False  # Hobbled!

        # Check if target is empty or enemy
        if board[x2][y2] is None or board[x2][y2].color != self.color:
            return True

        return False

class Elephant(Piece):
    def __init__(self, color):
        super().__init__('Elephant', color)
        
    def is_valid_move(self, start, end, board):
        sx, sy = start
        ex, ey = end
        
        # Must move exactly 2 squares diagonally
        if abs(ex - sx) != 2 or abs(ey - sy) != 2:
            return False
            
        # Check river crossing restriction
        if self.color == 'red' and ex < 5:  # Red can't cross to x < 5
            return False
        if self.color == 'black' and ex > 4:  # Black can't cross to x > 4
            return False
            
        # Check blocking piece in the middle
        mid_x = (sx + ex) // 2
        mid_y = (sy + ey) // 2
        if board[mid_x][mid_y] is not None:
            return False
            
        # Check target square (must be empty or opponent)
        target = board[ex][ey]
        if target is not None and target.color == self.color:
            return False
            
        return True

class Advisor(Piece):
    def __init__(self, color):
        super().__init__('Advisor', color)
        
    def is_valid_move(self, start, end, board):
        sx, sy = start
        ex, ey = end
        
        # Must move exactly 1 square diagonally
        if abs(ex - sx) != 1 or abs(ey - sy) != 1:
            return False
            
        # Must stay within the palace
        if self.color == 'red':
            palace_condition = (ex >= 7 and ex <= 9 and ey >= 3 and ey <= 5)
        else:  # black
            palace_condition = (ex >= 0 and ex <= 2 and ey >= 3 and ey <= 5)
            
        if not palace_condition:
            return False
            
        # Check target square (must be empty or opponent)
        target = board[ex][ey]
        if target is not None and target.color == self.color:
            return False
            
        return True