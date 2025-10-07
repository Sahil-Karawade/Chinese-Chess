from .pieces import Rook, Knight, Elephant, Advisor, General, Cannon, Soldier

def initialize_board():
    board = [[None for _ in range(9)] for _ in range(10)]

    # Black pieces (top side)
    board[0][0] = Rook('black')
    board[0][1] = Knight('black')
    board[0][2] = Elephant('black')
    board[0][3] = Advisor('black')
    board[0][4] = General('black')
    board[0][5] = Advisor('black')
    board[0][6] = Elephant('black')
    board[0][7] = Knight('black')
    board[0][8] = Rook('black')
    board[2][1] = Cannon('black')
    board[2][7] = Cannon('black')
    board[3][0] = Soldier('black')
    board[3][2] = Soldier('black')
    board[3][4] = Soldier('black')
    board[3][6] = Soldier('black')
    board[3][8] = Soldier('black')

    # Red pieces (bottom side)
    board[9][0] = Rook('red')
    board[9][1] = Knight('red')
    board[9][2] = Elephant('red')
    board[9][3] = Advisor('red')
    board[9][4] = General('red')
    board[9][5] = Advisor('red')
    board[9][6] = Elephant('red')
    board[9][7] = Knight('red')
    board[9][8] = Rook('red')
    board[7][1] = Cannon('red')
    board[7][7] = Cannon('red')
    board[6][0] = Soldier('red')
    board[6][2] = Soldier('red')
    board[6][4] = Soldier('red')
    board[6][6] = Soldier('red')
    board[6][8] = Soldier('red')

    return board

