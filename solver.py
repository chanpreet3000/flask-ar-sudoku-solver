def find_empty(board):
    """checkes where is an empty or unsolved block"""
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)

    return None


def valid(board, num, pos):
    # Check row
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if board[i][j] == num and (i, j) != pos:
                return False

    return True

def check(bo):
    for i in range(9):
        map = [0] * 10
        for j in range(9):
            if (bo[i][j] != 0):
                map[bo[i][j]] += 1
        for val in map:
            if (val > 1):
                return False

    for j in range(9):
        map = [0] * 10
        for i in range(9):
            if (bo[i][j] != 0):
                map[bo[i][j]] += 1
        for val in map:
            if (val > 1):
                return False
    return True


def solve(bo):
    for i in range(9):
        for j in range(9):
            if bo[i][j] == 0:
                for num in range(1, 10):
                    if (valid(bo, num, (i, j))):
                        bo[i][j] = num
                        if (solve(bo)):
                            return True
                        bo[i][j] = 0
                return False

    return True


def get_board(bo):
    if (check(bo) and solve(bo)):
        return bo
    else:
        raise ValueError
