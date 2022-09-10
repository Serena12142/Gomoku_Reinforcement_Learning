import copy,random
import numpy as np

score_model_5_glob= [(10000, (1, 1, 1, 1, 1)), #-1
                (100, (1, 1, 1, 1, 0)), #0, 0
                (100, (0, 1, 1, 1, 1)), #0, 0
                (100, (1, 1, 0, 1, 1)), #0, 0
                (100, (1, 0, 1, 1, 1)), #0, 0
                (100, (1, 1, 1, 0, 1)), #0, 0
                (10, (0, 0, 1, 1, 1)), #1
                (10, (0, 1, 0, 1, 1)), #1
                (10, (0, 1, 1, 0, 1)), #1
                (10, (0, 1, 1, 1, 0)), #1
                (10, (1, 0, 0, 1, 1)), #1
                (10, (1, 0, 1, 0, 1)), #1
                (10, (1, 0, 1, 1, 0)), #1
                (10, (1, 1, 0, 0, 1)), #1
                (10, (1, 1, 0, 1, 0)), #1
                (10, (1, 1, 1, 0, 0)),] #1
score_model_6_glob= [(5000, (0, 1, 1, 1, 1, 0)), #-1, 0
                (100, (0, 1, 1, 1, 0, 0)), #0, 1
                (100, (0, 0, 1, 1, 1, 0)), #0, 1
                (100, (0, 1, 1, 0, 1, 0)), #0, 1
                (100, (0, 1, 0, 1, 1, 0)), #0, 1
                (10, (0, 0, 1, 1, 0, 0)), #1
                (10, (0, 1, 1, 0, 0, 0)), #1
                (10, (0, 0, 0, 1, 1, 0)), #1
                (10, (0, 0, 1, 0, 1, 0)), #1
                (10, (0, 1, 0, 1, 0, 0)), #1
                (1, (0, 0, 0, 1, 0, 0)),  #2
                (1, (0, 0, 1, 0, 0, 0))]  #2
DEBUG=0

########################################## scoring functions ##############################################
def detect_row(board,col, y_start, x_start, d_y, d_x,score_model_5, score_model_6):
    score = 0
    y = y_start
    x = x_start
    y_prev = y - d_y
    x_prev = x - d_x
    y_end_5 = y + 4 * d_y
    x_end_5 = x + 4 * d_x
    y_end_6 = y + 5 * d_y
    x_end_6 = x + 5 * d_x
    y_last = y + 6 * d_y
    x_last = x + 6 * d_x
    # prev, y/x, 2, 3, 4, end_5, end_6, last
    def inbound(y,x):
        return (0 <= y < 8 and 0 <= x < 8)
    seq_5 = [-1]
    for i in range(4):
        if board[y + i * d_y][x + i * d_x] == col:
            seq_5.append(1)
        elif board[y + i * d_y][x + i * d_x] == " ":
            seq_5.append(0)
        else:
            seq_5.append(-1)
    passed=True
    while inbound(y_end_5,x_end_5):
        seq_5 = seq_5[1:]
        if board[y_end_5][x_end_5] == col:
            seq_5.append(1)
        elif board[y_end_5][x_end_5] == " ":
            seq_5.append(0)
        else:
            seq_5.append(-1)
        #only valid if prev is not self and a seqence of self is passed
        if (not inbound(y_prev,x_prev) or board[y_prev][x_prev]!=col) and passed:
            #qualify for 6 - y/x and end_6 is empty and y_last is not self
            if inbound(y,x) and board[y][x]==" " and\
               inbound(y_end_6,x_end_6) and board[y_end_6][x_end_6]==" " and\
               (not inbound(y_last,x_last) or board[y_last][x_last]!=col):
                seq_6 = seq_5[:]
                if board[y_end_6][x_end_6] == col:
                    seq_6.append(1)
                elif board[y_end_6][x_end_6] == " ":
                    seq_6.append(0)
                else:
                    seq_6.append(-1)
                for seq_score, seq in score_model_6:
                    if seq_6 == list(seq):
                        passed=False
                        score += seq_score
            #qualify for 5 - end_6 is not self
            elif not inbound(y_end_6,x_end_6) or board[y_end_6][x_end_6]!=col:
                for seq_score, seq in score_model_5:
                    if seq_5 == list(seq):
                        passed=False
                        score += seq_score
        elif inbound(y_end_6,x_end_6) and board[y_end_6][x_end_6]==col:
            passed=True
        x += d_x
        y += d_y
        y_prev = y - d_y
        x_prev = x - d_x
        y_end_5 = y + 4 * d_y
        x_end_5 = x + 4 * d_x
        y_end_6 = y + 5 * d_y
        x_end_6 = x + 5 * d_x
        y_last = y + 6 * d_y
        x_last = x + 6 * d_x
    return score

def detect_rows(board, col, score_model_5, score_model_6):
    score = 0
    size = 8
    for i in range(size):
        score += detect_row(board, col, 0, i, 1, 0,score_model_5, score_model_6)
        score += detect_row(board, col, i, 0, 0, 1,score_model_5, score_model_6)
    # diagnals
    for i in range(size - 4):
        score += detect_row(board,col, 0, i, 1, 1,
                                 score_model_5, score_model_6)
    for i in range(1, size - 4):
        score += detect_row(board,col, i, 0, 1, 1,
                                 score_model_5, score_model_6)
    for i in range(4, size):
        score += detect_row(board,col, 0, i, 1, -1,
                                 score_model_5, score_model_6)
    for i in range(1, size - 4):
        score += detect_row(board,col, i, size - 1, 1, -1,
                                 score_model_5, score_model_6)
    return score

##################get blank spaces on the board with an order centered around the centroid of the given board##################
def get_legal_move(board):
    mid = 4
    legal_moves = []
    for i in range(mid+1):
        if i == 0:
            points = [[mid, mid]]
        else:
            points = [[mid - i, a] for a in range(mid - i, mid + i)] + \
                     [[b, mid + i] for b in range(mid - i, mid + i)] + \
                     [[mid + i, c] for c in range(mid + i, mid - i, -1)] + \
                     [[d, mid - i] for d in range(mid + i, mid - i, -1)]
        for point in points:
            if 0 <= point[0] < 8 and 0 <= point[1] < 8:
                if board[point[0]][point[1]] == ' ':
                    legal_moves.append(point)

    return legal_moves

################move when there is threat############################
def play(board,col):
    if col == 'b': 
        opponent = 'w'
    else:
        opponent = 'b'
    moves = get_legal_move(board)
    
    # winning threat
    winning_threat = detect_rows(board, col, score_model_5_glob[1:6], score_model_6_glob[0:1])
    if winning_threat > 0:
        if DEBUG:print("winning threat")
        for move in moves:
            board[move[0]][move[1]] = col
            if detect_rows(board, col, score_model_5_glob[0:1], score_model_6_glob[0:0]) > 0:
                return move
            board[move[0]][move[1]] = ' '
    opponent_winning_threat = detect_rows(board,opponent, score_model_5_glob[1:6], score_model_6_glob[0:1])
    if opponent_winning_threat > 0:
        if DEBUG:print("opponent winning threat")
        for move in moves:
            board[move[0]][move[1]] = opponent
            if detect_rows(board, opponent, score_model_5_glob[0:1], score_model_6_glob[0:0]) > 0:
                return move
            board[move[0]][move[1]] = ' '

    # 1-step winning threat
    winning_threat_1 = detect_rows(board, col, score_model_5_glob[0:0], score_model_6_glob[1:5])
    if winning_threat_1 > 0:
        if DEBUG:print("winning threat 1")
        for move in moves:
            board[move[0]][move[1]] = col
            if detect_rows(board, col, score_model_5_glob[0:0], score_model_6_glob[0:1]) > 0:
                return move
            board[move[0]][move[1]] = ' '
    opponent_winning_threat_1 = detect_rows(board,opponent, score_model_5_glob[0:0], score_model_6_glob[1:5])
    if opponent_winning_threat_1 > 0:
        if DEBUG:print("opponent winning threat 1")
        for move in moves:
            board[move[0]][move[1]] = opponent
            if detect_rows(board, opponent, score_model_5_glob[0:0], score_model_6_glob[0:1]) > 0:
                return move
            board[move[0]][move[1]] = ' '

    # 1-step 2 threats
    for move in moves:
        board[move[0]][move[1]] = col
        if detect_rows(board,col,score_model_5_glob[1:6],score_model_6_glob[1:5])>=200:
            if DEBUG:print("1-step 2 threats self",move)
            return move
        board[move[0]][move[1]] = opponent
        if detect_rows(board,opponent,score_model_5_glob[1:6],score_model_6_glob[1:5])>=200:
            if DEBUG:print("1-step 2 threats opp")
            return move
        board[move[0]][move[1]] = ' '
    
    #best score
    best_move=moves[0]
    best_score=0
    for move in moves:
        board[move[0]][move[1]] = col
        score=detect_rows(board,col,score_model_5_glob,score_model_6_glob)-\
               detect_rows(board,opponent,score_model_5_glob,score_model_6_glob)
        board[move[0]][move[1]] = ' '
        if score>best_score:
            best_score=score
            best_move=move
    return best_move

if __name__ == '__main__':
    board = [[' ',' ',' ',' ',' ',' ',' ',' '],
             [' ',' ',' ',' ',' ',' ',' ',' '],
             [' ',' ',' ','w',' ',' ',' ',' '],
             [' ','b','b','b','b','w',' ',' '],
             [' ',' ','b',' ','w',' ',' ',' '],
             [' ',' ',' ',' ',' ',' ',' ',' '],
             [' ',' ',' ',' ',' ',' ',' ',' '],
             [' ',' ',' ',' ',' ',' ',' ',' ']]
    #print(detect_rows(board,'b',score_model_5_glob,score_model_6_glob),
    #      detect_rows(board,'w',score_model_5_glob,score_model_6_glob))
    print(play(board,'w'))

















