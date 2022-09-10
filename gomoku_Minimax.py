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
MAX_SCORE=100000

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

def evaluate(board,root_player,current_player,next_player): 
    current_player_score=detect_rows(board,current_player,
                                     score_model_5_glob,score_model_6_glob)
    next_player_score=detect_rows(board,next_player,
                                  score_model_5_glob,score_model_6_glob)

    score=current_player_score-next_player_score #*0.1

    if root_player==current_player:
        return score
    else:
        return -score

def get_moves(board):
    mid = 4
    legal_moves = []
    for i in range(mid+1):
        empty_cycle = True
        if i == 0:
            points = [[mid, mid]]
        else:
            points=[[mid - i, a] for a in range(mid - i, mid + i)]+\
                    [[b, mid + i] for b in range(mid - i, mid + i)]+\
                    [[mid + i, c] for c in range(mid + i, mid - i, -1)]+\
                    [[d, mid - i] for d in range(mid + i, mid - i, -1)]
        for point in points:
            if 0 <= point[0] < 8 and 0 <= point[1] < 8 and\
               board[point[0]][point[1]] == ' ':
                has_neighbour=False
                for row in range(point[0]-1,point[0]+2):
                    for col in range(point[1]-1,point[1]+2):
                        if 0<=row<8 and 0<=col<8 and\
                           board[row][col]!=' ':
                            has_neighbour=True
                if has_neighbour:
                    legal_moves.append(point)
    #np.random.shuffle(legal_moves)
    return legal_moves

def minimax(board,root_player,current_player,depth,alpha,beta):
    if current_player == 'b':
        next_player = 'w'
    else:
        next_player = 'b'
    if root_player == 'b':
        opponent = 'w'
    else:
        opponent = 'b'
    list_of_moves=get_moves(board)
    
    if detect_rows(board,opponent,score_model_5_glob[:1],[])>0:
        return None,-MAX_SCORE
    if detect_rows(board,root_player,score_model_5_glob[:1],[])>0:
        return None,MAX_SCORE
    if len(list_of_moves)==0:
        return None,0
    if depth == 0:
        return None,evaluate(board,root_player,current_player,next_player)
    
    if current_player!=root_player: #min
        minimum=MAX_SCORE+1
        min_move=None
        for move in list_of_moves:
            board[move[0]][move[1]]=current_player
            _, value=minimax(board,root_player,next_player,
                             depth-1,alpha,beta)
            board[move[0]][move[1]]=' '
            if value<minimum:
                minimum=value
                min_move=move
            if value<beta:
                beta=value
            if alpha>=beta:
                break
        return min_move, minimum #*0.99

    else: #max
        maximum=-MAX_SCORE-1
        max_move=None
        for move in list_of_moves:
            board[move[0]][move[1]]=current_player
            _, value=minimax(board,root_player,next_player,
                             depth-1,alpha,beta)
            board[move[0]][move[1]]=' '
            if value>maximum:
                maximum=value
                max_move=move
            if value>alpha:
                alpha=value
            if alpha>=beta:
                break
        return max_move, maximum #*0.99

def play(board, col):
    move,value = minimax(board,col,col,2,-MAX_SCORE-1,MAX_SCORE+1)
    #print(move, value)
    return move[0],move[1]

if __name__=='__main__':
    board = [[' ',' ',' ',' ',' ',' ',' ',' '],
             [' ',' ',' ',' ',' ',' ',' ',' '],
             [' ',' ','b',' ',' ','b',' ',' '],
             [' ',' ',' ','b','w','b',' ',' '],
             ['w','w',' ','w','w','w',' ',' '],
             [' ',' ','w',' ','w','b',' ',' '],
             [' ','w',' ',' ',' ',' ',' ',' '],
             [' ',' ',' ',' ',' ',' ',' ',' ']]
    print(play(board,'w'))
    

    
