import random,copy

def print_board(board):
    s = "*"
    for i in range(len(board[0])-1):
        s += str(i%10) + "|"
    s += str((len(board[0])-1)%10)
    s += "*\n"
    for i in range(len(board)):
        s += str(i%10)
        for j in range(len(board[0])-1):
            s += str(board[i][j]) + "|"
        s += str(board[i][len(board[0])-1]) 
        s += "*\n"
    s += (len(board[0])*2 + 1)*"*"
    print(s)

def randomBoard(randomSteps=2):
    board = [[' ']*8 for _ in range(8)]
    y=-1 
    x=-1
    for _ in range(randomSteps):
        while not (0 <= y < 8 and 0 <= x < 8 and board[y][x]==' '):
            y=random.randint(0,7)
            x=random.randint(0,7)
        board[y][x]='b'
        while not (0 <= y < 8 and 0 <= x < 8 and board[y][x]==' '):
            y=random.randint(0,7)
            x=random.randint(0,7)
        board[y][x]='w'
    return board

def is_full(board):
    for row in board:
        for cell in row:
            if cell==" ":
                return False
    return True

def is_5(board,col, start_y,start_x, dir_y,dir_x):
    for i in range(5):
        if not (0 <= start_y+i*dir_y < 8 and 0 <= start_x+i*dir_x < 8) or\
           board[start_y+i*dir_y][start_x+i*dir_x]!=col:
            return False
    if 0 <= start_y-dir_y < 8 and 0 <= start_x-dir_x < 8 and\
       board[start_y-dir_y][start_x-dir_x]==col:
        return False
    if 0 <= start_y+5*dir_y < 8 and 0 <= start_x+5*dir_x < 8 and\
       board[start_y+5*dir_y][start_x+5*dir_x]==col:
        return False
    return True
    
def detect_5(board, col, y, x):
    for dir_y,dir_x in [(0,1),(1,0),(1,1),(1,-1)]:
        for i in range(5):
            start_y=y-dir_y*i
            start_x=x-dir_x*i
            if is_5(board,col, start_y,start_x, dir_y,dir_x):
                return True
    return False            

def human_input(board,col):
    move_y=-1
    move_x=-1
    while move_y<0 or move_y>=8 or move_x<0 or move_x>=8 or\
          board[move_y][move_x]!=' ': 
        print(f"Your move ({col}):")
        move_y = int(input("y coord: "))
        move_x = int(input("x coord: "))
    return move_y,move_x

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.move_y = None
            self.move_x = None
            
        def run(self):
            self.move_y,self.move_x = func(*args, **kwargs)

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        print("TIMEOUT")
        return None
    else:
        return it.move_y,it.move_x

import time     
def play_gomoku(ai1,ai2,games,display=True,timeout_duration=-1):
    ai1_score=[0 for i in range(games)]
    ai2_score=[0 for i in range(games)]
    ai1_times=[0.0 for i in range(games)]
    ai2_times=[0.0 for i in range(games)]
    for game in range(games):
        start_board = randomBoard()
        if display:
                print_board(start_board)
                print('-'*50)
                
        col='b'
        current_player=ai1
        board = copy.deepcopy(start_board)
        while True:
            if timeout_duration!=-1:
                move_y, move_x = timeout(current_player,
                                         (copy.deepcopy(board), col),{},
                                         timeout_duration,(-1,-1))
            else:
                start=time.time()
                move_y, move_x = current_player(copy.deepcopy(board),col)
                time_used=time.time()-start
                if col=='b':
                    ai1_times[game]=time_used
                else:
                    ai2_times[game]=time_used

            if 0 <= move_y < 8 and 0 <= move_x < 8 and\
               board[move_y][move_x]==' ':
                board[move_y][move_x]=col
            else:
                print("INVALID MOVE")
                if col=='b':
                    ai1_score[game]=-1
                else:
                    ai1_score[game]=1
                break

            if display:
                print_board(board)
            
            if detect_5(board, col, move_y, move_x):
                if col=='b':
                    ai1_score[game]=1
                    if display:
                        print("black won")
                else:
                    ai1_score[game]=-1
                    if display:
                        print("white won")
                break
            elif is_full(board):
                if display:
                    print("draw")
                break
                 
            if col=='b':
                col='w'
                current_player=ai2
            else:
                col='b'
                current_player=ai1

        # --------------------------------- change player ---------------------------------
        
        col='b'
        current_player=ai2
        board = copy.deepcopy(start_board)
        while True:
            if timeout_duration!=-1:
                move_y, move_x = timeout(current_player,
                                         (copy.deepcopy(board), col),{},
                                         timeout_duration,(-1,-1))
            else:
                start=time.time()
                move_y, move_x = current_player(copy.deepcopy(board),col)
                time_used=time.time()-start
                if col=='w':
                    ai1_times[game]=time_used
                else:
                    ai2_times[game]=time_used

            if 0 <= move_y < 8 and 0 <= move_x < 8 and\
               board[move_y][move_x]==' ':
                board[move_y][move_x]=col
            else:
                print("INVALID MOVE")
                if col=='b':
                    ai2_score[game]=-1
                else:
                    ai2_score[game]=1
                break

            if display:
                print_board(board)
            
            if detect_5(board, col, move_y, move_x):
                if col=='b':
                    ai2_score[game]=1
                    if display:
                        print("black won")
                else:
                    ai2_score[game]=-1
                    if display:
                        print("white won")
                break
            elif is_full(board):
                if display:
                    print("draw")
                break
                 
            if col=='b':
                col='w'
                current_player=ai1
            else:
                col='b'
                current_player=ai2
        
    if games<=10:
        print(ai1_score)
        print(ai2_score)
    print(sum(ai1_score),sum(ai2_score))
    print(min(ai1_times),sum(ai1_times)/len(ai1_times),max(ai1_times))
    print(min(ai2_times),sum(ai2_times)/len(ai2_times),max(ai2_times))

import gomoku_MCTS_pvnet1, gomoku_MCTS_pvnet2
def evaluate(model1,model2,games):
    gomoku_MCTS_pvnet1.setModel(model1)
    gomoku_MCTS_pvnet2.setModel(model2)
    ai1=gomoku_MCTS_pvnet1.play
    ai2=gomoku_MCTS_pvnet2.play
    ai1_score=[0 for i in range(games)]
    ai2_score=[0 for i in range(games)]
    for game in range(games):
        start_board = randomBoard()              
        col='b'
        current_player=ai1
        board = copy.deepcopy(start_board)
        while True:
            move_y, move_x = current_player(copy.deepcopy(board),col)
            board[move_y][move_x]=col
            #print_board(board)
            
            if detect_5(board, col, move_y, move_x):
                if col=='b':
                    ai1_score[game]=1
                else:
                    ai1_score[game]=-1
                break
            elif is_full(board):
                break
                 
            if col=='b':
                col='w'
                current_player=ai2
            else:
                col='b'
                current_player=ai1

        # --------------------------------- change player ---------------------------------
        
        col='b'
        current_player=ai2
        board = copy.deepcopy(start_board)
        while True:
            move_y, move_x = current_player(copy.deepcopy(board),col)
            board[move_y][move_x]=col
            #print_board(board)
             
            if detect_5(board, col, move_y, move_x):
                if col=='b':
                    ai2_score[game]=1
                else:
                    ai2_score[game]=-1 
                break
            elif is_full(board):
                break
                 
            if col=='b':
                col='w'
                current_player=ai1
            else:
                col='b'
                current_player=ai2

    return sum(ai1_score),sum(ai2_score)

    
if __name__ == '__main__':
    
    import gomoku_Score, gomoku_Minimax
    import gomoku_MCTS_pvnet1, gomoku_MCTS_pvnet2
    ai1=gomoku_Score.play
    ai2=gomoku_Minimax.play
    play_gomoku(ai1,ai2,10,display=True,timeout_duration=-1)
    '''
    print(evaluate("checkpoint2-selfplay-3000.h5","model-oldbest.h5",10))
    print(evaluate("checkpoint2-selfplay-3000.h5","checkpoint2-selfplay-2700.h5",10))
    '''











