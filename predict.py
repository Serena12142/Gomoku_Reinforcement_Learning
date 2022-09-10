import time,copy,random,pickle
import numpy as np
from tensorflow import keras, Graph
from tensorflow.compat.v1 import Session
import tensorflow as tf

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

def getState(board,col):
    inputs=np.zeros((2,8,8))
    for i in range(8):
        for j in range(8):
            if board[i][j]==col:
                inputs[0][i][j]=1
            elif board[i][j]!=' ':
                inputs[1][i][j]=1
    return inputs

#predict
model=keras.models.load_model("checkpoint2-selfplay-bn.h5")
def predict(board,col):
    print_board(board)
    print(col)
    result=model.predict_on_batch(np.reshape(getState(board,col),(1,2,8,8)))
    policy=result[0][0]
    value=result[1][0][0]
    print(np.reshape(policy,(8,8)))
    print(value)
    max_prob=0
    max_prob_index=0
    for i in range(len(policy)):
        if policy[i]>max_prob:
            max_prob=policy[i]
            max_prob_index=i
    print(max_prob_index//8,max_prob_index%8)

for i in range(10):
    predict(randomBoard(random.randint(0,32)),random.choice(['b','w']))
