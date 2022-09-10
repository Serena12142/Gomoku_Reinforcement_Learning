import numpy as np
import copy,random,math

class Board_state():
    def __init__(self,board,col,actions=None,inputs=None):
        self.board=board
        self.current_player=col
        if self.current_player=='b':
            self.next_player='w'
        else:
            self.next_player='b'

        if actions!=None:
            self.actions=actions
            self.inputs=inputs
        else:
            self.actions=[]
            self.inputs=np.zeros((2,8,8))
            for i in range(8):
                for j in range(8):
                    if board[i][j]==' ':
                        self.actions.append(i*8+j)
                    elif board[i][j]==self.current_player:
                        self.inputs[0][i][j]=1
                    else:
                        self.inputs[1][i][j]=1

    def is_5(self,board,col, start_y,start_x, dir_y,dir_x):
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
        
    def detect_5(self, col, y, x):
        for dir_y,dir_x in [(0,1),(1,0),(1,1),(1,-1)]:
            for i in range(5):
                start_y=y-dir_y*i
                start_x=x-dir_x*i
                if self.is_5(self.board,col, start_y,start_x, dir_y,dir_x):
                    return True
        return False

    def take_action(self,action):
        self.board[action//8][action%8]=self.current_player
        temp=self.next_player
        self.next_player=self.current_player
        self.current_player=temp

        self.inputs[[0, 1]] = self.inputs[[1, 0]]
        self.inputs[1][action//8][action%8]=1

        self.actions.remove(action)
        
    def make_copy(self):
        return Board_state(copy.deepcopy(self.board),self.current_player,
                           actions=self.actions[:],
                           inputs=copy.deepcopy(self.inputs))
    
class Node():
    def __init__(self,parent,prior_probability):
        self.parent=parent
        self.edges={} # action: child node
        self.N=0
        self.Q=0
        self.P=prior_probability

    def select(self,c_puct):
        #return child node that maximize UCB
        maximum=-np.inf
        best_node=None
        best_action=None
        for action,node in self.edges.items():
            U=c_puct*node.P*np.sqrt(self.N)/(1+node.N)
            j=node.Q+U
            if j>maximum:
                maximum=j
                best_node=node
                best_action=action
        return best_action,best_node

    def expand(self,actions,probabilities):
        #generate all child nodes
        for i in range(len(actions)):
            self.edges[actions[i]]=Node(self,probabilities[i])

    def update(self,leaf_value):
        self.N+=1
        self.Q+=(leaf_value-self.Q)/self.N
        if self.parent:
            self.parent.update(-leaf_value)
            
class MCTS():
    def __init__(self,state,policy_value_fn,n_playout=1000,c_puct=5):
        self.start_state=state
        self.playout_state=state.make_copy()
        self.root=Node(None,1)
        self.policy_value=policy_value_fn
        self.n_playout=n_playout # playouts per decision
        self.c_puct=c_puct # level of exploration, c > 0

    def get_acts_probs(self):
        #expand the root
        actions, probabilities, value=self.policy_value(self.start_state)
        self.root.expand(actions, probabilities)
        
        #playouts
        for i in range(self.n_playout):
            self.playout()

        # calculate probabilities (pi) based on N (visit count)
        actions=[]
        visits=[]
        for action,node in self.root.edges.items():
            actions.append(action)
            visits.append(node.N)
        total_visits=sum(visits)
        probabilities=[visit/total_visits for visit in visits]

        return actions, probabilities
    
    def playout(self):
        # copy manually to avoid __init__ and deepcopy
        self.playout_state.current_player=self.start_state.current_player
        self.playout_state.next_player=self.start_state.next_player
        for i in range(8):
            for j in range(8):
                self.playout_state.board[i][j]=self.start_state.board[i][j]
                self.playout_state.inputs[0][i][j]=self.start_state.inputs[0][i][j]
                self.playout_state.inputs[1][i][j]=self.start_state.inputs[1][i][j]
        self.playout_state.actions=self.start_state.actions[:]
        # start at root
        state=self.playout_state
        node=self.root
        last_action=None
        # greedily go down the tree until node is leaf (has no child)
        while node.edges!={}:
            action,node=node.select(self.c_puct)
            state.take_action(action)
            last_action=action
        # evaluate the leaf using policy value function
        actions, probabilities, value=self.policy_value(state)
        #value=0
        
        #draw
        if len(actions)==0:
            leaf_value=0
        #last move wins
        if state.detect_5(state.next_player, last_action//8, last_action%8):
            leaf_value=-1
        #continue
        else:
            #expland the tree
            node.expand(actions, probabilities)
            leaf_value=value

        # update Q and N values recursively
        node.update(-leaf_value)

import tensorflow as tf
from tensorflow import keras,Graph
from tensorflow.compat.v1 import Session

name="checkpoint2-selfplay-3000.h5"
graph=Graph()
with graph.as_default():
    session=Session()
    with session.as_default():
        model = keras.models.load_model(name)
def setModel(name):
    global graph,session,model
    with graph.as_default():
        session=Session()
        with session.as_default():
            model = keras.models.load_model(name)
def policy_value_net(state):
    global graph,session,model
    inputs=np.reshape(state.inputs,(1,2,8,8)).astype(np.float32)
    with graph.as_default():
        with session.as_default():
            result=model.predict_on_batch(inputs)
    full_probabilities=result[0][0]
    value=result[1][0][0]
    actions=state.actions
    probabilities=[full_probabilities[action] for action in actions] 
    return actions, probabilities, value
board = [[' ']*8 for i in range(8)]
state=Board_state(board,'b')
policy_value_net(state)

def play(board, col):
    mcts=MCTS(Board_state(board,col),policy_value_net,
              n_playout=256,c_puct=5)
    
    actions,probabilities=mcts.get_acts_probs()

    #greedy: choose max probability action
    max_prob=0
    max_prob_index=0
    for i in range(len(probabilities)):
        if probabilities[i]>max_prob:
            max_prob=probabilities[i]
            max_prob_index=i
    action=actions[max_prob_index]
    
    return action//8,action%8

if __name__=='__main__':
    board = [[' ',' ',' ',' ',' ',' ',' ',' '],
             [' ',' ',' ','b',' ',' ',' ',' '],
             [' ','w',' ',' ',' ',' ',' ',' '],
             [' ',' ',' ',' ',' ',' ',' ',' '],
             [' ',' ',' ',' ',' ',' ',' ',' '],
             [' ',' ','b',' ',' ',' ',' ',' '],
             [' ',' ',' ',' ','w','w',' ',' '],
             [' ',' ',' ',' ',' ',' ',' ',' ']]
    play(board,'b')
    #current_state=Board_state(board,'b')
    #print(current_state.score())
