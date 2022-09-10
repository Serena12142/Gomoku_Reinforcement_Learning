import time,copy,random,pickle
import numpy as np
from tensorflow import keras, Graph
from tensorflow.compat.v1 import Session
import tensorflow as tf
from tensorflow.keras.callbacks import History

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
        self.edges={} 
        self.N=0
        self.Q=0
        self.P=prior_probability

    def select(self,c_puct):
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
        for i in range(len(actions)):
            self.edges[actions[i]]=Node(self,probabilities[i])

    def update(self,leaf_value):
        self.N+=1
        self.Q+=(leaf_value-self.Q)/self.N
        if self.parent:
            self.parent.update(-leaf_value)
            
class MCTS():
    def __init__(self,state,policy_value_fn,n_playout=1000,c_puct=5,temp=1):
        self.start_state=state
        self.playout_state=state.make_copy()
        self.root=Node(None,1)
        self.policy_value=policy_value_fn
        self.n_playout=n_playout 
        self.c_puct=c_puct 
        self.temp=temp 

    def get_acts_probs(self):
        actions, probabilities, value=self.policy_value(self.start_state)
        self.root.expand(actions, probabilities)
        self.root.N+=1
        for i in range(self.n_playout):
            self.playout()

        actions=[]
        visits=[]
        for action,node in self.root.edges.items():
            actions.append(action)
            visits.append(node.N)
        total_visits=sum([visit*1.0/self.temp for visit in visits])
        probabilities=[visit*1.0/self.temp/total_visits for visit in visits]

        return actions, probabilities
    
    def playout(self):
        self.playout_state.current_player=self.start_state.current_player
        self.playout_state.next_player=self.start_state.next_player
        for i in range(8):
            for j in range(8):
                self.playout_state.board[i][j]=self.start_state.board[i][j]
                self.playout_state.inputs[0][i][j]=self.start_state.inputs[0][i][j]
                self.playout_state.inputs[1][i][j]=self.start_state.inputs[1][i][j]
        self.playout_state.actions=self.start_state.actions[:]

        state=self.playout_state
        node=self.root
        last_action=None

        while node.edges!={}:
            action,node=node.select(self.c_puct)
            state.take_action(action)
            last_action=action

        actions, probabilities, value=self.policy_value(state)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #value=0 
        
        if len(actions)==0:
            leaf_value=0

        if state.detect_5(state.next_player, last_action//8, last_action%8):
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            leaf_value=-1 #try incrementing this value for training defense? 

        else:
            node.expand(actions, probabilities)
            leaf_value=value

        node.update(-leaf_value)

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

def transpose_board(inputs):
    new_board = [inputs]
    for i in range(3):
        new_board.append(np.zeros((2,8,8)))
    for i in range(3):
        copy_board = copy.deepcopy(new_board[i])
        for j in range(8):
            for k in range(8):
                new_board[i+1][0][k][7-j] = copy_board[0][j][k]
                new_board[i+1][1][k][7-j] = copy_board[1][j][k]
    return new_board
def reflect_board(board):
    reflect = copy.deepcopy(board)
    for i in range(8):
        for j in range(4):
            reflect[0][i][j],reflect[0][i][7-j]=reflect[0][i][7-j],reflect[0][i][j]
            reflect[1][i][j],reflect[1][i][7-j]=reflect[1][i][7-j],reflect[1][i][j]
    return reflect

def transpose_probs(probs):
    new_board = [probs]
    for i in range(3):
        new_board.append(np.zeros(64))
    for i in range(3):
        copy_board = copy.deepcopy(new_board[i])
        for j in range(8):
            for k in range(8):
                new_board[i+1][k*8+(7-j)] = copy_board[j*8+k]
    return new_board

def reflect_prob(board):
    reflect = copy.deepcopy(board)
    for i in range(8):
        for j in range(4):
            reflect[i*8+j],reflect[i*8+(7-j)]=reflect[i*8+(7-j)],reflect[i*8+j]
    return reflect

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
model=keras.models.load_model("checkpoint2-selfplay-3000.h5")
def policy_value_net(state):
    global model
    inputs=np.reshape(state.inputs,(1,2,8,8)).astype(np.float32)
    result=model.predict_on_batch(inputs)
    full_probabilities=result[0][0]
    value=result[1][0][0]
    actions=state.actions
    probabilities=[full_probabilities[action] for action in actions] 
    return actions, probabilities, value

def train(games):
    global model
    game=0
    game_data=[]
    hist=History()
    losses={"total":[],"policy":[],"value":[],"round":[]}
    while game!=games:
        game_data_b=[]
        game_data_w=[]
        board=randomBoard()
        state=Board_state(board,'b')
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        mcts=MCTS(state,policy_value_net,n_playout=512,c_puct=10,temp=1e-3)
        last_action=0
        while len(state.actions)!=0 and not\
              state.detect_5(state.next_player, last_action//8, last_action%8):

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            mcts.start_state=state
            mcts.root=Node(None,1)
            actions, probabilities = mcts.get_acts_probs()

            train_probs=np.zeros(64)
            for i in range(len(actions)):
                train_probs[actions[i]]=probabilities[i]
            data=[copy.deepcopy(state.inputs),
                  train_probs,0]
            if state.current_player=='b':
                game_data_b.append(data)
            else:
                game_data_w.append(data)
            max_prob=0
            max_prob_index=0
            for i in range(len(probabilities)):
                if probabilities[i]>max_prob:
                    max_prob=probabilities[i]
                    max_prob_index=i
            action=actions[max_prob_index]
            state.take_action(action)
            last_action=action
            #print(np.reshape(state.board,(8,8)))

        winlose=False
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if state.detect_5(state.next_player, last_action//8, last_action%8):
            if state.current_player=='b': #black loses
                value=-1.0
                for i in range(len(game_data_b)-1,-1,-1):
                    game_data_b[i][-1]=value
                    #value*=0.95
                value=1.0
                for i in range(len(game_data_w)-1,-1,-1):
                    game_data_w[i][-1]=value
                    #value*=0.95
            else:
                value=1.0
                for i in range(len(game_data_b)-1,-1,-1):
                    game_data_b[i][-1]=value
                    #value*=0.95
                value=-1.0
                for i in range(len(game_data_w)-1,-1,-1):
                    game_data_w[i][-1]=value
                    #value*=0.95
            winlose=True
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #game+=1
        #print(game, winlose, end=" ")
        '''
        if not winlose:
            continue
        '''
        '''
        game_data+=game_data_b+game_data_w
        
    for data in game_data:
        inputs=data[0]
        policy=data[1]
        print(sum(policy))
        value=[data[2]]
        print(inputs)
        print(np.reshape(policy,(8,8)))
        print(value)
        max_prob=0
        max_prob_index=0
        for i in range(len(policy)):
            if policy[i]>max_prob:
                max_prob=policy[i]
                max_prob_index=i
        print(max_prob_index//8,max_prob_index%8)
    print(len(game_data))
    pickle.dump(game_data,open("heuristic_policy_data.pickle",'wb'))
        '''
        #train
        state_inputs=[]
        target_probabilities=[]
        target_values=[]
        for data in game_data_b+game_data_w:
            inputs=data[0]
            possibilities=data[1]
            value=[data[2]]
            #print(inputs)
            #print(possibilities)
            #print(value)
            all_input_rotations=transpose_board(inputs)
            all_prob_rotations=transpose_probs(possibilities)   
            for i in range(4):
                state_inputs.append(copy.deepcopy(all_input_rotations[i]))
                target_probabilities.append(copy.deepcopy(all_prob_rotations[i]))
                target_values.append(value)
                state_inputs.append(reflect_board(all_input_rotations[i]))
                target_probabilities.append(reflect_prob(all_prob_rotations[i]))
                target_values.append(value)
        rng_state = np.random.get_state()
        np.random.shuffle(state_inputs)
        np.random.set_state(rng_state)
        np.random.shuffle(target_probabilities)
        np.random.set_state(rng_state)
        np.random.shuffle(target_values)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #if len(target_values)>100:
        game+=1
        print(game, winlose, len(target_values))
        #else:
        #   continue

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if winlose:
            losses["round"].append(len(target_values))
        else:
            losses["round"].append(480)
        X=np.array(state_inputs[:32])
        y1=np.array(target_probabilities[:32])
        y2=np.array(target_values[:32])
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        model.fit(X,[y1,y2],batch_size=len(X),epochs=1,verbose=1,callbacks=[hist])
        losses["total"].append(hist.history["loss"][0])
        losses["policy"].append(hist.history["dense_loss"][0])
        losses["value"].append(hist.history["dense_2_loss"][0])
        '''
        if winlose:
            model.fit(X,[y1,y2],batch_size=32,epochs=1,verbose=1,callbacks=[hist])
            losses["total"].append(hist.history["loss"][0])
            losses["policy"].append(hist.history["dense_loss"][0])
            losses["value"].append(hist.history["dense_2_loss"][0])
            losses["round"].append(len(X))
        else:            
            model.fit(X,[y1,y2],batch_size=len(X),epochs=1,verbose=1,callbacks=[hist])
            losses["total"].append(hist.history["loss"][0])
            losses["policy"].append(hist.history["dense_loss"][0])
            losses["value"].append(hist.history["dense_2_loss"][0])
            losses["round"].append(480)
        '''
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if game%100==0:
            model.save(f"checkpoint2-selfplay-{3000+game}.h5")
            pickle.dump(losses,open("loss_data.pickle",'wb'))
    model.save(f"checkpoint2-selfplay-{3000+game}.h5")
    pickle.dump(losses,open("loss_data.pickle",'wb'))
    
train(1000)
    
'''
model=keras.models.load_model("checkpoint2-policy-bn.h5")
game_data=pickle.load(open("heuristic_policy_data.pickle",'rb'))
state_inputs=[]
target_probabilities=[]
target_values=[]
for data in game_data:
    inputs=data[0]
    possibilities=data[1]
    value=[data[2]]
    all_input_rotations=transpose_board(inputs)
    all_prob_rotations=transpose_probs(possibilities)   
    for i in range(4):
        state_inputs.append(copy.deepcopy(all_input_rotations[i]))
        target_probabilities.append(copy.deepcopy(all_prob_rotations[i]))
        target_values.append(value)
        state_inputs.append(reflect_board(all_input_rotations[i]))
        target_probabilities.append(reflect_prob(all_prob_rotations[i]))
        target_values.append(value)
rng_state = np.random.get_state()
np.random.shuffle(state_inputs)
np.random.set_state(rng_state)
np.random.shuffle(target_probabilities)
np.random.set_state(rng_state)
np.random.shuffle(target_values)
X=np.array(state_inputs)#[:10000])
y1=np.array(target_probabilities)#[:10000])
y2=np.array(target_values)#[:10000])
model.fit(X,[y1,y2],batch_size=32,epochs=1,verbose=1)
model.save("checkpoint2-policy-round1.pickle.h5")
'''
