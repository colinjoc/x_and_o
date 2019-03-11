# -*- coding: utf-8 -*-
"""
About:

Train a neural network to play x's and o's (or tic-tac-toe).

Used the following article to learn and implement the algorithm: https://www.nervanasys.com/demystifying-deep-reinforcement-learning/

"""

from keras.models import Sequential
import keras
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from math import sin
from itertools import product,permutations
import random

def check_win(board,position):
	# this funciton checks if the position and board layout suppplied results in a win
	board_c = [board[0:3],board[3:6],board[6:9]]
	position = [position/3,position%3]

	pos = position
	value = board_c[pos[0]][pos[1]]
	#check row
	if board_c[pos[0]]==[value]*3:
		#win!
		return True
	elif [board_c[0][pos[1]],board_c[1][pos[1]],board_c[2][pos[1]]]==[value]*3:
		#win!
		return True
	elif [board_c[0][0],board_c[1][1],board_c[2][2]]==[value]*3 or [board_c[0][2],board_c[1][1],board_c[2][0]]==[value]*3:
		#win!
		return True
	else:
		return False

def position_check(board,position):
	# check the position against the board and return if it is allowed and if it is a winning move
	board_c=board
	if board_c[position]!=0:
		#disallowed move
		return -1
	board_c[position]=1
	if check_win(board_c,position)==True:
	
		return 1
	else:
		return 0.5	


# create keras neural network. 
model = Sequential()
model.add(Dense(18, input_dim=9, init='uniform',bias=True, activation='softsign')) # 18 node layer connected to 9 input nodes. Activation function in softsign
model.add(Dense(9, init='uniform',bias=True, activation='softsign')) # 9 node layer. Activation function in softsign
model.compile(loss='mse',optimizer='sgd', metrics=['mean_squared_error']) # compile the neural network. use mean squared error as the loss function

#model = keras.models.load_model("single_31600Its_500000Mem.h5") # load a previously trained neural network to continue training. Uncomment this and comment out lines 53-56 if you want to use this



# -------------- some parameters ------------------
gamma = 1.0 # discounted future reward. see linked article
number_its=10000000 # number of games you want to play while training

epsilon_check=0.25 # exploration fraction. probability that the neural network picks a random move. see linked article


x_list=range(number_its)
win_list=[None]*number_its # record wins
memory=[] # this list is going to store old moves
memory_out=[]

for i in range(number_its):
	board = list([0]*9) # define the board. 0  is empty. 1 is x (played by the neural network) and -1 is the oponent.
	left = range(9) # list of indexes of empty positions
	game = True # game is running
	while game:
		next_state=board[:] # copy the board. use this when checking Q values 
		q_vals = model.predict(np.asarray([board])) # use neural network to calculate Q values. read linked article for more (top)
		q_max = max(q_vals[0]) 
		# random move 
		epsilon = random.random() 
		if epsilon>epsilon_check:
			action_index = np.where(q_vals[0]==q_max)[0][0]

		else:
			action_index=left[random.randint(0,len(left)-1)]

		# max q value from next state
		next_max_q=0; 
		
		for coord in left: # go through all available coordinates
			next_state[coord]=1 # set to 1
			next_q_vals = model.predict(np.asarray([next_state]))[0] # get q values if this coordinate is chosen
			if max(next_q_vals)>next_max_q:
				next_max_q =max(next_q_vals) # save next_max_q
			next_state[coord]=0	# reset to 0
		
		# next_q_vals[1] is now the max Q(s',a') in linked article
	 
		r = position_check(next_state,action_index) # reward for this move. action index is the index of this turn.
		
		if r==0.5: 
			# ok move
			next_state[action_index]=1 # make the move at action index
			left.remove(action_index) # remove the index from list of free indices
			if left == []:
				# draw game, all moves used up	
				game = False # this game is over
				win_list[i]=0

			if game:
				# now for the computers turn
				vs_index=random.randint(0,len(left)-1) # choose a random playable index
				vs_coord=left[vs_index] # choose a random playable index
				left.remove(vs_coord) #remove it
				next_state[vs_coord]=-1 # play that coordinate
				if check_win(next_state,vs_coord)==True and game:
					# if computer wins, update the reward for the neural network to -1
					r = -1
					game = False
					win_list[i]=-1
			
				if left == [] and r==0.5:
					# draw game
					game = False	
					win_list[i]=0
				
		elif r==1:		
			# win. Weeeey!
			next_state[action_index]=1
			
			win_list[i]=1
			game=False
		else:
			# illegal move. 
			next_state[action_index]=0
			game=False


		if len(memory)>500000: # memory full! Replace a random game memory with this move	
			replace_index=random.randint(0,len(memory)-1)
			memory[replace_index]=[board[:],action_index,r,next_state[:],q_vals[:]]
		else:	
			memory.append([board[:],action_index,r,next_state[:],q_vals[:]]) # record the board layout, the move for the neural network, reward, next state and q_vals
		board=next_state[:]# update the board for the next turn id game== True still
		
		if i>10: # if i>10, start training the neural network	
			random_indices= random.sample(range(len(memory)),min([int(len(memory)*0.25),1000])) # pick a number of random memories to use to train the neural network.
			random_in=[] # list of input states to be fed into the neural network for training
			random_out=[] # list of matching output values we want our neural network to return
			for j in random_indices:
				# sort out the variable stored in the memory
				old_state=memory[j][0]
				action=memory[j][1]
				reward=memory[j][2]
				new_state=memory[j][3]
				old_q_vals=memory[j][4]
				if reward==1 or reward==-1: # win/lose, dont need to check next states
					target= reward 
				else:
					remain = [x for x in range(len(new_state)) if new_state[x]==0] # empty sites
					next_max_q=0; 
					# calculate the total reward that should be produced for this state
					for coord in remain:
						next_state[coord]=1
						next_q_vals = model.predict(np.asarray([next_state]))[0]
						if max(next_q_vals)>next_max_q:
							next_max_q =max(next_q_vals)
						next_state[coord]=0	
					target = reward+gamma*next_max_q 

				#make the out matrix!
				output = old_q_vals[0]
				output[action]=target # want the neural network to calculate this Q value for this action
				random_in.append(old_state) 
				random_out.append(output)
			# now train the neural network to return the desired random_out matrix when random_in is input
			if i%1000==0:
				print "iteration",i
				model.fit(np.asarray(random_in), np.asarray(random_out), nb_epoch=100, batch_size=int(len(memory)*0.25))#,verbose=0) # this prints the output of training this iteration
			else:
				model.fit(np.asarray(random_in), np.asarray(random_out), nb_epoch=100, batch_size=int(len(memory)*0.25),verbose=0) # training with no output
		
print "number_its: %s, epsilon: %s, Win %s, draw %s, lost %s" %(number_its,epsilon_check,float(len([x for x in win_list if x==1]))/number_its,float(len([x for x in win_list if x==0]))/number_its,float(len([x for x in win_list if x==-1]))/number_its) # print a few updates to the screen so we can get an idea how the neural network is performing















