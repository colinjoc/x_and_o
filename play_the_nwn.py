"""
A very simple code to play the neural network throuhg terminal
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
	board_c=board
	if board_c[position]!=0:
		#disallowed move
		return -1
	board_c[position]=1
	if check_win(board_c,position)==True:
	
		return 1
	else:
		return 0.5	
def user_print(board):
	# print the board to terminal
	output=[]
	for x in board:
		if x == 1:
			output.append("x")
		elif x ==-1:
			output.append("o")
		else:
			output.append("-")
	print " %s | %s | %s "%(output[0],output[1],output[2])
	print "___________"
	print " %s | %s | %s "%(output[3],output[4],output[5])
	print "___________"
	print " %s | %s | %s "%(output[6],output[7],output[8])

model = keras.models.load_model("single_31600Its_500000Mem.h5") # load neural network
board=[0]*9 # empty board
game = True
left=range(9)
while game:
	print "\n"
	nwn=True
	q_vals = model.predict(np.asarray([board])) # q_vals from neural network
	while nwn:

		q_max = max(q_vals[0]) 
		action_index = np.where(q_vals[0]==q_max)[0][0]
		if board[action_index]==0:
			board[action_index]=1
			if check_win(board,action_index):
				print "NWN wins!"
				user_print(board)
				game = False
				break
			nwn=False
		else:
			q_vals[0][action_index]=-1
		
	user_in = True
	user_print(board)
	left.remove(action_index) 
	if left==[]:
		print "Draw!"
		break
	while user_in and game:
		user_coords=raw_input("Choose a row and column. 0 0 is the top left position\n")
		row = int(user_coords[0])
		col = int(user_coords[1:])
		user_index =3*row+col
		if board[user_index]==0:
			board[user_index]=-1
			user_print(board)			
			user_in = False
		else:
			print "You chose a position that is already taken"
		if check_win(board,user_index):
			"You win!"
			game = False
			break
	if game:
		left.remove(user_index)
