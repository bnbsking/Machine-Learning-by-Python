import numpy as np

n = 300
A = np.zeros(n)
A[0]=2; A[1]=5; A[2]=3
B = np.zeros(n)
B[0]=4; B[1]=-2; B[2]=9

for i in range(3,n):
	A[i] = (3*A[i-1] + 1*A[i-2] + 4*A[i-3] + 1*B[i-1] + 5*B[i-2] + 9*B[i-3])%8
	B[i] = (2*A[i-1] + 7*A[i-2] + 1*A[i-3] + 8*B[i-1] + 2*B[i-2] + 8*B[i-3])%8

X, y = [], []
for i in range(300-3):
	X.append( [  [A[i],B[i]],  [A[i+1],B[i+1]],  [A[i+2],B[i+2]]  ] )	
	#X.append( [A[i:i+3], B[i:i+3]] )
	y.append( [A[i+3], B[i+3]] )
X, y = np.array(X), np.array(y)

print(X.shape, y.shape)
