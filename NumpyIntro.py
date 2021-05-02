import numpy as np 
import copy
import math 
from datetime import *
# # Numpy is a data science module 
# # Numpy is multi dimensional array libary
# # Numpy array is faster than python built in list
# # Numpy arrays have less bytes than list

# a = np.array([1,2,3],dtype = "int8")
# b = np.array([2,4,6])
# c = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(a)
# print(b)
# print(a*b)

# # get dimensions of array
# print(a.ndim)
# print(b.ndim)
# print(c.ndim)

# # Get shape of array (rows,columns)

# print(a.shape) # (3,) means 0 rows and three columns
# print(b.shape)
# print(c.shape)

# # get type
# print(a.dtype) # defualt of int 32 which is = 4 bytes
# print(a.itemsize)


# Manipulating arrays (rows,columns)

z = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(z)
print()
print(z[:,:])
print(z[0,1:3])
print(z[0,4:7:2])
print(z[:,0:7:2])


# accessing and chnaging data in array

z[:,0:7:2] = 9 # all rows and each second column replaced by a 9
print(z)

z[:,2] = [13,15] # all rows each number on the second column replace with a 13 and 15
print(z)

# 3d example

y = np.array([[[1,2],[3,4],[5,6],[7,8]]])

y[0,1,1] = 90
print(y)
print(y.ndim)

y[0,:,1] = 88
print(y)

y[0,:,:] = 0
print(y)

p = np.array([[[1,2,3],[4,5,6],[7,8,9]]])
print(p)
p[0,0:3:2,1] = [90,99]
print(p)

#initialising arrays

zeros = np.zeros([3,4,3])
print(zeros)
print("Zeros has " + " " + str(zeros.ndim) + " " + "has dimensions")

ones = np.ones([2,2,3])
print(ones)
print("ones has " + " " + str(ones.ndim) + " " + "has dimensions")


l = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[15,16,17]]])
p = np.array([[[20,19,18],[17,16,15],[14,13,12]],[[11,10,9],[8,7,6],[5,4,3]]])
print(l.ndim)
print(l.shape)
o = l*p
print(o.ndim)

arr1 = np.array([[[1,2,3],[4,5,6],[7,8,9]]])
print(arr1.ndim)
print(arr1.shape)

arr2 = np.array([[[2,4,6],[8,10,12],[14,16,18]]])
print(arr2.shape)
print(arr2.ndim)

arr3 = arr1 * arr2
print(arr3)

arr4 = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr5 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr4)
print(arr4.ndim)
print(arr4.shape)
print(arr5.ndim)

print(arr4*arr5) # array mutplication is not the same as matrix mutiplication 

arry = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]])
arrz = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]])
print(arry.ndim)
print(arry.shape)

print(arry * arrz)

oness = np.ones([4,2,2])
print()
print(oness)

## array full of number
print(np.full([4,2,2],99))

## or we can use pre determined array shapes

print(np.full(z.shape, 99)) ## in this case we use the z array shape and fil with the integer 99

print(np.random.rand(2,5)) # array with random decimal with 2 rows and 5 columns

## we can initialise an array with random integer values

print(np.random.randint(0,6,[1,2,2])) # range of random number followed by shape(in this a 3d shape of 2 by 2)

print(np.identity(5)) ## Produces an identity matrix of matrix size n, in this case 5 by 5 matrix

## Repeating an array mutiple times

arrj = np.array([[1,2,3]])

print(np.repeat(arrj,3,axis = 0))

identity = np.identity(6)
print(identity)
print(np.repeat(identity,3,axis = 1))


abc = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(abc.shape)
print(np.repeat(abc,2,axis = 1))

ee = np.array([[[1,2,3],[3,4,5]],[[6,7,8],[9,10,11]]])
print(ee.shape)
print(ee)


print(ee)
print("Hello")

ej = ee[0:2,0:2,0:3]
print('Yess?')
print(ej)

arrayb = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]])
arrayb[0:2,0:3,0:3] = [[[2,4,6],[8,10,12],[14,16,18]],[[20,22,24],[26,28,30],[32,34,36]]]

print(arrayb)
print(ej==ee)

# changing an array from one to another
initial = np.ones([5,5])
initial[1:4,1:4] = [[0,0,0],[0,9,0],[0,0,0]]
print(initial)

# 3d example

onesss = np.ones([2,2,3])
print(onesss)

## add [9,9,9] on each last row of each item in 3d array

onesss[0:2,1,0:3] = [9,9,9]
print(onesss)

arrayOnes = np.ones([5,5])
print("5 by 5 array")
print(arrayOnes)

arrayOnes[0,1:4:2] = [0]
arrayOnes[4,1:4:2] = [0]
arrayOnes[1:4:2,0:5:4] = [0]
print()
print(arrayOnes)

copied_array = copy.deepcopy(arrayOnes)
print("Copied array")
print(copied_array) 

#mathematics in numpy

a = np.array([90,2,3])
b = np.array([4,5,6])
trig = np.array([math.pi,math.pi/2,2*math.pi])

print()
print(b.ndim)
print(a*b)
print(b**2)
## sin and cos require angles in radains
print(np.sin(trig))

## Algebra


arr1 = np.array([[1,2,3],[4,4,3],[7,8,3]])
# print(arr1.shape)
# arr2 = np.array([[1,2,3],[1,2,3]])
# print(np.matmul(arr1,arr2)) # takes arguments of matrix (matrix must agree with each other to multiply) 

# for matrix to be able to multiply the columns of first matrix must be equal to rows of second

# determinent of matrix
print(np.linalg.det(arr1))
print(np.linalg.eig(arr1))
print()
print(np.linalg.inv(arr1))
print(np.linalg.matrix_power(arr1,2))

# Statistics

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a.sum()) # will summate all numbers within array
# we can also specify the axis direction we want the sum to be calculated
print(a.sum(axis = 0)) # adds 2d matrix downwards in the rows - produces matrix
print(a.sum(axis =1)) # adds 2d matrix on a flat plane - produces a matrix

# Reorganising array 
a = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]) # 3d array with each item having 2 by 3 shape
print(a.reshape([1,4,3])) # changes 3d shape into one item with 4 by 1 shape
print(a.reshape([4,1,3]))

b = np.array([[1,2,3],[50,40,20]])
b = b.reshape([3,2])
print(b)
print(b.sum(axis = 1))

b = b.reshape([2,1,3])
print(b)
print(b.ndim)
print(b.shape)

# Vertically stacking
# Two seperate arrays can be stacked vertically to produce one main array

v1 = np.array([[1,2,3],[4,5,6]])
v2 = np.array([[2,4,6],[8,10,12]])

a1 = np.array([1,2,3])
a2 = np.array([4,5,6])
v3 = np.vstack([v1,v2]) 
print(v3)

a3 = np.vstack([a1,a2])
print(a3)
print(a3.ndim)

# Horizontally stacking 
# Stacks arrays horzontally on x axis

a1 = np.ones([3,5])
a2 = np.zeros([3,7])

a3 = np.hstack([a1,a2])
print('HSTACKKKK')
print(a3)

# Vstack and Hstack with 3d matrix

l1 = np.ones([2,4,4])
l2 = np.zeros([3,4,4])

# vstack
print('Testing vstacking 3d array')
v3 = np.vstack([l1,l2])
print(v3)
print("Vstack")
print(v3)
# vstack must have the same dimesions to match

# hstack - Can't hstack a 3d array

a = np.zeros([3,4])
print(a)
b = np.full([3,6], 99)
c = np.hstack([a,b])
print(c)



a = np.array([1])
b = np.ones([1])

# hstacking and vstacking 1d array

print(np.vstack([a,b])) # vstacking 1d array to make 2 dimensional arrat
print(np.hstack([a,b])) # hstacking 1d array to make 2 dimensional array

# Miscellaneous

# reading txt files into an numpy array

file = np.genfromtxt('data.txt',delimiter = ',',dtype = "int32")
print(file)

# file[0,0:8] = [3,6,9,12,15,18,21]
# print(file)

print(file.sum(axis = 1))
print()
# Boolean masking and advanced indexing

file = np.genfromtxt('randnum.txt',delimiter = ",",dtype = 'int32')
print(file)

print(file>50) # goes through each item in 2d array and check for the condition, then returns an array full of booleans

# Indexing numbers in array that are all greater than a condition 
print("Advanced indexing")
a = file[file>50]# gets all numbers that satisfy the index and put innto an array
print(a.ndim)
print(a.shape)
print(a.reshape([9,2]))


## you can index specfic numbers from array
print(a[[1,3,10]]) ## get item in array based on index number

# checking for conditions down columns and down rows

print(np.any(file<-20,axis = 0)) # checks if any value down the 0 axis(down the rows) satisfy the condition

# you can check if all the rows or columns meet a condition 
print(np.all(file<15000, axis = 1))
 
# we can check for multiple condition in array using numpy syntax

print(~(file>50) & (file<100)) # uses & keyword instead of and in numpy and '~' key word for not

## Practise Question

a = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25],[26,27,28,29,30]])

print(a)
print(a.ndim)
print(a.shape)

# Getting 11,12,16,17
print(a[2:4,0:2])

## getting daigonal numbers from 2 followed by 8 14 20
print(a[[0,1,2,3],[1,2,3,4]]) # list within list index that map the rows and colums

## getting 4,5 and top rows and last numbers on last two rows

print(a[[0,4,5],3:]) # gets the row indexes and column of each number columns index 3 onwards

a = np.full([3,3], '-')
a[0,0:] = 1
# print(a)




