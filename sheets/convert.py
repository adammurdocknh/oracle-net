import numpy
# from numpy import genfromtxt
import os

# print(os.getcwd())

sheet = './sheets/bler.csv'

my_data = numpy.genfromtxt(sheet, delimiter=',')



print(my_data[1,:])