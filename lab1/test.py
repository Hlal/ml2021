import numpy as np
import pickle
from createtree import datainput

def grabtree(filename):
    fr=open(filename,'rb')#需要制定'rb'，以byte形式读取
    return pickle.load(fr)

def testdata(decisiontree, datatest):
    result = []

def main():
    data = datainput(filename='.\data\dna.test')
    decisiontree = grabtree(filename='.\data\decisiontree')
    print(decisiontree)
    print(data)


if __name__ == '__main__':
    main()