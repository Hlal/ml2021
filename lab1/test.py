import numpy as np
import pickle
from createtree import datainput

def grabtree(filename):
    fr=open(filename,'rb')#需要制定'rb'，以byte形式读取
    return pickle.load(fr)

def judgelabel(decisiontree, data):
    key, = decisiontree
    #print(key, key[4:], data[int(key[4:])], decisiontree[key])
    #print(decisiontree[key][data[int(key[4:])]])
    nexttree = decisiontree[key][data[int(key[4:])]]
    if isinstance(nexttree, np.int32):
        return int(nexttree)
    else:
        return judgelabel(nexttree, data)

def testdata(decisiontree, datatest):
    result = []
    for i in range(datatest.shape[0]):
        testlabel = judgelabel(decisiontree, datatest[i,:])
        result.append(testlabel)
    resultnp = np.array(result)
    return resultnp


def main():
    data = datainput(filename='.\data\dna.test')
    decisiontree = grabtree(filename='.\data\decisiontree')
    result = testdata(decisiontree, data)
    accuracy = (result == data[:,-1]).sum() / data.shape[0]
    fr = open('.\data\dna.result', 'w')
    fr.write('Result Label:'+ str(list(result)) +'\n' + 'Test Label:' + str(list(data[:,-1])) + '\n' + 'Accuracy: ' + str(accuracy))
    fr.close()
    print('Result Label:', list(result), '\n', 'Test Label:', list(data[:,-1]))
    print('Accuracy: ', accuracy)
    #print(decisiontree)
    #print(data)


if __name__ == '__main__':
    main()