import re
import numpy as np
import pickle
from math import log10


def datainput(filename,attr_num=60):            #读取文件数据
    #filename = '.\data\dna.data'
    fr = open(filename)
    datalines = fr.readlines()                  #读文件的所有行
    t = []
    d = {'000':0, '001':1, '010':2, '100':3}
    for dataline in datalines:                  #对文件每一行
        dataline = dataline[0:-2]               #去掉每行最后的分号
        dataline = dataline.replace(' ','')     #去除所有空格
        temp = re.findall(r'.{3}', dataline)    #按每3个字符分割字符串，丢弃不足3个的部分
        temp.append(int(dataline[-1]))          #加上label
        for i in range(len(temp) - 1):          #将TGCA变为0,1,2,3
            #temp[i]=int(temp[i],2)
            temp[i] = d[temp[i]]
        t.append(temp)
    datamatrix = np.array(t)
    #print(datamatrix)
    return datamatrix


def calentropy(datamatrix):
    ent = np.zeros(datamatrix.shape[1] - 1)
    datanum = datamatrix.shape[0]
    for i in range(datamatrix.shape[1] - 1):
        num = np.zeros(4, dtype = np.int32)
        for j in range(4):
            lab = np.ones(3, dtype=np.int32)*0.0000000000000000000001   #当某一列属性没有某个label时，防止log里的数字为0
            id = datamatrix[:,i]==j
            num[j] = id.sum()
            if(num[j] == 0):
                continue
            lab[0] += (datamatrix[:,-1][id] == 1).sum()
            lab[1] += (datamatrix[:, -1][id] == 2).sum()
            lab[2] += (datamatrix[:, -1][id] == 3).sum()
            #print(num[j]==lab.sum())
            info = -((lab[0]/num[j])*log10(lab[0]/num[j]) + (lab[1]/num[j])*log10(lab[1]/num[j]) +(lab[2]/num[j])*log10(lab[2]/num[j]))
            ent[i] += (num[j]/datanum)*info
    #minefeature = np.argmin(ent)
    #print(ent)
    return ent
#def calentropy(datamatrix):                                 #计算信息熵

def createtree(datamatrix):
    #print('depth', depth)
    if(len(np.unique(datamatrix[:,-1]))==1):                #label全部相同，终止循环
        return datamatrix[0,-1]                             #返回label
    if(np.unique(datamatrix[:,0:-1], axis=0).shape[0]==1):  #所有属性都用到了,结束循环（datamatrix最后一列为标签）
        return np.argmax(np.bincount(datamatrix))
    #开始循环过程
    ent = calentropy(datamatrix)
    minefeature = np.argmin(ent)
    decisiontree = {'BASE' + str(minefeature):{}}
    #print(decisiontree)
    for i in range(4):                                      #对0,1,2,3(TGCA)四个值进行分割数据
        id = datamatrix[:,minefeature]==i
        #print(i, datamatrix[:,:][id])
        if(datamatrix[:,:][id].size==0):
            continue
        decisiontree['BASE' + str(minefeature)][i] = createtree(datamatrix[:,:][id])
    return decisiontree

def storetree(decisiontree, filename):
    fw = open(filename, 'wb')
    pickle.dump(decisiontree, fw)
    fw.close()

def main():
    data = datainput(filename = '.\data\dna.data')
    #g = Digraph("Decision Tree", filename='ID3-DT',format='png')
    decisiontree = createtree(data)
    storetree(decisiontree, filename='.\data\decisiontree')
    #visualization(decisiontree, g)
    print(decisiontree)



if __name__=='__main__':
    main()

