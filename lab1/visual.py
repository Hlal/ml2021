from graphviz import Digraph    #需同时安装graphviz库和windows软件
import numpy as np
import pickle

def grabtree(filename):
    fr=open(filename,'rb')#需要制定'rb'，以byte形式读取
    return pickle.load(fr)

def visualization(decisiontree, g, parentnode=None, att=None):
    if decisiontree is None:
        return
    #g.attr('node', shape='box')
    key, = decisiontree
    d = {0:'T', 1:'G', 2:'C', 3:'A'}
    #print(type(key))
    if(not parentnode):
        g.node(key)
        for k in decisiontree[key]:
            visualization(decisiontree[key][k], g, key, k)
    else:
        #print(type(parentnode), type(key), type(att))
        sun = parentnode+str(att)+key
        g.node(sun, label=key)
        g.edge(parentnode, sun, label=d[int(att)])
        for k in decisiontree[key]:
            if isinstance(decisiontree[key][k], np.int32):
                end = sun + str(k) + str(decisiontree[key][k])
                g.node(end, label=str(decisiontree[key][k]))
                g.edge(sun, end, label=d[k])
            else:
                visualization(decisiontree[key][k], g, sun, str(k))

def main():

    g = Digraph("Decision Tree", filename='.\data\ID3-DT',format='png')
    g.attr('node', shape='box')
    decisiontree = grabtree(filename='.\data\decisiontree')
    print(decisiontree)
    visualization(decisiontree, g)
    g.view()

if __name__ == '__main__':
    main()
