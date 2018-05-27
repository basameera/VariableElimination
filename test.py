import numpy as np
import pandas as pd
import itertools
df1 = pd.DataFrame({
    'A': [2, 3],
    'B': [4, 5]
})

df2 = pd.DataFrame({
    'A': [-1, 2],
    'B': [3, -1],
    'C': [3, -1]
})

# print df1
# print df2

# df3 = df1.multiply(df2, axis='columns', level=None)
# print df3


graph = {'A': ['C'],
         'B': ['C'],
         'C': ['D', 'E'],
         'D': [],
         'E': []
        }


def find_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if not graph.has_key(start):
        return None
    for node in graph[start]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
            if newpath:
                return newpath
    return None
# print graph
# path = find_path(graph, 'A', 'E')
# print path

df5 = pd.DataFrame({
    'J': [True,False,True,False,True,False,True,False],
    'A': [True,True,False,False,True,True,False,False],
    'B': [True,True,True,True,False,False,False,False],
    'P': [1,2,3,4,5,6,7,8]
})

# print df5

# df6 = df5.groupby(['A','B']).sum()

# df6.reset_index(inplace=True)
# print df6

# # cols = list(df6)
# nunique = df6.apply(pd.Series.nunique)
# cols_to_drop = nunique[nunique == 1].index
# df6 = df6.drop(cols_to_drop, axis=1)
# print df6
# # marginalize
# # print df6.describe()

# probSum = df6['P'].sum()
# print probSum

# # df6['P'] = df6['P'].apply(lambda x: x/probSum)
# df6['P'] = df6['P']/probSum
# print df6

# print df6['P'].sum()

# df1 = pd.DataFrame({
#     'M': [True,False,True,False],
#     'A': [True,True,False,False],
#     'prob': [0.7,0.3,0.01,0.99]
# })

# df2 = pd.DataFrame({
#     'J': [True,False,True,False],
#     'A': [True,True,False,False],
#     'prob': [0.9,0.1,0.05,0.95]
# })

df1 = pd.DataFrame({
    'Alarm':      [True,False,True,False,True,False,True,False],
    'Burglary':   [True,True,False,False,True,True,False,False],
    'Earthquake': [True,True,True,True,False,False,False,False],
    'prob': [0.95,0.05,0.29,0.71,0.94,0.06,0.001,0.999]
})

df2 = pd.DataFrame({
    'Burglary': [True,False],
    'prob': [0.01,0.99]
})

# df1 = pd.read_csv('df1.csv')
# df2 = pd.read_csv('df2.csv')

# print df1
# print df2

# print df1.dtypes
# print df2.dtypes

print '========================================================='


def pdfTable(df1, df2):
    print 'pdfTable'
    aaa = []
    for x in df1.columns:
        aaa.append(x)
    for x in df2.columns:
        aaa.append(x)

    # Create new DataFrame
    table = list(itertools.product([True, False], repeat=len(set(aaa))))
    newdf = pd.DataFrame(table, columns=set(aaa))

    sLength = len(newdf[aaa[0]])
    newdf['prob'] = pd.Series(np.zeros(sLength), index=newdf.index)
    # print newdf
    return newdf


def conPDF(df1, df2):
    print 'conPDF'
    newDF = pdfTable(df1.drop(['prob'], axis=1), df2.drop(['prob'], axis=1))
    # print newDF
    #calc new PDF values
    df1col = list(df1.columns)
    df1col.pop(df1col.index('prob'))

    df2col = list(df2.columns)
    df2col.pop(df2col.index('prob'))

    for index, row in newDF.iterrows():
        probVal1 = 0
        probVal2 = 0

        temp1 = df1
        # print 'temp1 '
        # print temp1
        for x in df1col:
            temp1 = temp1.loc[temp1[x] == row[x]]
            if len(temp1)==1:
                # print 'Val found ====',temp1['prob']
                probVal1 = temp1['prob']

        temp1 = df2
        for x in df2col:
            temp1 = temp1.loc[temp1[x] == row[x]]
            if len(temp1)==1:
                # print 'Val found ====',temp1['prob']
                probVal2 = temp1['prob']

        newDF.at[index, 'prob'] = np.multiply(probVal1,probVal2)
        # print '---------------newProb', newDF.at[index, 'prob']
        # break
    print newDF
    return newDF

#conPDF(df1, df2)

# tempVal = df1.loc[df1['A'] == True].loc[df1['M'] == True]['prob']

# print tempVal

# df3 = pd.DataFrame({
#     'Cancer':      ['True','False','True','False','True','False','True','False'],
#     'Pollution':   ['low','low','high','high','low','low','high','high'],
#     'Smoker': ['True','True','True','True','False','False','False','False'],
#     'prob': [0.03,0.97,0.05,0.95,0.001,0.999,0.02,0.98]
# })

# df4 = pd.DataFrame({
#     'Smoker': ['True','False'],
#     'prob': [0.3,0.7]
# })

df3 = pd.DataFrame({
    'M': [True,False,True,False],
    'A': [True,True,False,False],
    'prob': [0.7,0.3,0.01,0.99]
})

df4 = pd.DataFrame({
    'J': [True,False,True,False],
    'A': [True,True,False,False],
    'prob': [0.9,0.1,0.05,0.95]
})


def pdfTable1(df1, df2):
    # print 'pdfTable'
    aaa = []
    for x in df1.columns:
        aaa.append(x)
    for x in df2.columns:
        aaa.append(x)

    # Create new DataFrame
    table = list(itertools.product([True, False], repeat=len(set(aaa))))
    newdf = pd.DataFrame(table, columns=set(aaa))

    sLength = len(newdf[aaa[0]])
    newdf['prob'] = pd.Series(np.zeros(sLength), index=newdf.index)
    return newdf

def conPDF1(df1, df2):
    # print '****conPDF'
    newDF = pdfTable(df1.drop(['prob'], axis=1), df2.drop(['prob'], axis=1))

    #calc new PDF values
    df1col = list(df1.columns)
    df1col.pop(df1col.index('prob'))

    df2col = list(df2.columns)
    df2col.pop(df2col.index('prob'))
    for index, row in newDF.iterrows():
        probVal1 = 0
        probVal2 = 0

        temp1 = df1
        for x in df1col:
            temp1 = temp1.loc[temp1[x] == row[x]]
            if len(temp1)==1:
                probVal1 = temp1['prob']

        temp1 = df2
        for x in df2col:
            temp1 = temp1.loc[temp1[x] == row[x]]
            if len(temp1)==1:
                probVal2 = temp1['prob']

        newDF.at[index, 'prob'] = np.multiply(probVal1,probVal2)

    return newDF

# print df3
# print df4

# print df3.dtypes
# print df4.dtypes


# get column names
df3Col = list(df3.columns)
df4Col = list(df4.columns)

comCol = list(np.intersect1d(df3Col, df4Col))
if 'prob' in comCol:
    comCol.pop(comCol.index('prob'))
# print comCol

df3 = df3.rename(index=str, columns={"prob": "prob_x"})
df4 = df4.rename(index=str, columns={"prob": "prob_y"})
# print df3
# print df4

if len(comCol)==1:
    merged_inner = pd.merge(left=df3,right=df4, left_on=comCol[0], right_on=comCol[0])
    print merged_inner
    # print merged_inner.dtypes

    merged_inner['prob_z'] = np.multiply(merged_inner['prob_x'],merged_inner['prob_y'])
    merged_inner = merged_inner.drop(['prob_x','prob_y'], axis=1)
    merged_inner = merged_inner.rename(index=str, columns={"prob_z": "prob"})
    print merged_inner
    # print merged_inner.dtypes