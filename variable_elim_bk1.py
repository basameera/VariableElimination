"""
@Author: Joris van Vugt, Moira Berens

Implementation of the variable elimination algorithm for AISPAML assignment 3

"""
# TODO
'''
0. How Pandas Dataframe work - DONE
1. Represent network someway - Top to bottom structure is important = DONE
2. Generate Formula - DONE
    1. Basic formula
    2. Reduce using BN structure
3. Identify factors - Factor Formula - DONE

4. Factor calculation
    1. Sum - DONE
    2. Marginalize - DONE
    3. Multiply - 
'''

import numpy as np
import pandas as pd
import itertools

class Prob():
    def __init__(self, query=[], obsv=[]):
        self.Q = query
        self.O = obsv

    def getQ(self):
        return self.Q

    def getO(self):
        return self.O


class Fact():
    def __init__(self, query=[], obsv=[]):
        self.FF = []
        for x in query:
            self.FF.append(x)

        for x in obsv:
            self.FF.append(x)

    def get(self):
        return self.FF


class VariableElimination():

    def __init__(self, network):
        self.network = network
        self.addition_steps = 0
        self.multiplication_steps = 0

    def calcFactorDistribution(self, f_remove, frList, allProb):
        print '============================================== calcFactorDistribution'
        print 'f_remove:>', f_remove, '   |  frList:>', frList

        probTables = []
        #allProb = self.network.probabilities

        #print '!!!!!!!!!!!!!!!!!!  All Prob'
        #for x in allProb:
        #    print '---', x, '---\n', allProb[x]
        #print '!!!!!!!!!!!!!!!!!!  All Prob'

        alreadyProb = []
        for y in range(len(frList)):
            for x in allProb:
                # finding prob. tables with factor elements as columns

                if set(frList[y]).issubset(allProb[x].columns):
                    # if allProb[x] not in probTables:
                    if frList[y] not in alreadyProb:
                        probTables.append(allProb[x])
                        alreadyProb.append(frList[y])

        del alreadyProb
        for z in range(len(frList)):
            print frList[z]
            print probTables[z]

        # Multiply -> Sum -> Marginalize
        #print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', frList
        
        if len(frList) == 1:  # Nothing to Multiply, GOTO Sum
            frList_1 = frList[0]
            #print 'frList_1', frList_1
            # sum - along rows with same value
            frList_1.pop(frList_1.index(f_remove))
            #print 'frList_1', frList_1
            newFactor = probTables[0].groupby(frList_1).sum()

            # reset dataframe indexes
            newFactor.reset_index(inplace=True)
            # print newFactor

            # remove columns with same value
            nunique = newFactor.apply(pd.Series.nunique)
            cols_to_drop = nunique[nunique == 1].index
            newFactor.drop(cols_to_drop, axis=1)
            # print 'newFactor'
            # print newFactor

            # Marginalize
            probSum = newFactor['prob'].sum()

            newFactor['prob'] = newFactor['prob']/probSum
            # print 'newFactor'
            # print newFactor
            return newFactor

        else:
            print 'Multiply'
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
                print '**********************************************************************conPDF'
                newDF = pdfTable(df1.drop(['prob'], axis=1), df2.drop(['prob'], axis=1))

                #calc new PDF values
                df1col = list(df1.columns)
                df1col.pop(df1col.index('prob'))

                df2col = list(df2.columns)
                df2col.pop(df2col.index('prob'))
                print df1col
                print df2col
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
                # print newDF
                return newDF
            
            # multiply
            outDF = None
            while probTables:
                # isolate the first 2 DataFrames(Prob. tables)
                df1 = probTables.pop(0)
                df2 = probTables.pop(0)

                # df1 = pd.read_csv('df1.csv')
                # df2 = pd.read_csv('df2.csv')
                # df1.to_csv('df1.csv', index=False)
                # df2.to_csv('df2.csv', index=False)
                # print '-*********************************************************************************************'
                # print 'df1'
                # print df1
                # print 'df2'
                # print df2
                #print '<>? len probTables >> ', len(probTables)

                dfcol = list(df1.columns)
                dfcol.pop(dfcol.index('prob'))

                booleandf = df1.select_dtypes(include=[object])
                booleanDictionary = {'True':True, 'False':False}
                for column in booleandf:
                    df1[column] = df1[column].map(booleanDictionary)
                

                dfcol = list(df2.columns)
                dfcol.pop(dfcol.index('prob'))
                
                booleandf = df2.select_dtypes(include=[object])
                booleanDictionary = {'True':True, 'False':False}
                for column in booleandf:
                    df2[column] = df2[column].map(booleanDictionary)

                # print df1
                # print df2
                # print df1.dtypes
                # print df2.dtypes

                if len(probTables)==0:
                    outDF = conPDF(df1, df2)
                else:
                    probTables.append(conPDF(df1, df2))

            # sum -> marg.
            if outDF is None:
                print 'NONE'
            else:
                print 'Ready to Sum'
                # print outDF
                # sum - along rows with same value
                outDFcol = list(outDF.columns)
                outDFcol.pop(outDFcol.index(f_remove))
                outDFcol.pop(outDFcol.index('prob'))
                newFactor = outDF.groupby(outDFcol).sum()

                # reset dataframe indexes
                newFactor.reset_index(inplace=True)
                # print 'RESET newFactor'
                # print newFactor

                # remove columns with same value
                nunique = newFactor.apply(pd.Series.nunique)
                cols_to_drop = nunique[nunique == 1].index
                # print cols_to_drop
                newFactor = newFactor.drop(cols_to_drop, axis=1)
                # print 'newFactor'
                # print newFactor

                # Marginalize
                probSum = newFactor['prob'].sum()
                newFactor['prob'] = newFactor['prob']/probSum
                
                # print '@@@ newFactor'
                # print newFactor
                outDF = newFactor

            return outDF

    def run(self, query, observed, elim_order):
        """
        Use the variable elimination algorithm to find out the probability
        distribution of the query variable given the observed variables

        Input:
            query:      The query variable
            observed:   A dictionary of the observed variables {variable: value}
            elim_order: Either a list specifying the elimination ordering
                        or a function that will determine an elimination ordering
                        given the network during the run

        Output: A variable holding the probability distribution
                for the query variable

        ['Burglary', 'MaryCalls', 'Alarm', 'JohnCalls', 'Earthquake']
        """
        # print self.network.values

        '''
        Network structure ordering
        '''
        par = self.network.parents
        parLen = {}
        # parLen = {'XXX':4,'YYY':3}
        for x in par:
            parLen[x] = len(par[x])
        # print 'parLen',parLen

        networkOrder = []
        temp = []
        # plen = len(parLen)
        for x in parLen:
            if parLen[x] == 1:
                networkOrder.append(x)
                temp.append(x)

        [parLen.pop(key) for key in temp]
        # print parLen
        # print networkOrder

        temp = []
        leafNodes = []
        for x in parLen:
            if parLen[x] == 0:
                leafNodes.append(x)
                temp.append(x)
        [parLen.pop(key) for key in temp]
        # print parLen
        # print leafNodes

        for k in sorted(parLen, key=lambda k: parLen[k], reverse=True):
            # print k
            networkOrder.append(k)

        for x in leafNodes:
            networkOrder.append(x)

        # print networkOrder

        '''
        Construct formula
        '''
        mainFormula = []

        def testa(arr, outarr):
            if arr:
                outarr.append(Prob(arr[0], arr[1:]))
                arr.pop(0)
                testa(arr, outarr)

        testa(networkOrder, mainFormula)

        # for x in range( len(mainFormula)):
        #     print '(',mainFormula[x].getQ(),'|', mainFormula[x].getO(),')'

        '''
        Reduce formula using BN structure
        '''
        # print par
        reducedFormula = []
        for x in range(len(mainFormula)):
            # print mainFormula[x].getQ(), '+++' ,par[mainFormula[x].getQ()], '===' , mainFormula[x].getO()
            # print (mainFormula[x].getO()).index(par[mainFormula[x].getQ()][0])
            tempY = []
            for y in par[mainFormula[x].getQ()]:
                for z in mainFormula[x].getO():
                    if y == z:
                        tempY.append(y)
            reducedFormula.append(Prob([mainFormula[x].getQ()], tempY))

        # print "--------------------------------------------"
        # for x in range(len(reducedFormula)):
        #     print '(',reducedFormula[x].getQ(),'|', reducedFormula[x].getO(),')'

        '''
        Factor formula
        '''
        factorFormula = []
        temp = []
        for x in range(len(reducedFormula)):
            temp.append(
                Fact(reducedFormula[x].getQ(), reducedFormula[x].getO()))

        for x in range(len(temp)):
            factorFormula.append(temp[x].get())
            # print factorFormula
        '''
        Factor elimination
        '''
        factorFormula_1 = factorFormula[:]

        print 'Start elim_order:', elim_order

        # change prob. dataframe to a factor dataframe
        allProbT = self.network.probabilities
        allProbFrame = {}
        fval = 0
        for x in allProbT:
            allProbFrame[fval] = allProbT[x]
            fval += 1


        # Loop here - through all Factors n the formula
        fact_remove = elim_order.pop(0)
        #fact_remove = 'Burglary'
        print 'fact_remove:', fact_remove

        removeFactors = []

        for x in factorFormula:
            if fact_remove in x:
                removeFactors.append(x)
                factorFormula_1.pop(factorFormula_1.index(x))

        # calc new factor prob. distribution
        print 'removeFactors', removeFactors
        print 'factorFormula', factorFormula_1

        # abcd = [['MaryCalls', 'Alarm'],['Alarm', 'JohnCalls']]
        #fact_remove = 'Burglary'

        #removeFactors = [['Burglary', 'Alarm', 'Earthquake'],['Burglary']] # multiply - sum - marg.
        # removeFactors = [['Burglary', 'Alarm', 'Earthquake']] # sum - marg.

        newFactor = self.calcFactorDistribution(fact_remove, removeFactors, allProbFrame)
        
        print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  newFactor:'
        print newFactor

        # TODO : add the new factor var. to the Formular
        newFacForm = list(newFactor.columns)
        newFacForm.pop(newFacForm.index('prob'))
        #print newFacForm
        factorFormula_1.append(newFacForm)
        print 'factorFormula >>', factorFormula_1

        # TODO : add the new factor var. to the allProbFrame
        allProbFrame[len(allProbFrame)] = newFactor


        # ['Burglary', 'MaryCalls', 'Alarm', 'JohnCalls', 'Earthquake']
