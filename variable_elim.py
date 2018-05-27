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
# import itertools

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
        # print '============================================== calcFactorDistribution'
        # print 'f_remove:>', f_remove, '   |  frList:>', frList

        probTables = []
        alreadyProb = []
        facts_to_remove = []
        #create a subset of factor values that contain frList
        for y in range(len(frList)):
            for x in allProb:
                # finding prob. tables with factor elements as column
                tempCOl = list(allProb[x].columns)
                tempCOl.pop(tempCOl.index('prob'))

                if set(frList[y])==set(tempCOl):
                    # if allProb[x] not in probTables:
                    if frList[y] not in alreadyProb:
                        probTables.append(allProb[x])
                        alreadyProb.append(frList[y])
                        facts_to_remove.append(x)
                        break
                    else:
                        for z in range(len(facts_to_remove)):
                            T = facts_to_remove[z]
                            if set(allProb[T].columns) == set(allProb[x].columns):

                                if not set(allProb[T]['prob'])==set(allProb[x]['prob']):
                                    probTables.append(allProb[x])
                                    alreadyProb.append(frList[y])
                                    facts_to_remove.append(x)
                                    break
        
        del alreadyProb

        if not len(frList) == len(probTables):
            print '\n++++++++++++++!!!!!!!!!!!!!!\nTHese 2 values should be equal'
            print 'len(frList[z]):',len(frList)
            print 'len(probTables[z]):',len(probTables)
            print '++++++++++++++++!!!!!!!!!!!!!!\n'

        # Multiply -> Sum -> Marginalize
        if len(frList) == 1:  # Nothing to Multiply, GOTO Sum
            frList_1 = frList[0]
            # sum - along rows with same value
            frList_1.pop(frList_1.index(f_remove))
            newFactor = probTables[0].groupby(frList_1).sum()

            # reset dataframe indexes
            newFactor.reset_index(inplace=True)

            # remove columns with same value - not 'prob'
            nunique = newFactor.apply(pd.Series.nunique)
            cols_to_drop = list(nunique[nunique == 1].index)
            if cols_to_drop:
                cols_to_drop.pop(cols_to_drop.index('prob'))
                newFactor = newFactor.drop(cols_to_drop, axis=1)

            # Marginalize
            probSum = newFactor['prob'].sum()

            newFactor['prob'] = newFactor['prob']/probSum
            return newFactor, facts_to_remove

        else:
            # multiply
            outDF = None
            while probTables:
                # isolate the first 2 DataFrames(Prob. tables)
                df3 = probTables.pop(0)
                df4 = probTables.pop(0)

                df3Col = list(df3.columns)
                df4Col = list(df4.columns)

                comCol = list(np.intersect1d(df3Col, df4Col))
                if 'prob' in comCol:
                    comCol.pop(comCol.index('prob'))
                # print comCol

                df3 = df3.rename(index=str, columns={"prob": "prob_x"})
                df4 = df4.rename(index=str, columns={"prob": "prob_y"})

                merged_inner = pd.merge(left=df3,right=df4, left_on=comCol[0], right_on=comCol[0])
                merged_inner['prob_z'] = np.multiply(merged_inner['prob_x'],merged_inner['prob_y'])
                merged_inner = merged_inner.drop(['prob_x','prob_y'], axis=1)
                merged_inner = merged_inner.rename(index=str, columns={"prob_z": "prob"})

                if len(probTables)==0:
                    outDF = merged_inner
                else:
                    probTables.append(merged_inner)


            # sum -> marg.
            if outDF is None:
                print '***NONE***'
            else:
                # sum - along rows with same value
                outDFcol = list(outDF.columns)
                outDFcol.pop(outDFcol.index(f_remove))
                outDFcol.pop(outDFcol.index('prob'))
                # print outDFcol
                newFactor = outDF.groupby(outDFcol).sum()
                
                # reset dataframe indexes
                newFactor.reset_index(inplace=True)
                # print newFactor
                # remove columns with same value
                nunique = newFactor.apply(pd.Series.nunique)
                # print nunique
                cols_to_drop = list(nunique[nunique == 1].index)
                # print cols_to_drop
                if cols_to_drop:
                    cols_to_drop.pop(cols_to_drop.index('prob'))
                    newFactor = newFactor.drop(cols_to_drop, axis=1)

                # Marginalize
                probSum = newFactor['prob'].sum()
                newFactor['prob'] = newFactor['prob']/probSum
                
                outDF = newFactor

            return outDF, facts_to_remove

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

        """

        '''
        Network structure ordering (hierarchical, top to bottom)
        '''
        # TODO: **
        par = self.network.parents
        parLen = {}
        # Calculate the degrees of each node, using parents dict.
        for x in par:
            parLen[x] = len(par[x])

        networkOrder = []
        temp = []
        # find leaf nodes
        for x in parLen:
            if parLen[x] == 1:# only one parent
                networkOrder.append(x)
                temp.append(x)

        [parLen.pop(key) for key in temp]

        temp = []
        grandParNodes = []
        # find grand parent nodes
        for x in parLen:
            if parLen[x] == 0: #no parents
                grandParNodes.append(x)
                temp.append(x)
        [parLen.pop(key) for key in temp]

        for k in sorted(parLen, key=lambda k: parLen[k], reverse=True):
            networkOrder.append(k)

        for x in grandParNodes:
            networkOrder.append(x)
        
        # networkOrder: has the BN structure from bottom to top because,
        #               that's how the JPD formula is created next
        print '>> networkOrder', networkOrder

        '''
        Construct JPD formula
        '''
        mainFormula = []

        def JPDformula(arr, outarr):
            if arr:
                outarr.append(Prob(arr[0], arr[1:]))
                arr.pop(0)
                JPDformula(arr, outarr)

        JPDformula(networkOrder, mainFormula)

        '''
        Reduce formula using BN structure
        '''
        # TODO: **
        reducedFormula = []
        for x in range(len(mainFormula)):
            tempY = []
            for y in par[mainFormula[x].getQ()]:
                for z in mainFormula[x].getO():
                    if y == z:
                        tempY.append(y)
            reducedFormula.append(Prob([mainFormula[x].getQ()], tempY))

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

        factorID = len(allProbFrame)
        
        # Factor elimination
        while elim_order:
            fact_remove = elim_order.pop(0)
            removeFactors = []

            factorFormulaTemp = factorFormula_1[:]
            for x in factorFormulaTemp:
                if fact_remove in x:
                    removeFactors.append(x)
                    factorFormula_1.pop(factorFormula_1.index(x))

            # calc new factor prob. distribution
            newFactor, facts_to_remove = self.calcFactorDistribution(fact_remove, removeFactors, allProbFrame)

            for x in facts_to_remove:
                allProbFrame.pop(x)

            # TODO : add the new factor var. to the Formular
            newFacForm = list(newFactor.columns)
            newFacForm.pop(newFacForm.index('prob'))

            factorFormula_1.append(newFacForm)
            # print 'factorFormula >>', factorFormula_1

            # TODO : add the new factor var. to the allProbFrame
            allProbFrame[factorID] = newFactor
            factorID += factorID
            if elim_order:
                print ''
            else:
                print '\n+++++++++++++++++++++++++++++++++++ Final Value +++++++++++++++++++++++++++++++++++++'
                print newFactor
                print '+++++++++++++++++++++++++++++++++++ Final Value +++++++++++++++++++++++++++++++++++++\n'
        # print 'allProbFrame'
        # print allProbFrame
        # ['Burglary', 'MaryCalls', 'Alarm', 'JohnCalls', 'Earthquake']



