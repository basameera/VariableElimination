"""
@Author: Joris van Vugt, Moira Berens

Entry point for testing the variable elimination algorithm

"""
from read_bayesnet import BayesNet
from variable_elim import VariableElimination

if __name__ == '__main__':
    # the class BayesNet represents a Bayesian network from a .bif file
    # in several variables
    # net = BayesNet('earthquake.bif') 
    net = BayesNet('cancer.bif')
    # net = BayesNet('asia.bif')

    # these are the variables that should be used for variable elimination
    
    # print '============='
    # print 'values',net.values
    # for x in net.values:
    #     if len(net.values[x])>2:
    #         print '!!!!!!!!!!!!!!!!!!!!!!!!! Danger'

    # print '============='
    # print 'probabilities'
    # for x in net.probabilities:
    #     print '---',x,'---\n',net.probabilities[x]

    # print '============='
    # print 'parents', net.parents

    # print '============='
    print 'nodes', net.nodes

    # print '\n\n === VE ===\n\n'
    
    # Make your variable elimination code in a seperate file: 'variable_elim'. 
    # you can call this file as follows:

    # If variables are known beforehand, you can represent them in the following way: 
    # evidence = {'Burglary': 'True'}

    # determine you heuristics before you call the run function. This can be done in this file or in a seperate file
    # The heuristics either specifying the elimination ordering (list) or it is a function that determines the elimination ordering
    # given the network. An simple example is: 
    # elim_order = net.nodes

	#call the elimination ordering function for example as follows:   
    #ve.run('Alarm', evidence, elim_order)

    #Query var - can be more than one (joint probability distribition)
    # Qvar = ['MaryCalls']
    # Qvar = ['JohnCalls']
    # Qvar = ['Burglary']
    # Qvar = ['Earthquake']
    # Qvar = ['Alarm']



    # Qvar = ['Xray']
    Qvar = ['Cancer']
    # Qvar = ['Dyspnoea']

    #observed var
    Obvar = {}
    #Obvar = {'Burglary': 'True'}
    
    #elim_order
    # elim_order = ['JohnCalls', 'Burglary', 'Earthquake', 'Alarm']
    elim_order = net.nodes
    elim_order.pop(elim_order.index(Qvar[0]))
    print elim_order
    # elim_order = ['MaryCalls', 'Burglary', 'Earthquake', 'Alarm']

    # elim_order = ['Dyspnoea', 'Smoker', 'Cancer', 'Pollution']
    # elim_order = ['Xray', 'Smoker', 'Cancer', 'Pollution']
    

    ve = VariableElimination(net)
    ve.run(Qvar, Obvar, elim_order)

 
