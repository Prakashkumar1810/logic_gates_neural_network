import learn  #inporting learn module

#printing the inputs
input = '0,0\n0,1\n1,0\n1,1\n'
print('input:\n'+input)

#creating a list of objects of various gates using corresponding data files
gates = []
gates.append(learn.network('and.txt'))
gates.append(learn.network('or.txt'))
gates.append(learn.network('eor.txt'))
gates.append(learn.network('nand.txt'))
gates.append(learn.network('nor.txt'))

#iterating through each object instance
for gate in gates:
	gate.load()        #loading the dataset
	gate.train(5000)   #training the network using back propagation agorithm
	gate.predict()     #predicting the output using forward propagation
	
