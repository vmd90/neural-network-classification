
# Aluno: Victor Municelli Dario
# NUSP: 7152778

from pybrain.structure import *
from pybrain.datasets import *
from pybrain.supervised.trainers import *
import arff
import matplotlib.pyplot as plt

def classify(network, data):
    output = network.activate(data).argmax(axis=0)
 
    print("Classified as: " + str(output))
    return output
#-----------------------------------------------------------
def main():
	L = list(arff.load('iris.arff'))

	ds = ClassificationDataSet(4, class_labels=['Iris-setosa','Iris-versicolor','Iris-virginica'])

	for row in L:
		sl = row['sepallength']
		sw = row['sepalwidth']
		pl = row['petallength']
		pw = row['petalwidth']
		cl = 0
		if row['class'] == 'Iris-versicolor':
			cl = 1
		elif row['class'] == 'Iris-virginica':
			cl = 2
		ds.appendLinked([sl,sw,pl,pw], [cl])

	test, train = ds.splitWithProportion(0.25)
	test._convertToOneOfMany()
	train._convertToOneOfMany()

	#n = buildNetwork(train.indim, 5, train.outdim)

	# iniciando RNA
	n = FeedForwardNetwork()
	# adicionando as 3 camadas
	n.addInputModule(LinearLayer(train.indim, name='in'))
	n.addModule(SigmoidLayer(5, name='hidden'))
	n.addOutputModule(LinearLayer(train.outdim, name='out'))
	# conectando as camadas
	n.addConnection(FullConnection(n['in'], n['hidden']))
	n.addConnection(FullConnection(n['hidden'], n['out']))
	n.sortModules()

	# Criando o objeto "trainer" da rede neural
	trainer = BackpropTrainer(n, train, learningrate=0.1, verbose=True)

	# Treinando a rede
	err = 1.0
	while err > 0.01:
	    err = trainer.train()
#------------------------------------------------------

if __name__ == '__main__':
	main()
