import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import load_model
import os
import argparse

#----------------------------
# get command line variables
#----------------------------
parser = argparse.ArgumentParser(description='Make models by keras. Place Y on the head column in the cleaned dataset with header names on the top row. Rows containing null values will be deleated.')
parser.add_argument('--mode', choices=['create', 'predict'], dest='mode', metavar='create/predict', type=str, nargs='+', required=True,
                    help='an integer for the accumulator')
parser.add_argument('--input_file', dest='input_file', type=str, nargs='+', required=True,
                    help='path to dataset or model')
parser.add_argument('--method', choices=['binary', 'multiple', 'regression'], metavar='binary/multiple/regression', dest='method', type=str, nargs='+', required=True,
                    help='Model type you solve')
parser.add_argument('--output_file', dest='output_file', default=False, required=False,
                    help='If you input output_file it will save result as directed path.')
parser.add_argument('--model_file', dest='model_file', default=False, nargs='*',
                    help='If you input model_file it will save or load a model.')
parser.add_argument('--definition', metavar='array of data type such as str, int and float with delimiter [,]', dest='definition', default=False, nargs='*',
                    help='If you define data type of columns, send array of full column definitions.')

args = parser.parse_args()

#----------------------------
# functions
#----------------------------
class MakeModel:
	#init
	def __init__(self, args):
		self.X = self.Y = []
		self.row_length = self.column_length = 0
		self.method = args.method[0]
		self.ifp = args.input_file[0]
		
		if args.model_file != False:
			self.mfp = args.model_file[0]
		else:
			self.mfp = False
		
		if args.output_file != False:
			self.ofp = args.output_file[0]
		else:
			self.ofp = False
		
		if args.definition != False:
			self.dfin = args.definition.split(",")
		else:
			self.dfin = False
		        
	#create layers
	def create_model(self, evMethod, neurons, layers, act, learn_rate, cls, mtr):
		# Criate model
		model = Sequential()
		model.add(Dense(neurons, input_dim=self.column_length, kernel_initializer='normal', activation='relu'))
		for i in range(1, layers):
			model.add(Dense(int(numpy.ceil(numpy.power(neurons,1/i)*2)), kernel_initializer='normal', activation='relu'))
		model.add(Dense(cls, kernel_initializer='normal', activation=act))
		# Compile model
		adam = optimizers.Adam(lr=learn_rate)
		model.compile(loss=evMethod, optimizer=adam, metrics=mtr)
		return model

	#load dataset
	def load_dataset(self):
		dataframe = pandas.read_csv(self.ifp, header=0).dropna()
		if self.dfin != False:
			dataframe[dataframe.columns].apply(lambda x: x.astype(self.dfin[dataframe.columns.get_loc(x.name)]))
		dataframe_X = pandas.get_dummies(dataframe[dataframe.columns[1:]])	#create dummy variables
		if self.method == 'multiple':
			dataframe_Y = pandas.get_dummies(dataframe[dataframe.columns[0]])	#create dummy variables
		else:
			dataframe_Y = dataframe[dataframe.columns[0]]
		#print(dataframe_Y.head())
		#print(dataframe_X.head())
		self.row_length, self.column_length = dataframe_X.shape
		self.X = dataframe_X.values
		self.Y = dataframe_Y.values
			
	#train
	def train_model(self):
		#pipe to Grid Search
		estimators = []
		estimators.append(('standardize', StandardScaler()))
		
		#rely on chosen method parameters
		if self.method == 'binary':
			evMethod = ['binary_crossentropy']
			activation = ['sigmoid']
			metr = [['accuracy']]
			estimators.append(('mlp', KerasClassifier(build_fn=self.create_model, epochs=10, batch_size=200, verbose=1)))
			cls = [1]
		elif self.method == 'multiple':
			evMethod = [['categorical_crossentropy']]
			activation = ['softmax']
			metr = [['accuracy']]
			estimators.append(('mlp', KerasClassifier(build_fn=self.create_model, epochs=10, batch_size=200, verbose=1)))
			cls = [self.Y.shape[1]]
		else:
			evMethod = ['mean_squared_error']
			activation = [None]
			metr = [None]
			estimators.append(('mlp', KerasRegressor(build_fn=self.create_model, epochs=10, batch_size=200, verbose=1)))
			cls = [1]

		pipeline = Pipeline(estimators)
		
		#test parameters
		batch_size = list(set([int(numpy.ceil(self.row_length/i)) for i in [1000,300,100]]))
		epochs = [10, 50, 100]
		neurons = list(set([int(numpy.ceil(self.column_length/i)*2) for i in numpy.arange(1,3,0.4)]))
		learn_rate = [0.001, 0.005, 0.01, 0.07]
		layers = [1,2,3,4,5]
		#test parameter
		"""batch_size = [31]
		epochs = [100]
		neurons = [32]
		learn_rate = [0.01]
		layers = [5]"""
		#execution
		param_grid = dict(mlp__neurons = neurons, mlp__batch_size = batch_size, mlp__epochs=epochs, mlp__learn_rate=learn_rate, mlp__layers=layers, mlp__act=activation, mlp__evMethod=evMethod, mlp__cls=cls, mlp__mtr=metr)
		grid = GridSearchCV(estimator=pipeline, param_grid=param_grid)
		grid_result = grid.fit(self.X, self.Y)
		
		#output best parameter condition
		clf = []
		clf = grid_result.best_estimator_
		print(clf.get_params())
		accuracy = clf.score(self.X, self.Y)
		if self.method in ['binary', 'multiple']:
			print("\nAccuracy: %.2f" % (accuracy))
		else:
			print("Results: %.2f (%.2f) MSE" % (accuracy.mean(), accuracy.std()))
		
		#save model
		if self.mfp != False:
			clf.steps[1][1].model.save(self.mfp)

	#predict dataset
	def predict_ds(self):
		model = load_model(self.mfp)
		model.summary()
		sc = StandardScaler()
		self.X = sc.fit_transform(self.X)
		pr_Y = model.predict(self.X)
		if len([self.Y != '__null__']) > 0:
			if self.method == 'binary':
				predictions = [float(numpy.round(x)) for x in pr_Y]
				accuracy = numpy.mean(predictions == self.Y)
				print("Prediction Accuracy: %.2f%%" % (accuracy*100))
			elif self.method  == 'multiple':
				predictions = []
				for i in range(0, len(pr_Y)-1):
					for j in range(0, len(pr_Y[i])-1):
						predictions.append(int(round(pr_Y[i][j]) - self.Y[i][j]))
				accuracy_total = len([x for x in predictions if x == 0])/len(predictions)
				accuracy_tooneg = len([x for x in predictions if x == -1])/len(predictions)
				accuracy_toopos = len([x for x in predictions if x == 1])/len(predictions)
				print("Prediction Accuracy: %.2f%% (positive-error:%.2f%%/negative-error:%.2f%%)" % (accuracy_total*100, accuracy_tooneg*100, accuracy_toopos*100))
			else:
				accuracy = numpy.mean((self.Y - pr_Y)**2)
				print("MSE: %.2f" % (numpy.sqrt(accuracy)))
				
		#save predicted result
		if self.ofp != False:
			numpy.savetxt(self.ofp, pr_Y, fmt='%5s')

#----------------------------
# select mode
#----------------------------
m = MakeModel(args)
if args.mode == ['create']:
	#make model
	m.load_dataset()
	m.train_model()
else:
	#predict dataset
	m.load_dataset()
	m.predict_ds()
