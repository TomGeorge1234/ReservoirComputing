import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as anim
import copy
import time 
import pickle
from IPython.display import Video
from pprint import pprint
import matplotlib 
import random
from tqdm.notebook import tqdm
import os
from datetime import datetime
from matplotlib import rcParams
from cycler import cycler 
plt.style.use("seaborn")
rcParams['figure.dpi']= 300
rcParams['axes.labelsize']=5
rcParams['axes.labelpad']=2
rcParams['axes.titlepad']=3
rcParams['axes.titlesize']=5
rcParams['axes.xmargin']=0
rcParams['axes.ymargin']=0
rcParams['xtick.labelsize']=4
rcParams['ytick.labelsize']=4
rcParams['grid.linewidth']=0.5
rcParams['legend.fontsize']=4
rcParams['lines.linewidth']=0.5
rcParams.update({'figure.autolayout': True})
rcParams['xtick.major.pad']=2
rcParams['xtick.minor.pad']=2
rcParams['ytick.major.pad']=2
rcParams['ytick.minor.pad']=2
rcParams['xtick.color']='grey'
rcParams['ytick.color']='grey'
rcParams['figure.titlesize']='medium'
rcParams['axes.prop_cycle']=cycler('color', ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'])

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']




defaultHyperparams = {'Nres' : 300,     #no. reservoir units 
					'Nz' : 1,           #no. output units
					'Nin' : 26,         #no. input units
					'Nres_in' : 300,    #no. reservoir units which directly connect to input
					'Nres_out' : 300,   #no. reservoir units which directly connect to output
					'p' : 0.8,          #probability two reservoir units are connected 
					'ipr' : 1,          #no. of input units each input-connected reservoir unit joins
					'dt' : 1,           #simulation time constant in ms
					'tau' : 10,         #neuron time constant in ms
					'alpha' : 100,      #FORCE learning rate (aka. P-matrix regularisation param)
					'g_res' : 1.5,      #connection strength of reservoir units  
					'g_FB' : 1,         #connection strength of feedback 
					'biasInput' : 0,    #a global input sent to all neurons 
					'sigma' : 0.3}      #network noise std


"""
RESERVOIR CLASS
Builds a single reservoir using passed hyperparmeters 
"""
class Reservoir():
	"""A reservoir network, initialised with passed hyperparameters
	• runDynamicsStep() - takes input vector and runs a single full dynamics step
	• runTrainingStep() - takes the desired output and updates the weights (and P matrix) according to the rules of FORCE learning
	"""	
	def __init__(self,hyperparams=None):
		"""Initialises hyperparameters as network atributes
		Args:
			hyperparams (dict): Dictionary of hyperparemters. Defaults to None which then triggers Default Hyperperparams to be used
		"""		
		if hyperparams is None: hyperparams = defaultHyperparams
		self.hyperparams = hyperparams

		self.Nres = hyperparams['Nres']          #no. reservoir units 
		self.Nz = hyperparams['Nz']              #no. output units
		self.Nin = hyperparams['Nin']        #no. input units
		self.Nres_in = min(hyperparams['Nres_in'],self.Nres)    #no. reservoir units which directly connect to input
		self.Nres_out = min(hyperparams['Nres_out'], self.Nres)  #no. reservoir units which directly connect to output
		self.biasInput = hyperparams['biasInput']
		self.p = hyperparams['p']                #probability two reservoir units are connected 
		self.ipr = hyperparams['ipr']            #no. of input units each input-connected reservoir unit joins
		self.dt = hyperparams['dt']              #simulation time constant in ms
		self.tau = hyperparams['tau']            #neuron time constant in ms
		self.alpha = hyperparams['alpha']        #FORCE learning rate (aka. P-matrix regularisation param)
		self.g_res = hyperparams['g_res']        #connection strength of reservoir units  
		self.g_FB = hyperparams['g_FB']          #connection strength of feedback 
		self.sigma = hyperparams['sigma']        #network noise std

		self.className = "Reservoir"

		self.hist = {} #a dictionary where history is stored 

		self._initialise() #initialises networks weights etc. 

	def _initialise(self):
		"""Initialises all nodes (input, reservoir and output) and their connections/weights
		"""		
		#Input nodes to reservoir nodes 
		#each reservoir neurons connects to on average self.ipr input nodes
		J_GI = np.zeros(shape=(self.Nres,self.Nin))
		for i in range(self.Nres_in):
			idx = list(range(self.Nin))
			random.shuffle(idx)
			choosenIdxs = idx[:self.ipr]
			for j in range(self.Nin):
				if j in choosenIdxs:
					J_GI[i,j] = np.random.normal()
		self.J_GI = J_GI 

		#Reservoir to reservoir recurrent weights
		#nodes connect to each otyher with probability p
		self.J_GG = np.zeros(shape=(self.Nres,self.Nres))
		for i in range(self.Nres):
			for j in range(self.Nres):
				if np.random.rand() < self.p:
					self.J_GG[i,j] = (1/np.sqrt(self.p * self.Nres)) * np.random.normal() #recurrent connection
		
		#Reservoir to output
		self.w = np.random.randn(self.Nz,self.Nres_out)/np.sqrt(self.Nres_out)

		#An index list which define which reservoir nodes connect to the output (typical this is not all of them)
		self.synapseList = np.arange(self.Nres-self.Nres_out,self.Nres) 

		#Output to reservoir (i.e. feedback)
		self.J_Gz = np.random.uniform(low=-1,high=1,size=(self.Nres,self.Nz)) #feedback connection 

		#Reservoir node initial values
		self.x = np.random.normal(size=(self.Nres)) #Node values (membrane potentials) initialised randomly
		self.r = np.tanh(self.x) #Activated node values (tanh activated "firing rates")
		self.z = np.matmul(self.w, self.r[self.synapseList]) #initial output 

		#FORCE learning inverse correlation matrix initial value
		self.P = (1.0/self.alpha) * np.identity(self.Nres_out) #learning matrix 


	def runDynamicsStep(self,inputVec=None,returnItems=None):
		"""Runs a dynamics step to update reservoir node values
		Args:
			inputVec (array, optional): The value of the input nodes at current time step. Defaults to None therefore zero.
			returnItems (list of integers, optional): Which node values would you like the to return. Network updates is completed irrespective of whether/which nodes are returned. Defaults to None.
		Returns:
			[type]: [description]
		"""		
		if inputVec is None: inputVec = np.zeros(self.Nin)

		#Dynamical Equations:  see Asabuki et al. 2018 or FORCE learning paper (Sussillo) 
		self.x = ((1 - self.dt/self.tau)*self.x +  #self-decay
				(self.dt/self.tau)*self.g_res*np.matmul(self.J_GG,self.r) +  #recurrent input
				(self.dt/self.tau)*self.g_FB*np.matmul(self.J_Gz,self.z) +  #feedback input
				(self.dt/self.tau)*np.matmul(self.J_GI,inputVec) +  #external input
				(self.dt/self.tau)*np.ones(self.Nres)*self.biasInput + #bias input
				np.sqrt(self.dt)*self.sigma*np.random.randn(self.Nres)) #noise
		self.r = np.tanh(self.x)
		self.z = np.matmul(self.w,self.r[self.synapseList])

		#returns reservoir variable to the user 
		returnables = {}
		if returnItems == None: return
		if 'z' in returnItems: returnables['z'] = self.z
		if 'r' in returnItems: returnables['r'] = self.r
		if 'x' in returnItems: returnables['x'] = self.x
		return returnables

	def runTrainingStep(self,desiredOutputs):
		"""Givenn a desired output, this trains the network (by FORCE learning, i.e. RLS) 
		Args:
			desiredOutputs: desider output for current time step that the weights will be fitted to
		Returns:
			float: the error, before the weight update
		"""		
		#update P
		r = self.r[self.synapseList]
		k = np.dot(self.P, r)
		rPr = np.dot(r,k)
		c = 1.0 / (1.0 + rPr)
		self.P = self.P - c*np.outer(k, k)
		#update the w's
		e_minus = self.z - desiredOutputs 
		dw = - (c * (e_minus * np.tile(k,(self.Nz,1)).T)).T
		self.w = self.w + dw
		return e_minus






"""
INPUTS
Experiments usually involve training the reservoirs on input timeseries with varying levels of temporaly regularity. 
These functions assist with this by building a dictionary storing not only the actual input but also data such as the time or what 'chunks' we may be in. 

Currently inputs are limited to temporal resolution of dt = 1ms but I should generalise this
"""
defaultInputParams = {'experiment' : 'normal', #normal, schapiro, schaeffer etc. 
					'totalTime' : 500, #s
					'width' : 50, #ms 
					'chunkList' : [['a','b','c','d']], #in normal experiments chunks appears repeatedly between interchunk intervals 
					'chunkProbs' : None, #defauls to uniform probability of each chunk
					'chunkLabels' : None, #defaults to [0,1,2,...]
					'gapRange' : [5,9],  #gap between chunks randomly from this range (inclusive)
					'syllables' : alphabet, #all 
					'singleChunkOnly' : False, #True if you only want a single chunk e.g. for testing
					'interChunkSyllables': None, #if defined, syllables between chunks will only be from here, chunks will never be from here. Must be subset of syllables (optionally plus ' ').
					'chunkSize' : None, #schaeffer experiment specific (gives number of syllables in each chunk),
					'gain' : 2,
					'pdfs' : None}  #schaeffer experiment specific (gives probability of letters in chunk)  

def _syllableShape(syllable=' ',inputParams=defaultInputParams):
	"""returns input with one (specified) syllable set to rising and falling exponetial 
	Args:
		syllable (str, optional): The syllable (i.e. which one of the inputs) to set non-zero. Defaults to ' ' i.e. no syllable.
		inputParams (dictionary of inputParams, optional): The parameters of the inputs (syllable list, syllable time etc.). Defaults to defaultInputParams.
	Returns:
		array: of shape (N_input x 2*width.dt) to be added to the inputData frame at the required point.
	"""	
	syllables, width = inputParams['syllables'],inputParams['width']
	syllableShape = np.zeros(shape=(2*width,len(syllables)))
	inputShape = np.zeros(width*2) #the rising falling exponential shape of the inputs
	inputShape[0:width]       = inputParams['gain']*(1 - np.exp(-(np.arange(0,width)/10)))
	inputShape[width:2*width] = inputParams['gain']*np.exp(-(np.arange(0,width)/10))

	if syllable == ' ': pass #' ' special character reserved for no input
	else:
		for char in syllable:
			syllableShape[:,syllables.index(char)] = inputShape
	return syllableShape 





def _getNextSyllable(inputParams): #an iterator functions that when calls spits out the next syllables. This does all the calculation for when to move between chunks etc. 
	"""This function return (yields) the next syllable to be added to input. It records whether it is in or between chunks and yields next syllable correspoding to the experimental parameters given in the inputParams.

	Args:
		inputParams (dict): The parameters of the inputs (syllable list, syllable time etc.)
	Yields:
		str: the next syllable
		idx: whether the current syllable is in a chunk (idx = 0,1,2 corresponding to chunkId) or interchunk/random ('r'). Used for plotting later on.  
	"""
	interChunk = True #start not in a chunk 

	syllables = inputParams['syllables']
	experiment = inputParams['experiment']
	chunkList = inputParams['chunkList']
	chunkProbs = inputParams['chunkProbs']
	chunkLabels = inputParams['chunkLabels']
	gapRange = inputParams['gapRange']
	interChunkSyllables = inputParams['interChunkSyllables']

	while True: 
		#choose random syllable between chunks 
		# choose randomly from the syllable list unless interChunkSyllables is define, in whic case pick from these. 
		if interChunk == True: 
			for i in range(random.randint(gapRange[0],gapRange[1])):
				if interChunkSyllables is not None:
					nextSyllable = random.choices(interChunkSyllables)[0] #change depending on experiment
				else: 
					nextSyllable = random.choices(syllables)[0] #change depending on experiment
				yield nextSyllable, 'r'
			interChunk = False
			continue
		
		#while inside a chunk iterate through the chunk syllables and return them at each step 
		#if syllable ends in '+' this represents OR. e.g. "abc+" means return "a" or "b" or "c"
		#if syllable ends in '_' this represents .NOT e.g. "abc_" means return any syllable except "a" or "b" or "c" or those syllables in interChunkSyllables
		if interChunk == False:
			if experiment == 'normal':
				idx = random.choices(range(len(chunkList)),weights=chunkProbs)[0]
				for nextSyllable in chunkList[idx]:
					if len(nextSyllable) != 1:
						if nextSyllable[-1] == '_': #anything BUT these syllables 
							disallowed = [char for char in nextSyllable[:-1]]
							if interChunkSyllables is not None: 
								disallowed.extend(interChunkSyllables)
							nextSyllable = random.choice([sy for sy in syllables if sy not in disallowed])		
						if nextSyllable[-1] == '+':
							nextSyllable = random.choice([sy for sy in nextSyllable[:-1]])	
					yield nextSyllable, chunkLabels[idx]
				interChunk = True

			#random walk round community structure graph (see Schapiro et al. 2019?))
			if experiment == 'schapiro':
				syllable = 'a'
				Tmat = np.array([[0,1,1,1,0,0,0,0,0,0,0,0,0,0,1],
								[1,0,1,1,1,0,0,0,0,0,0,0,0,0,0],
								[1,1,0,1,1,0,0,0,0,0,0,0,0,0,0],
								[1,1,1,0,1,0,0,0,0,0,0,0,0,0,0],
								[0,1,1,1,0,1,0,0,0,0,0,0,0,0,0],
								[0,0,0,0,1,0,1,1,1,0,0,0,0,0,0],
								[0,0,0,0,0,1,0,1,1,1,0,0,0,0,0],
								[0,0,0,0,0,1,1,0,1,1,0,0,0,0,0],
								[0,0,0,0,0,1,1,1,0,1,0,0,0,0,0],
								[0,0,0,0,0,0,1,1,1,0,1,0,0,0,0],
								[0,0,0,0,0,0,0,0,0,1,0,1,1,1,0],
								[0,0,0,0,0,0,0,0,0,0,1,0,1,1,1],
								[0,0,0,0,0,0,0,0,0,0,1,1,0,1,1],
								[0,0,0,0,0,0,0,0,0,0,1,1,1,0,1],
								[1,0,0,0,0,0,0,0,0,0,0,1,1,1,0]])
				while True:
					nextSyllable = random.choices(syllables,weights = Tmat[syllables.index(syllable)])[0]
					if nextSyllable in ['a','b','c','d','e']: chunkLabel = 0
					elif nextSyllable in ['f','g','h','i','j']: chunkLabel = 1
					elif nextSyllable in ['k','l','m','n','o']: chunkLabel = 2
					yield nextSyllable, chunkLabel
					syllable = nextSyllable

			#select syllable randomly from a pdf specific to current chunk
			if experiment == 'schaeffer':
				pdfs = inputParams['pdfs']
				idx = random.choices(range(len(pdfs)),weights=chunkProbs)[0]
				for _ in range(inputParams['chunkSize']):
					nextSyllable = random.choices(chunkList[0],weights=pdfs[idx])
					yield nextSyllable, idx
				interChunk = True





def getInputs(inputParams,totalTime=None):
	"""Constructs the inputs given all the information contained in inputParams. Returns a dict with all the information/data about the input inside. 
	This dict is used later on by variety of functions. This function relies on _getNextSyllable and _syllableShape
	Args:
		inputParams (dict): dictionary of inputParameters. Non-option but if anything keys within the dictionary ar enot define these will be pulled from defaultInputParams
		totalTime (float, optional): If defined this will overrule 'totalTime' define in inputParams to limit the time-length of the input. Defaults to None.
	Returns:
		dict: dictionary containing inputData and information. See bottom of function for description
	"""

	#pulls from defaultParams if any inputParams are not present
	for item in list(defaultInputParams.keys()):
		try: inputParams[item]
		except KeyError:
			if item == 'chunkProbs': inputParams[item] = [1]*len(inputParams['chunkList'])
			else: inputParams[item] = defaultInputParams[item]
	if inputParams['singleChunkOnly'] == True: 
		inputParams['totalTime'] = inputParams['width'] * len(inputParams['chunkList'][0]) / 1000
		inputParams['gapRange'] = [0,0]
	if inputParams['chunkLabels'] == None: 
		inputParams['chunkLabels'] = list(np.arange(len(inputParams['chunkList'])))

	if totalTime is not None: 
		inputParams['totalTime'] = totalTime

	width = inputParams['width']

	chunkLabelList = []
	syllableList = []
	t = np.arange(inputParams['totalTime'] * 1000) / 1000 #time in units of seconds
	T = len(t) #maximum index of time array 

	#initialse inout data array as all zeros 
	data = np.zeros(shape=(T,_syllableShape(inputParams=inputParams).shape[1])) 

	s = 0 
	#pulls next syllabme from _getNextSyllable then creates the input array for this syllable and adds it to the full input data array
	for nextSyllable, nextChunkLabel in _getNextSyllable(inputParams=inputParams):
		if s*inputParams['width'] >= T: 
			break
		else:
			nextInput = _syllableShape(nextSyllable,inputParams)
			data[s*width:min(T,s*width+nextInput.shape[0]),:] += nextInput[:min(T,s*width+nextInput.shape[0])-s*width,:]
			chunkLabelList.extend([nextChunkLabel] * inputParams['width'])
			syllableList.extend([nextSyllable] * inputParams['width'])
		s += 1

	chunkData = []
	oldChunkLabel, oldChunkChangeId = chunkLabelList[0], 0
	for i in range(len(t)):
		chunkLabel = chunkLabelList[i]
		if chunkLabel != oldChunkLabel or i == len(t)-1:
			chunkChangeId = [i]
			if oldChunkLabel != 'r': 
				chunkData.append([oldChunkLabel,t[oldChunkChangeId],t[chunkChangeId]])
			oldChunkChangeId = chunkChangeId
		oldChunkLabel = chunkLabel

	inputData = {'data':data,                    #the input data that will be passed to the reservoir
				't':t,                           #timestamps for all the inputs
				'chunkData':chunkData,           #lists of where all the chunks start and finish and what their chunkId is. 
				'chunkLabelList':chunkLabelList, #same size as t, a label for every time stamp of what chunk you're currently in (contains same info as chunkData just different form which is sometimes useful)
				'syllableList':syllableList,     #list of syllables as they appear in the input data (one entry per syllable, not per time step)
				'inputParams':inputParams}       #all of the input parameters which went into making this (quite useful for stuff later on) 

	return inputData





def plotInputs(inputs,tstart=0,tend=5,title=None,saveName=""):
	"""Takes the inputs and plots it nicely showing where the chunks are (shaded) and listing syllables on yaxis

	Args:
		inputs (dict): the basic input data dictionary returned when you make the input (i.e. returned by getInputs)
		tstart (int, optional): time to start plotsting at. Defaults to 0.
		tend (int, optional): time to end plotting at. Defaults to 5.
		title (str, optional): Plto title. Defaults to None.
		saveName (str, optional): Name for saving to file. Defaults to "".

	Returns:
		[type]: [description]
	"""	
	inputArray = inputs['data']
	t = inputs['t']
	chunkData = inputs['chunkData']
	syllables = inputs['inputParams']['syllables']

	fig, ax = plt.subplots(figsize=(4,2))
	T = t[-1]

	s, e = np.abs(t - tstart).argmin(), np.abs(t - min(tend,t[-1])).argmin(),
	yextent = 0.5*(3/26)*inputs['data'].shape[1] #scales y axis to make it look ok regardless of length you are plotting 

	ax.imshow(inputArray.T[:,s:e],extent=[tstart,min(tend,t[-1]),0,yextent])
	ax.set_ylabel("Input")
	ax.set_xlabel("Time / s")
	plt.yticks((np.linspace(0,yextent*(1-1/len(syllables)),len(syllables)) + 0.5*yextent/len(syllables))[::-1],syllables)
	for c in chunkData: #shade where chunks are 
		if c[1] < min(tend,t[-1]) and c[2] > tstart:
			rect = matplotlib.patches.Rectangle((c[1],yextent),(c[2]-c[1]),-yextent,linewidth=0,edgecolor='r',facecolor='C%s'%(c[0]),alpha=0.3)
			ax.add_patch(rect)
	ax.grid(False)
	if title is not None: 
		ax.set_title(title)

	saveName = "inputs"+saveName
	saveFigure(fig, saveName)

	return fig, ax





def saveFigure(fig,saveTitle=""):
	"""saves figure to file, by data (folder) and time (name) 
	Args:
		fig (matplotlib fig object): the figure to be saved
		saveTitle (str, optional): name to be saved as. Current time will be appended to this Defaults to "".
	"""	
	today =  datetime.strftime(datetime.now(),'%y%m%d')
	if not os.path.isdir(f"./figures/{today}/"):
		os.mkdir(f"./figures/{today}/")
	figdir = f"./figures/{today}/"
	now = datetime.strftime(datetime.now(),'%H%M')
	path = f"{figdir}{saveTitle}_{now}.png"
	fig.savefig(f"{figdir}{saveTitle}_{now}.pdf", dpi=400,tight_layout=True)
	return path





	
	
class ReservoirPair():
	"""Initilises and controls a pair of reservoirs which intereact as defined in Asabuki et al. 2018
	These two networks can be called individually as ReservoirPair.res1 and ReservoirPair.res2. At which point they are nothing more than instances of the reservoir class defined above
	The point of this class is to make it quick and easy to train two networks to do temporal chunking (as defined in the paper) 
	Thus it has a number of functions: 
	• trainPair implements the mutual prediction training algorithm 
	• store inputs allows you to store an new input (the basic input dictionary from getInput) under a name which can later be trained or tested on. 
	• f rectified tanh activating applied to output for the neighbour to predict (mutual inhibition if multiple outputs)
	
	"""	
	def __init__(self,hyperparams,inputs=None):
		"""Initialises the reservoir pair 
		Args:
			hyperparams (dict): Dictionary of hyperparameters, primarily for the reservoir networks
			inputs (dict, optional): If defined these will be stored as an input under the name 'train', by default training will be on this input. Defaults to None.
		"""		
		self.hyperparams = hyperparams
		self.className = "ReservoirPair"

		self.inputDict = {} #a dictionary in which input dictionaries will be stored (for training and testing) 
		self.hist = {} #a dictionary forsaving historical data (useful for later plotting)
		
		self.z1_list = None
		self.z2_list = None
		
		# set inputs 
		if inputs != None: 
			self.storeInputs(inputs,name='train')

		#initialise each reservoir
		self.res1 = Reservoir(hyperparams)
		self.res2 = Reservoir(hyperparams)
	
	def storeInputs(self,inputs,name='input'):
		"""Stores inputs in a dictionary. Useful as multiple inputs (from getInput) can be saved, some used for training, some used tfor testing/experimenting later. 
		Args:
			inputs (dict): standard inputDict from getInput
			name (str): What to call this input . Defaults to 'input'
		"""		
		self.inputDict[name] = inputs

	def trainPair(self,inputs=None,window=5,saveTrain=False,returnItems=['z'],maxTrainTime=None):
		"""Trains the reservoir pair by mutual prediction algorithm discussed in Asabuki et al. 2018.
		Args:
			inputs (dict): the input dictionary to be trained on, if not defined will try reservoirPair.inputDict['train']. Defaults to None.
			window (int, optional): The output-to-be-predicted is normalised by a running average this long, in seconds. Defaults to 5.
			saveTrain (bool, optional): If true, the items in returnItems will be saved to history. Defaults to False.
			returnItems (list, optional): Which items to return to the user. Defaults to ['z'].
		"""	

		if inputs is None: 
			inputs = self.inputDict['train']

		dt = self.hyperparams['dt']
		if ((self.z1_list is None) or (self.z2_list is None)):
			self.z1_list = np.zeros(shape=(self.hyperparams['Nz'],int(window/(dt/1000))))
			self.z2_list = np.zeros(shape=(self.hyperparams['Nz'],int(window/(dt/1000))))
			self.z_boollist = np.zeros(shape=(int(window/(dt/1000))),dtype=bool)
			self.readyToTrain = False


		if saveTrain == True:
			self.res1.hist['train'] = {}
			self.res1.hist['train']['z'] = np.zeros(shape=(self.hyperparams['Nz'],self.inputDict['train']['data'].shape[0],1))
			self.res1.hist['train']['r'] = np.zeros(shape=(self.hyperparams['Nres'],self.inputDict['train']['data'].shape[0],1))
			self.res1.hist['train']['inputs'] = inputs

		if maxTrainTime is None: endIdx = len(inputs['data'])
		else: endIdx = np.argmin(np.abs(inputs['t'] - maxTrainTime))
		for i in tqdm(range(endIdx),desc='Training reservoir pair'):
			reservoir1Output = self.res1.runDynamicsStep(inputs['data'][i],returnItems=returnItems)
			reservoir2Output = self.res2.runDynamicsStep(inputs['data'][i],returnItems=returnItems)

			z1 = reservoir1Output['z']
			z2 = reservoir2Output['z']
			if 'r' in returnItems:
				r1 = reservoir1Output['r']
				r2 = reservoir2Output['r']

			if saveTrain == True: #save outputs 
				if 'z' in returnItems:
					self.res1.hist['train']['z'][:,i,0] = z1
				if 'r' in returnItems:
					self.res1.hist['train']['r'][:,i,0] = r1

			self.z1_list = np.roll(self.z1_list, -1); self.z1_list[:,-1] = z1 #roll short term history arrays and replace last with new output
			self.z2_list = np.roll(self.z2_list, -1); self.z2_list[:,-1] = z2
			self.z_boollist = np.roll(self.z_boollist, -1); self.z_boollist[-1] = True
			if np.all(self.z_boollist): 
				self.readyToTrain = True

			if self.readyToTrain == True: 
				y1 = (z1 - np.mean(self.z1_list,axis=1)) / np.std(self.z1_list,axis=1) #normalise the output
				y2 = (z2 - np.mean(self.z2_list,axis=1)) / np.std(self.z2_list,axis=1)

				(y1,y2) = self.rectifiedTanhOutput(y1,y2) #rectiry tanh it as per paper 

				self.res1.runTrainingStep(y2) #res1 tries to predict res2
				self.res2.runTrainingStep(y1) #res2 tries to predict res1

	def rectifiedTanhOutput(self,ya,yb,beta=3,gamma=0.5):
		"""function applied to the outputs before being passed to train the neghbouring reservoir
			rectified tanh. if multidimensional, there is also inhibition. 
		Args:
			ya (list or array): current output from res1 (singe time step, one entry per output node (Nz))
			yb (list or array): current output from res1 (singe time step, one entry per output node (Nz))
			beta (int, optional): Sharpness of tanh (higher = more step-function-like). Defaults to 3.
			gamma (float, optional): Amount of inhibition (if >= 2 outputs). Defaults to 0.5.

		Returns:
			tuple: (ya,yb) after function applied 
		"""		
		try: gamma = self.hyperparams['gamma']
		except KeyError: pass
		if self.hyperparams['Nz'] > 1:
			for i in range(self.hyperparams['Nz']):
				for j in range(i+1,i+1+self.hyperparams['Nz']-1):
					ya[i] += - gamma*ya[j%self.hyperparams['Nz']]
					yb[i] += - gamma*yb[j%self.hyperparams['Nz']]                    
		ya = np.tanh(ya/beta)
		yb = np.tanh(yb/beta)
		ya = np.maximum(ya,0)
		yb = np.maximum(yb,0)
		return (ya,yb)

	def resetWeights(self):
		res1.w = np.random.randn(res1.Nz,res1.Nres_out)/np.sqrt(res1.Nres_out)
		res2.w = np.random.randn(res2.Nz,res2.Nres_out)/np.sqrt(res2.Nres_out)
		return






def testReservoir(reservoir,inputs=None,Ntest=1,testName='test',saveItems=['z','r'],verbose=True,maxTestTime=None,plotAfter=False):
	"""Tests reservoir (if reservoir pair is passed, it tests res1 of this pair) on an inputs and stores to history. 
	Args:
		reservoir (reservoir): the reservoir to test 
		input (dict): the input dict (from getInputs) to test the reservoir on 
		Ntest (int, optional): How many identical (different due to random initilisation and noise) tests to run. Defaults to 1.
		testName (str, optional): What to call the test when it is saved. Defaults to 'test'.
		testData (str, optional): Name of data store in inputsDict, to be tested. Defaults to testName.
		saveItems (list, optional): which items to save to history. Defaults to ['z'].
		verbose (bool, optional): If false, testing will be silent (no output). Defaults to True.
		plotAfter (bool): if True, plots the result of the test straight after 
	"""		
	if reservoir.className == 'ReservoirPair': #if you pass a reservoir pair, defaults to res1
		reservoir = reservoir.res1

	if inputs is None:
		inputs = reservoir.hist['train']['inputs']

	endTestTime_idx = len(inputs['t'])
	if (maxTestTime is not None):
		endTestTime_idx = np.argmin(np.abs(inputs['t'] - maxTestTime))


	Nz = reservoir.hyperparams['Nz']
	Nres = reservoir.hyperparams['Nres']

	testData = inputs['data']

	#prepare the history
	reservoir.hist[testName] = {}
	reservoir.hist[testName]['z'] = np.zeros(shape=(Nz,testData.shape[0],Ntest))
	reservoir.hist[testName]['r'] = np.zeros(shape=(Nres,testData.shape[0],Ntest))
	reservoir.hist[testName]['inputs'] = inputs


	#run the tests
	for i in tqdm(range(Ntest),desc='Running multiple tests of reservoir',disable=(not verbose)):
		for j in tqdm(range(endTestTime_idx),desc='Test %g'%i,leave=False,disable=(not verbose)):
			reservoirOutput = reservoir.runDynamicsStep(testData[j],returnItems=saveItems)
			if 'z' in saveItems: 
				reservoir.hist[testName]['z'][:,j,i] = reservoirOutput['z']
			if 'r' in saveItems:
				reservoir.hist[testName]['r'][:,j,i] = reservoirOutput['r']
	
	if plotAfter == True: 
		plotTest(reservoir)
	
	return 
	




print()

def plotTest(reservoir,
			testName='test',
			tstart=0,
			tend=5,
			colorOrders=None,
			plotTrials=False,
			title="",
			saveName=""):
	"""Plots the result of a test on a reservoir. `the input to, and the results from, this test are saved in reservoir.testhist under the key name.
	Dispalys the outputs of the reservoir, the plot is sahed according to whether the network is current in a chunk or not

	Args:
		reservoir (reservoir class): the reservoir which has been (perhaps trained and) tested 
		testName (str, optional): the name of the test to look up in testhist. Defaults to 'test'.
		tstart (float, optional): time in test to start plotting. Defaults to 0.
		tend (float, optional): time in test to stop plotting. Defaults to 5.
		colorOrders (list of strs, optional): List colours for the reservoir outputs to be plotted as. Defaults to None.
		plotTrials (bool, optional): If True, all trials are plotted in a light shade. Defaults to True.
		title (str): plot title. Defaults to "".
		saveName (str, optional): name for plot to be saved as. Defaults to "".

	Returns:
		fig: figure fig (matplotlib)
		ax: figure axis object (matplotlib)
	"""	
	if reservoir.className == 'ReservoirPair': #if you pass a reservoir pair, defaults to res1
		reservoir = reservoir.res1

	if colorOrders is None: colorOrders = np.arange(reservoir.Nz)+reservoir.Nz #define colours for plotting 

	t = reservoir.hist[testName]['inputs']['t']
	chunkData = reservoir.hist[testName]['inputs']['chunkData']
	testData = reservoir.hist[testName]['z']

	fig,ax = plt.subplots(figsize=(4,1))
	s,e = np.abs(t - tstart).argmin(), np.abs(t - tend).argmin()
	for i in range(reservoir.Nz):
		if plotTrials == True:
			for j in range(testData.shape[2]):
				ax.plot(t[s:e],np.array(testData)[i,s:e,j],alpha=0.1,c='C%g'%(colorOrders[i]),linewidth=0.1)
		ax.plot(t[s:e],np.mean(np.array(testData),axis=2)[i,s:e],alpha=1,c='C%g'%(colorOrders[i]))
		ax.fill_between(t[s:e],np.mean(np.array(testData),axis=2)[i,s:e]+np.std(np.array(testData),axis=2)[i,s:e],np.mean(np.array(testData),axis=2)[i,s:e]-np.std(np.array(testData),axis=2)[i,s:e],color='C%g'%(colorOrders[i]),alpha=0.2)

	for c in chunkData:
		if c[1] < tend and c[2] > tstart:
			rect = matplotlib.patches.Rectangle((c[1],3),(c[2]-c[1]),-6,linewidth=0,edgecolor='r',facecolor='C%s'%(c[0]),alpha=0.3)
			ax.add_patch(rect)

	ax.set_xlabel('Time / s')
	ax.set_ylabel('Activity')

	ax.set_title(title)
	
	saveName = "test"+saveName
	saveFigure(fig, saveName)

	return fig, ax 







def pickleAndSave(class_,name,saveDir='./savedItems/'):
	"""pickles and saves a class
	this is not an efficient way to save the data, but it is easy 
	use carefully with reservoir and Reservoireservoirair clases as these can get HUGE (~gigabytes)
	this will overwrite previous saves without warning
	Args:
		class_ (any class): the class/model to save
		name (str): the name to save it under 
		saveDir (str, optional): Directory to save into. Defaults to './savedItems/'.
	"""	
	with open(saveDir + name+'.pkl', 'wb') as output:
		pickle.dump(class_, output, pickle.HIGHEST_PROTOCOL)
	return 

def loadAndDepickle(name, saveDir='./savedItems/'):
	"""Loads and depickles a class saved using pickleAndSave
	Args:
		name (str): name it was saved as
		saveDir (str, optional): Directory it was saved in. Defaults to './savedItems/'.

	Returns:
		class: the class/model
	"""	
	with open(saveDir + name+'.pkl', 'rb') as input:
		item = pickle.load(input)
	return item











def dimensionalityReducedPlot(reservoir, testName='test',saveName="", method='PCA',dims=2):
	"""takes a reservoir and the results from a test done on that and plots 1st 2 principle component representations of all syllables inside and outside chunks. 
	Args:
		reservoir (Reservoir class): The reservoir where the test history data is stored 
		testName (str, optional): In the reservoir these should be a bank of histories. reservoir.hist[<testName>] will be the one plotted in this analysis. Defaults to 'test'.
		saveName (str, optional): Figure will be saved under this name in the figures directory. Defaults to ""
	Returns:
		fig: the figure object, matplotlib
		ax: axis object, matplotlib
	"""	
	if method == 'PCA':
		X = reservoir.hist[testName]['r'][:,:,0].T #after transpose shape is Ntimestep x Nres
		X = X - np.mean(X,axis=0)
		XX_T = np.matmul(X.T,X)
		eigenvals, eigenvecs = np.linalg.eig(XX_T)
		reducedX = np.matmul(X,eigenvecs[:,:dims])
	
	elif method == 'neurons':
		X = reservoir.hist[testName]['r'][:,:,0].T
		reducedX = X[:,:dims]



	inputs = reservoir.hist[testName]['inputs']
	syllablesByChunk = {}
	syllablesByChunk['r'] = {}
	for chunkLabel in inputs['inputParams']['chunkLabels']:
		syllablesByChunk[chunkLabel] = {}
	for (i,z) in enumerate(reducedX):
		syllable = inputs['syllableList'][i]
		chunkLabel = inputs['chunkLabelList'][i]

		try:
			syllablesByChunk[chunkLabel][syllable]
		except KeyError: #initialise
			syllablesByChunk[chunkLabel][syllable] = {}
			syllablesByChunk[chunkLabel][syllable]['count'] = 0
			syllablesByChunk[chunkLabel][syllable]['mean'] = np.array([0,0],dtype=float)
			syllablesByChunk[chunkLabel][syllable]['M'] = np.array([0,0],dtype=float)
			syllablesByChunk[chunkLabel][syllable]['std'] = np.array([0,0],dtype=float)

		syllablesByChunk[chunkLabel][syllable]['count'] += 1 #update count 
		mean = syllablesByChunk[chunkLabel][syllable]['mean']
		count = syllablesByChunk[chunkLabel][syllable]['count']
		delta = z - mean
		syllablesByChunk[chunkLabel][syllable]['mean'] += delta/count #update mean 
		mean = syllablesByChunk[chunkLabel][syllable]['mean']
		delta2 = z - mean
		syllablesByChunk[chunkLabel][syllable]['M'] += delta*delta2 #update M
		M = syllablesByChunk[chunkLabel][syllable]['M']
		syllablesByChunk[chunkLabel][syllable]['std'] = np.sqrt(M/count)

	fig, ax = plt.subplots(figsize=(1.5,1.5))
	for chunkLabel in syllablesByChunk:
		for syllable in syllablesByChunk[chunkLabel]:
			mean = syllablesByChunk[chunkLabel][syllable]['mean']
			std = syllablesByChunk[chunkLabel][syllable]['std']
			(x,y) = (mean[0],mean[1])
			(width,height)=(2*std[0],2*std[1])
			if chunkLabel == 'r': col = 'grey'
			else: col = 'C%s'%(chunkLabel)
			ellipse = matplotlib.patches.Ellipse((x,y), width, height,linewidth=0,edgecolor='r',facecolor=col,alpha=0.2)
			ax.add_patch(ellipse)
			if syllable == ' ':
				syllable = '_'
			ax.text(x,y,syllable,color=col)
	ax.set_xlim([-0.6*np.max(np.abs(reducedX)),0.6*np.max(np.abs(reducedX))])
	ax.set_ylim([-0.6*np.max(np.abs(reducedX)),0.6*np.max(np.abs(reducedX))])
	ax.set_xlabel("PCA1")
	ax.set_ylabel("PCA2")

	saveName = "pca"+saveName
	saveFigure(fig,saveName)
	
	return fig, ax



def averageChunkPlot(reservoir, inputs):
	return



def histogramWeights(reservoir,xmax=None,saveName=""):
	w = np.reshape(reservoir.w,-1)
	fig, ax = plt.subplots(figsize=(1,1))
	ax.set_xlabel("Weight strength")
	ax.set_ylabel("Count")
	ax.hist(w,bins=40)
	if xmax is not None: 
		ax.set_xlim(xmax,-xmax)
	saveName = "weights"+saveName
	saveFigure(fig,saveName)
	return fig, ax






"""
======================================================================
OTHER RANDOM FUNCTIONS I STILL NEED TO CLEAN UP 
======================================================================
"""








class AnimatedScatter(object):
	"""An animated scatter plot using matplotlib.animations.FuncAnimation."""
	def __init__(self, data, inputs, length=90,fps=20):
		self.stream = self.data_stream()
		self.inputs = inputs
		self.skip = int(1000/fps)
		frames = int(length*fps)
		self.data = data[::self.skip,:]
		self.count = 0
		# Setup the figure and axes...
		self.rectifiedTanhOutputig, self.ax = plt.subplots(figsize=(3,3))

		# Then setup FuncAnimation.
		self.ani = anim.FuncAnimation(self.fig, self.update, frames=frames,interval=int(1000/fps), 
										init_func=self.setup_plot, blit=True)

		plt.close()

	def setup_plot(self):
		"""Initial drawing of the scatter plot."""
		x, y, s, c = next(self.stream).T
		self.scat = self.ax.scatter([], [], c=[], s=[], vmin=0, vmax=1,
									cmap="Set2")
		self.ax.axis([-13, 13, -13, 13])
		self.ax.set_aspect('equal')
		self.ax.set_xlabel('PC1')
		self.ax.set_ylabel('PC2')
		# For FuncAnimation's sake, we need to return the artist we'll be using
		# Note that it expects a sequence of artists, thus the trailing comma.
		return self.scat,

	def data_stream(self):
		data = np.roll(self.data,-int(self.count),axis=0)
		colors = np.zeros(20)
		sizes = 3*np.arange(20)
		while True:
			xy = data[max(0,self.count-20+1):self.count+1,:]
			t = self.inputs['t'][int(self.count*self.skip)]
			color = 7 + 1/16
			for chunk in self.inputs['chunkData']:
				if t >= chunk[1] and t < chunk[2]:
					color = (chunk[0]/8) + (1/16)
					break
				else: pass
			colors = np.roll(colors,-1); colors[-1] = color
			self.count += 1
			yield np.c_[xy[:,0], xy[:,1], sizes[-len(xy):],colors[-len(xy):]]

	def update(self, i):
		"""Update the scatter plot."""
		data = next(self.stream)
		self.scat.set_offsets(data[:, :2])
		self.scat.set_sizes(data[:, 2])
		self.scat.set_array(data[:, 3])
		return self.scat,




class AnimatedChaos(object):
	"""An animated scatter plot using matplotlib.animations.FuncAnimation."""
	def __init__(self, data, length=10,fps=20,xylim=10,color='C0',axisOff=False):
		skip = int(len(data)/(length*fps))  
		self.data = data[::skip]
		self.axisOff = axisOff
		self.count = 0
		self.color=color
		self.xylim=xylim
		self.stream = self.data_stream()
		self.count = 0 #reset 
		frames = max(len(self.data), int(length*fps)) 
		# Setup the figure and axes...
		self.fig, self.ax = plt.subplots(figsize=(2,2))

		# Then setup FuncAnimation.
		self.ani = anim.FuncAnimation(self.fig, self.update, frames=frames-2,interval=int(1000/fps), 
										init_func=self.setup_plot, blit=True)

		plt.close()

	def setup_plot(self):
		"""Initial drawing of the scatter plot."""
		x, y = next(self.stream).T
		self.scat = self.ax.scatter([], [], vmin=0, vmax=1,cmap="Set2",s=6,c=self.color)
		self.ax.axis([-self.xylim, self.xylim, -self.xylim, self.xylim])
		self.ax.set_aspect('equal')
		self.ax.set_xlabel(r'$\delta x_1$')
		self.ax.set_ylabel(r'$\delta x_2$')
		if self.axisOff == True:
			plt.axis('off')
		return self.scat,

	def data_stream(self):
		while True:
			xy = self.data[self.count]
			self.count += 1
			yield np.c_[xy[:,0], xy[:,1]]

	def update(self, i):
		"""Update the scatter plot."""
		data = next(self.stream)
		self.scat.set_offsets(data[:, :2])
		return self.scat,




def getAveragePosterior(likelihood_c1, likelihood_c2, nObservations, priorCentre=0.5, priorSpread=0.01):
	n = 1000
	M_c1c1 = np.zeros((n,n))
	M_c2c1 = np.zeros((n,n))
	s = 0
	prior_c1 = np.arange(n)/n

	for i in range(n):
		for o in range(5): 
			posterior_c1 = (likelihood_c1[o]*prior_c1[i])/((likelihood_c1[o]*prior_c1[i]) + (likelihood_c2[o]*(1-prior_c1[i])))
			try: M_c1c1[i,int(np.floor(posterior_c1*n))] += posterior_c1*likelihood_c1[o]
			except IndexError: pass 

	p = np.linspace(0,1,1000)
	sigma = priorSpread

	pdfPrior_c1 = 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(priorCentre-p)**2/(2*sigma**2))

	avPosterior_c1 = []
	avPosterior_c2 = []
	stdPosterior_c1 = []
	stdPosterior_c2 = []

	for i in range(nObservations + 1):
		Ma_c1 = np.matmul(np.linalg.matrix_power(M_c1c1,i).T,pdfPrior_c1)

		avPosterior_c1.append(np.average(p, weights = Ma_c1))
		avPosterior_c2.append(1-avPosterior_c1[-1])
		stdPosterior_c1.append(np.sqrt(np.average((p-avPosterior_c1[-1])**2, weights = Ma_c1)))
		stdPosterior_c2.append(stdPosterior_c1[-1])

	return np.array(avPosterior_c1) #, np.array(avPosterior_c2), np.array(stdPosterior_c1), np.array(stdPosterior_c2)`