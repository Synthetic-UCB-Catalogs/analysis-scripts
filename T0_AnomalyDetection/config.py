import os

class Params:

	FILENAME = './SEVN/MIST/setA/Z0.02/sevn_mist'
	MODELS_FOLDER = os.path.join('./models/', '-'.join(FILENAME.split('/')[1:]))

	SEQ_LEN = 5
	UNITS = 128
	EPOCHS = 20
	STEPS_PER_EPOCH = 100