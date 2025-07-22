# train_model.py
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

data_file = config['Paths']['DATA_FILE']
model_file = config['Paths']['MODEL_FILE']
test_size = float(config['Model']['TEST_SIZE'])
random_state = int(config['Model']['RANDOM_STATE'])