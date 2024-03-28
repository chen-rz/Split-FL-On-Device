import time
import pickle
import argparse

import logging
logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from Sever import Sever
import config
import utils

parser=argparse.ArgumentParser()
parser.add_argument('--offload', help='Split FL or classic FL mode', type= utils.str2bool, default= False)
args=parser.parse_args()

LR = config.LR
offload = args.offload
first = True # First initializaiton control

logger.info('Preparing Sever.')
sever = Sever(0, config.SERVER_ADDR, config.SERVER_PORT, 'VGG5')
sever.initialize(config.split_layer, offload, first, LR)
first = False

if offload:
	logger.info('Split FL Training')
else:
	logger.info('Classic FL Training')

res = {}
res['training_time'], res['test_acc_record'], res['bandwidth_record'] = [], [], []

for r in range(config.R):
	logger.info('====================================>')
	logger.info('==> Round {:} Start'.format(r))

	s_time = time.time()
	state, bandwidth = sever.train(thread_number= config.K, client_ips= config.CLIENTS_LIST)
	aggregrated_model = sever.aggregate(config.CLIENTS_LIST)
	e_time = time.time()

	# Recording each round training time, bandwidth and test accuracy
	training_time = e_time - s_time
	res['training_time'].append(training_time)
	res['bandwidth_record'].append(bandwidth)

	test_acc = sever.test(r)
	res['test_acc_record'].append(test_acc)

	with open(config.home + '/results/result.pkl','wb') as f:
				pickle.dump(res,f)

	logger.info('Round Finish')
	logger.info('==> Round Training Time: {:}'.format(training_time))

	logger.info('==> Reinitialization for Round : {:}'.format(r + 1))
	if offload:
		agent = None # No RL agent
		split_layers = sever.adaptive_offload(agent, state)
	else:
		split_layers = config.split_layer

	if r > 49:
		LR = config.LR * 0.1

	sever.reinitialize(split_layers, offload, first, LR)
	logger.info('==> Reinitialization Finish')
