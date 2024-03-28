# TODO Network configration
SERVER_ADDR = '192.168.0.106'
SERVER_PORT = 24601

CLI_NAME_1 = 'pi-3b-0'
CLI_ADDR_1 = '192.168.0.102'

CLI_NAME_2 = 'pi-3b-1'
CLI_ADDR_2 = '192.168.0.103'

CLI_NAME_3 = 'pi-4b-0'
CLI_ADDR_3 = '192.168.0.104'

CLI_NAME_4 = 'pi-4b-1'
CLI_ADDR_4 = '192.168.0.105'

CLI_NAME_5 = 'nano-0'
CLI_ADDR_5 = '192.168.0.100'

K = 5 # Number of devices # TODO

# Unique clients order # TODO
HOST2IP = {CLI_NAME_1:CLI_ADDR_1, CLI_NAME_2:CLI_ADDR_2, CLI_NAME_3:CLI_ADDR_3, CLI_NAME_4:CLI_ADDR_4, CLI_NAME_5:CLI_ADDR_5}
CLIENTS_CONFIG= {CLI_ADDR_1:0, CLI_ADDR_2:1, CLI_ADDR_3:2, CLI_ADDR_4:3, CLI_ADDR_5:4}
CLIENTS_LIST= [CLI_ADDR_1, CLI_ADDR_2, CLI_ADDR_3, CLI_ADDR_4, CLI_ADDR_5]

# Dataset configration
dataset_name = 'CIFAR10'
home = '..'
dataset_path = home +'/dataset/'+ dataset_name +'/'
N = 50000 # data length

# Model configration
model_cfg = {
	# (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
	'VGG5' : [('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3), ('M', 32, 32, 2, 32*16*16, 0), 
	('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), ('M', 64, 64, 2, 64*8*8, 0), 
	('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), 
	('D', 8*8*64, 128, 1, 64, 128*8*8*64), 
	('D', 128, 10, 1, 10, 128*10)]
}
model_name = 'VGG5'
model_size = 1.28
model_flops = 32.902
total_flops = 8488192
split_layer = [2, 2, 2, 2, 2] #Initial split layers # TODO
model_len = 7

# FL training configration # TODO
R = 5 # FL rounds
LR = 0.01 # Learning rate
B = 100 # Batch size
