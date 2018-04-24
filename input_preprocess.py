import pandas as pd
import config

def make_history(data, index_list, history_num):

		history = pd.DataFrame()

		for i in index_list:
			for n in range(history_num):
				data[i+'_shifted_by_'+str(n+1)] = data[i].shift(n+1)

		return data[history_num:]
		# data.drop(data.index[])
		# df.drop(df.index[[1,3]], inplace=True)
		# return history

def shift(data, index_list, shift_num):
	for i in index_list:
		data[i] = data[i].shift(shift_num)

	return data[shift_num:]

def get_current(data):
	data['cur'] = ((data['cur_raw'] / 1024.0)*5000 - 2500) / 100
	data['power'] = data['cur'] * data['vol']

	return data

if __name__ == '__main__':
	x = pd.read_csv("0415233753_log_mod2.csv")

	ip = InputPreprocessor()

	x = ip.make_history(x, ['vel_x', 'vel_y'], 2)

	# x = x[1:]
	print x
	# print y