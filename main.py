#!/usr/bin/env python
#coding=utf8
"""
====================================
 :mod:`main` Main
====================================
.. moduleauthor:: Daewoo Kim
.. note:: note...

설명
=====

This is for analyzing energy consumption of drone,
and propose the neural network model for drone's energy consumption

참고
====
 * https://github.com/rhoowd/energy_drone

관련 작업자
===========

본 모듈은 다음과 같은 사람들이 관여했습니다:
 * Daewoo Kim
 * Se-eun Yoon

 CUDA_VISIBLE_DEVICES=0

"""
import config
import extract_data as ed
import log_result as log
import models
import graph
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

import time

import input_preprocess as ip

FLAGS = config.flags.FLAGS


if __name__ == '__main__':

	dataframe_for_x = pd.read_csv(FLAGS.f_dir+FLAGS.f_n)
	dataframe_for_y = pd.read_csv(FLAGS.f_dir+FLAGS.f_n)

	HISTORY_NUM = FLAGS.n_h

	history_labels = ['vel_x', 'vel_y', 'vel_z']
	x_labels = ['vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch', 'yaw', 'rc0', 'rc1', 'rc2', 'rc3']
	y_label = ['power']

	# make history columns
	for l in history_labels:
		for n in range(HISTORY_NUM):
			x_labels.append(l+"_shifted_by_" + str(n+1))	

	dataframe_for_x = ip.make_history(dataframe_for_x, history_labels, HISTORY_NUM)
	dataframe_for_x = dataframe_for_x.loc[:,x_labels]
	dataframe_for_y = dataframe_for_y.loc[:,y_label]

	dataframe_for_x = dataframe_for_x[1:]
	dataframe_for_y = dataframe_for_y.rolling(window=2).mean()[1:]

	# input normalization
	sc = StandardScaler()
	dataset_for_x = sc.fit_transform(dataframe_for_x)
	dataset_for_y = dataframe_for_y[HISTORY_NUM:].values

	x_data = dataset_for_x
	y_data = dataset_for_y


	log.logger.info("x_shape: " + str(x_data.shape) + ", y_shape:" + str(y_data.shape))

	# Split data for test
	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=FLAGS.test_size, random_state=FLAGS.seed)

	# Create model
	model = models.flexible_model_koo(input_dim=x_data.shape[1], output_dim=1)

	# Start training
	model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=FLAGS.n_e,
			  batch_size=FLAGS.b_s, verbose=FLAGS.verbose)

	# Evaluate the model
	scores = model.evaluate(x_test, y_test)
	log.logger.info("ACC(test):\t" + str(scores[2] * 100) + "%\t" + log.filename + " s" + str(FLAGS.seed) + "\t")
	log.logger.info("MSE(test):\t" + str(scores[1]) + "\t" + log.filename + " s"+ str(FLAGS.seed) + "\t")
	scores = model.evaluate(x_data, y_data)
	log.logger.info("ACC(all):\t" + str(scores[2] * 100) + "%\t" + log.filename + " s" + str(FLAGS.seed) + "\t")
	log.logger.info("MSE(all):\t" + str(scores[1]) + "\t" + log.filename + " s" + str(FLAGS.seed) + "\t")


	# save prediction result
	predictions = model.predict(x_test)
	y_test_t = y_test.reshape((-1, 1))
	result = np.concatenate((y_test_t,predictions),axis=1)
	now = time.localtime()
	s_time = "%02d%02d-%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
	result_file_name = "pred_result/"+str(FLAGS.depth)+ "-" + str(FLAGS.h_size)+"-"+str(FLAGS.b_s)+"-"+str(FLAGS.dropout_rate)+"-"+s_time+".csv"
	np.savetxt(result_file_name, result, delimiter=",")


	# Save model
	model_json = model.to_json()
	with open("result/model/"+log.filename+".json", "w") as json_file:
		json_file.write(model_json)  # serialize model to JSON
	model.save_weights("result/model/"+log.filename+".h5")  # weight
	print("Save model ... done")

	# Make graph
	if FLAGS.graph == 1:
		est = model.predict(x_data)
		graph.draw_graph(x_data[:, -1], y_data, est)
		print("Save graph ... done")
