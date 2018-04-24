import numpy as np
import json
import math


def extract_data_speed(n_history=10, n_sample=-1, filename="test2.dat"):
    """

    :param n_history: number of history it output
    :param n_sample: number of samples (-1: for all data)
    :param filename: filename
    :return:
    """

    f = open("data/energy/"+filename)
    cnt = 0
    s_cnt = 0
    v_arr = []
    ret = np.empty((0, n_history + 1))

    try:
        for line in f:
            if line[0] == "s":
                s_cnt = 0
                del v_arr[:]
                continue
            data = json.loads(line)
            if data['I'] == 0:
                s_cnt = 0
                del v_arr[:]
                continue

            if "loc" in data and data['loc']['alt'] > 0:  # verify data
                cnt += 1
                s_cnt += 1
                # get data
                energy = -1 * data['V'] * data['I']
                v = math.sqrt(pow(data['vx'], 2) + pow(data['vy'], 2))
                v_arr.append(v)

                if s_cnt > n_history:  # make sequence with history
                    v_arr.pop(0)
                    ret_arr = v_arr[:]
                    ret_arr.append(energy/100000000.0)
                    ret = np.append(ret, np.array([ret_arr]), axis=0)

                if 0 < n_sample < cnt-n_history+1:  # cut with number of sample
                    break

    finally:
        f.close()

    return ret


def extract_data_speed_diff(n_history=10, n_sample=-1, filename="test2.dat"):
    """

    :param n_history: number of history it output
    :param n_sample: number of samples (-1: for all data)
    :param filename: filename
    :return:
    """

    f = open("data/energy/"+filename)
    cnt = 0
    s_cnt = 0
    a_arr = []
    v_prev = 0
    ret = np.empty((0, n_history + 2))

    try:
        for line in f:
            if line[0] == "s":
                s_cnt = 0
                v_prev = 0
                del a_arr[:]
                continue
            data = json.loads(line)
            if data['I'] == 0:
                s_cnt = 0
                v_prev = 0
                del a_arr[:]
                continue

            if "loc" in data and data['loc']['alt'] > 0:  # verify data
                cnt += 1
                s_cnt += 1
                # get data
                energy = -1 * data['V'] * data['I']
                v = math.sqrt(pow(data['vx'], 2) + pow(data['vy'], 2))
                a = v - v_prev
                v_prev = v
                a_arr.append(a)

                if s_cnt > n_history:  # make sequence with history
                    a_arr.pop(0)
                    ret_arr = a_arr[:]
                    ret_arr.append(v)
                    ret_arr.append(energy / 100000000.0)
                    ret = np.append(ret, np.array([ret_arr]), axis=0)

                if 0 < n_sample < cnt-n_history+1:  # cut with number of sample
                    break

    finally:
        f.close()

    return ret


def extract_data_velocity(n_history=10, n_sample=-1, filename="test2.dat"):
    """

    :param n_history: number of history it output
    :param n_sample: number of samples (-1: for all data)
    :param filename: filename
    :return:
    """

    f = open("data/energy/"+filename)
    cnt = 0
    s_cnt = 0
    v_arr = []
    v_x = []
    v_y = []
    v_z = []
    ret = np.empty((0, 4*n_history + 1))

    try:
        for line in f:
            if line[0] == "s":
                s_cnt = 0
                del v_arr[:]
                del v_x[:]
                del v_y[:]
                del v_z[:]
                continue
            data = json.loads(line)
            if data['I'] == 0:
                s_cnt = 0
                del v_arr[:]
                del v_x[:]
                del v_y[:]
                del v_z[:]
                continue
            if "loc" in data and data['loc']['alt'] > 0:  # verify data
                cnt += 1
                s_cnt += 1
                # get data
                energy = -1 * data['V'] * data['I']
                v = math.sqrt(pow(data['vx'], 2) + pow(data['vy'], 2))
                v_arr.append(v)
                v_x.append(data['vx'])
                v_y.append(data['vy'])
                v_z.append(data['vz'])

                if s_cnt > n_history:  # make sequence with history
                    v_arr.pop(0)
                    v_x.pop(0)
                    v_y.pop(0)
                    v_z.pop(0)
                    ret_arr = v_arr + v_x + v_y + v_z
                    ret_arr.append(energy/100000000.0)
                    ret = np.append(ret, np.array([ret_arr]), axis=0)

                if 0 < n_sample < cnt-n_history+1:  # cut with number of sample
                    break

    finally:
        f.close()

    return ret


def extract_data_onehot(n_history=10, n_sample=0, filename="test2.dat"):
    """

    :param n_history: number of history it output
    :param n_sample: number of samples
    :param filename: filename
    :return:
    """

    # initialize
    f = open("data/energy/"+filename)
    cnt = 0
    v_arr = []
    ret = np.empty((0, n_history + 1))

    try:
        for line in f:
            data = json.loads(line)
            if "loc" in data and data['loc']['alt'] > 0:  # verify data
                cnt += 1
                # get data
                energy = -1 * data['V'] * data['I']
                energy_label = (energy - 400000000) / 30000000

                v = math.sqrt(pow(data['vx'], 2) + pow(data['vy'], 2))
                v_arr.append(v)

                if cnt > n_history:  # make sequence with history
                    v_arr.pop(0)
                    ret_arr = v_arr[:]
                    ret_arr.append(energy_label)
                    ret = np.append(ret, np.array([ret_arr]), axis=0)

                if 0 < n_sample < cnt-n_history+1:  # cut with number of sample
                    break

    finally:
        f.close()

    return ret


def extract_data_seq(x_dim=1, y_dim=1):

    # Training Data
    x_train = np.empty((0, x_dim))
    y_train = np.empty((0, y_dim))

    for filenum in range(1, 19):
        f = open("data/v_e/" + str(filenum) + ".dat")

        cnt = 0
        try:
            for line in f:
                cnt += 1
                data = json.loads(line)
                if "loc" in data:
                    if data['loc']['alt'] > 0:
                        # v = math.sqrt(pow(data['vx'], 2) + pow(data['vy'], 2))
                        # print '%s\n%s\n%s\n%s\n%s' % (cnt, data['time'], data['vx'], data['vx'], -1 * data['V'] * data['I'] / 1000000.0)
                        # x_train = np.append(x_train, np.array([[data['vx'], data['vy']]]), axis=0)
                        # y_train = np.append(y_train, np.array([[data['V'], data['I']]]), axis=0)
                        v = math.sqrt(pow(data['vx'], 2) + pow(data['vy'], 2))
                        energy = -1 * data['V'] * data['I']
                        x_train = np.append(x_train, np.array([[v]]), axis=0)
                        y_train = np.append(y_train, np.array([[energy]]), axis=0)
        finally:
            f.close()

    return x_train, y_train


def extract_data_debug(n_history=10, n_sample=-1, filename="test2.dat"):
    """

    :param n_history: number of history it output
    :param n_sample: number of samples (-1: for all data)
    :param filename: filename
    :return:
    """

    f = open("data/energy/"+filename)
    cnt = 0
    s_cnt = 0
    v_arr = []
    ret = np.empty((0, n_history + 1))

    try:
        for line in f:
            if line[0] == "s":
                s_cnt = 0
                del v_arr[:]
                continue
            data = json.loads(line)
            if data['I'] == 0:
                s_cnt = 0
                del v_arr[:]
                continue

            if "loc" in data and data['loc']['alt'] > 0:  # verify data
                cnt += 1
                s_cnt += 1
                # get data
                energy = -1 * data['V'] * data['I']
                v = math.sqrt(pow(data['vx'], 2) + pow(data['vy'], 2))
                v_arr.append(v)

                if s_cnt > n_history:  # make sequence with history
                    v_arr.pop(0)
                    ret_arr = v_arr[:]
                    ret_arr.append(energy/100000000.0)
                    ret = np.append(ret, np.array([ret_arr]), axis=0)

                if 0 < n_sample < cnt-n_history+1:  # cut with number of sample
                    break

    finally:
        f.close()

    return ret

if __name__ == '__main__':
    # print "This is for test"
    # xy = extract_data_velocity(n_history=3, n_sample=3)
    # print xy.shape
    # print xy

    print "SPEED"
    xy = extract_data_debug(n_history=10, n_sample=-1,filename="line.dat")
    print xy.shape
    print xy

    # print "DIFF"
    # xy = extract_data_speed_diff(n_history=10, n_sample=1)
    # print xy.shape
    # print xy

