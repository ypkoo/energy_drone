import sys
from input_preprocess import *
import pandas as pd

filename = sys.argv[1]
data = pd.read_csv(filename)
data = get_current(data)

data.to_csv(filename.split('.')[0]+"_power.csv", index=False, sep=",")