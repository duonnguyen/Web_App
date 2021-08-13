import numpy as np

#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
print(matplotlib.get_backend())

from matplotlib import pyplot as plt
plt.plot()

import pandas as pd
import warnings
import seaborn as sns
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from sklearn.cluster import DBSCAN
warnings.filterwarnings("ignore")

def generate_figures(response_json):
    data = response_json['data']
    print(data[0].keys())
    #print(meta_data[0].keys())

    #ts_dataframe = pd.read_csv("Anomaly Detection Data.csv")
    ori_data = data.copy()
    data = []
    for i in range(len(ori_data)):
        if i % 10 == 0:
            data.append(ori_data[i])
    #data = data[:1000]
    columns = list(data[0].keys())
    ts_dataframe = pd.DataFrame(data, columns=columns)
    ts_dataframe.head()

    ## Let's quickly plot and view the series
    Time = columns[1]
    Value = columns[0]
    plt.plot(ts_dataframe[Time], ts_dataframe[Value])
    #ts_dataframe.plot( y = 'Normalized Profit')
    plt.xlabel(Time)
    #plt.xticks([0, 20, 40, 60, 80, 99],[ts_dataframe[Time][0],ts_dataframe[Time][20], ts_dataframe[Time][40], ts_dataframe[Time][60], ts_dataframe[Time][80], ts_dataframe[Time][99]] ,rotation=45)
    plt.ylabel(Value)
    plt.savefig('static/before_anomaly_detection.png')
    plt.close()

    ts_dataframe.set_index(Time, inplace=True)
    ts_dataframe.head()

    ### Approach 3 - By DBSCAN

    clustering1 = DBSCAN(eps=50, min_samples=6).fit(np.array(ts_dataframe[Value]).reshape(-1,1))
    labels = clustering1.labels_

    labels

    outlier_pos = np.where(labels == -1)[0]

    x = []
    y = []
    for pos in outlier_pos:
        x.append(np.array(ts_dataframe[Value])[pos])
        y.append(ts_dataframe[Value].index[pos])

    plt.plot(ts_dataframe[Value].loc[ts_dataframe[Value].index], 'k-')
    plt.plot(y,x,'r*', markersize=8)
    plt.legend(['Actual', 'Anomaly Detected'])

    plt.xlabel(Time)
    #plt.xticks([0, 20, 40, 60, 80, 99],[ts_dataframe.index[0],ts_dataframe.index[20], ts_dataframe.index[40], ts_dataframe.index[60], ts_dataframe.index[80], ts_dataframe.index[99]] ,rotation=45)
    plt.ylabel(Value)
    plt.savefig('static/after_anomaly_detection.png')
    plt.close()

    ## End!