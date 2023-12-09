import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random

def splitColumns(datafile):
   df = pd.read_csv(datafile, delimiter='	')
   df.columns = ['first','second']
   return df['first'],df['second']

def extractFeatures(df,windowSize):
   featureList = []
   threshold = 3.5 * df[:499].std(axis=0)
   arr = df.to_numpy()
   i = 0
   while i < len(arr):
      if arr[i] > threshold:
         j = i
         maximum = arr[i]
         maxInd = i
         while(arr[j] > threshold):
            if arr[j] > maximum:
               maximum = arr[j]
               maxInd = j
            j+=1
         start = max(0, maxInd - windowSize // 2)
         end = min(len(arr), maxInd + windowSize // 2)
         window = arr[start:end]
         std = np.std(window)
         diff = np.max(np.abs(np.diff(window)))
         featureList.append({'std': std, 'diff': diff,'peakIndex' : maxInd })
         i = end
      else:
         i += 1
   featuresDF = pd.DataFrame(featureList)
   return featuresDF
def plotFeatures(features):
   plt.scatter(features['std'], features['diff'], c=features['cluster'], cmap='plasma')
   plt.xlabel('Standard Deviation')
   plt.ylabel('Maximum Difference Between Two Succesive Samples')
   plt.title('K-Means Visualization')
   plt.show()


def plotSignal(Osignal,featuers,windowSize):
     
   

   peaks = featuers['peakIndex'].values
   clusters = featuers['cluster'].values
   unique_clusters = np.unique(clusters)

   valid_indices = (peaks <= 20000) & (np.arange(len(clusters)) <= 20000)

   valid_peaks = peaks[valid_indices]
   valid_clusters = clusters[valid_indices]

   signal = Osignal.iloc[:20000].to_numpy()
   plt.plot(signal, label='Signal')

   cluster_colors = ['red', 'green', 'yellow'] 
   for i, cluster in enumerate(unique_clusters):
      cluster_peaks = valid_peaks[valid_clusters == cluster]
      plt.scatter(cluster_peaks, signal[cluster_peaks], label=f'Cluster {cluster}', color=cluster_colors[i], marker='o')

   plt.title(f'Signal with Clustered Peaks (First {20000} samples)')
   plt.xlabel('Peak')
   plt.ylabel('Amplitude')
   plt.legend()

   plt.show()






def plotAverageNeurons(feauture, data):
   clusters = np.unique(feauture["cluster"])

   color = ["#"+''.join([random.choice('0369CEF') for j in range(6)])
                for i in range(len(clusters))]

   avg_spikes = []
   for cluster in clusters:
       allSpikes = []
       for ind in feauture[feauture["cluster"] == cluster]["peakIndex"]:
           start = ind - 24
           end = ind + 24
           allSpikes.append([data[start:end]])
       avg_spikes.append(np.average(allSpikes, axis = 0)[0])

   indices = np.arange(len(avg_spikes[0]))
   for cluster in clusters:
       plt.plot(indices, avg_spikes[cluster], color=color[cluster], label="neuron "+str(cluster+1))
   plt.xlabel('Time')
   plt.ylabel('Amplitude')
   plt.title('Single Spike Waveform')
   plt.legend()

   plt.grid(True)
   plt.show()

def addClusters(features):
   X = features[['diff', 'std']]
   kmeans = KMeans(n_clusters=3)
   features['cluster'] = kmeans.fit_predict(X)
   return features

def main():
   windowSize = 0.002 * 24414
   windowSize = int(windowSize)
   first = pd.DataFrame()
   second = pd.DataFrame()
   first, second = splitColumns("Data.txt")
   features1 = extractFeatures(first,windowSize)
   features1 = addClusters(features1)
   plotFeatures(features1)
   plotSignal(first,features1,windowSize)
   plotAverageNeurons(features1, first)

   features2 = extractFeatures(second,windowSize)
   features2 = addClusters(features2)
   plotFeatures(features2)
   plotSignal(second,features2,windowSize)

   plotAverageNeurons(features2, second)

   #features1.to_csv("features1.csv",index = False)
   #features2.to_csv("features2.csv", index = False)  
main()
