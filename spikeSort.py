import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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
   X = features[['diff', 'std']]
   kmeans = KMeans(n_clusters=2)
   features['cluster'] = kmeans.fit_predict(X)
   plt.scatter(features['std'], features['diff'], c=features['cluster'], cmap='plasma')
   plt.xlabel('Standard Deviation')
   plt.ylabel('Maximum Difference Between Two Succesive Samples')
   plt.title('K-Means Visualization')
   plt.show()


def plotSignal(Osignal,fileName,windowSize):#Plotting the signal mapping the paeak with a red colored point
     
   # Read the DataFrame from the CSV file
   df = pd.read_csv(fileName)

   # Extract peaks and clusters from the dataframe
   peaks = df['peakIndex'].values
   clusters = df['cluster'].values
   unique_clusters = np.unique(clusters)

   # Plot the signal
   signal = Osignal.to_numpy()  # Replace 'YourSignalColumn' with the actual column name
   plt.plot(signal, label='Signal')
   cluster_colors = ['red', 'green', 'yellow']  # Add more colors as needed

   # Plot peaks with different colors for each cluster
   for i, cluster in enumerate(unique_clusters):
      cluster_peaks = peaks[clusters == cluster]
      plt.scatter(cluster_peaks, signal[cluster_peaks], label=f'Cluster {cluster}', color=cluster_colors[i], marker='o')

   # Customize the plot
   plt.title('Signal with Clustered Peaks')
   plt.xlabel('Peak')
   plt.ylabel('Amplitude')
   plt.legend()

   # Show the plot
   plt.show()

def plotFeatures(features):
   plt.scatter(features['std'], features['diff'], c=features['cluster'], cmap='plasma')
   plt.xlabel('Standard Deviation')
   plt.ylabel('Maximum Difference Between Two Succesive Samples')
   plt.title('K-Means Visualization')
   plt.show()
   return features

def plotAverageNeurons(feauture, data):
   allSpikes1 = []
   for ind in feauture[feauture["cluster"] == 0]["peakIndex"]:
    start = ind - 20
    end = ind + 20
    allSpikes1.append([data[start:end]])
   allSpikes2 = []
   for ind in feauture[feauture["cluster"] == 1]["peakIndex"]:
    start = ind - 20
    end = ind + 20
    allSpikes2.append([data[start:end]])
   avg_spikes = []
   avg_spikes.append(np.average(allSpikes1, axis = 0)[0])
   avg_spikes.append(np.average(allSpikes2, axis = 0)[0])
   indices = np.arange(len(avg_spikes[0]))
   plt.plot(indices, avg_spikes[0], color='blue', label="neuron one")
   plt.plot(indices, avg_spikes[1], color='red', label="neuron two")
   plt.xlabel('Time')
   plt.ylabel('Amplitude')
   plt.title('Single Spike Waveform')
   plt.legend()
   plt.grid(True)
   plt.show()

def addClusters(features):
   X = features[['diff', 'std']]
   kmeans = KMeans(n_clusters=2)
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
   plotSignal(first,"features1.csv",windowSize)
   plotAverageNeurons(features1, first)

   features2 = extractFeatures(second,windowSize)
   features2 = addClusters(features2)
   plotFeatures(features2)
   plotAverageNeurons(features2, second)

   #features1.to_csv("features1.csv",index = False)
   #features2.to_csv("features2.csv", index = False)  
main()
