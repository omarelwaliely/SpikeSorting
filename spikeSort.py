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
         start = max(0, i - windowSize // 2)
         end = min(len(arr), i + windowSize // 2)
         window = arr[start:end]

         std = np.std(window)
         diff = np.max(np.abs(np.diff(window)))
         featureList.append({'std': std, 'diff': diff,'peakIndex' : i })
         i = end
      else:
         i += 1
   featuresDF = pd.DataFrame(featureList)
   return featuresDF
def plotFeatures(features):
   X = features[['diff', 'std']]
   kmeans = KMeans(n_clusters=2)
   features['cluster'] = kmeans.fit_predict(X)
   plt.scatter(features['std'], features['diff'], c=features['cluster'], cmap='viridis')
   plt.xlabel('Standard Deviation')
   plt.ylabel('Maximum Difference between Two Succesive Samples')
   plt.title('K-Means Visualization')
   plt.show()

def main():
   windowSize = 0.002 * 24414
   windowSize = int(windowSize)
   print(windowSize)
   first = pd.DataFrame()
   second = pd.DataFrame()
   first, second = splitColumns("Data.txt")
   features1 = extractFeatures(first,windowSize)
   # features2 = extractFeatures(second,windowSize)
   # features2.to_csv("features2.csv", index = False)  
   plotFeatures(features1)
   features1.to_csv("features1.csv",index = False)
   # plotFeatures(features2)
main()
