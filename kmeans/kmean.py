
# %matplotlib inline
import numpy as np 
import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os



np.random.seed(50)
clusters = 3
states = 200
dataset = np.random.rand(2000,2)


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)




class Kmean:
  def __init__(self,n_clusters,iterations):
    self.clusters = n_clusters
    self.iterations = iterations

  def predict(self,x,y):
    distance = np.zeros(3)
    centroidsX = self.oldCentroids[:,0]
    centroidsY = self.oldCentroids[:,1]

      
    for i in range(3):
        distance[i] = self.distanceCal(centroidsX[i],centroidsY[i],x,y)
    Mincluster = np.argmin(distance,axis=0)
    return Mincluster
  def initCentroids(self,X):
    random_idx = np.random.permutation(X)
    centroids = random_idx[:self.clusters]
    return centroids

  def distanceCal(self,x1,y1,x2,y2):
    dis =  math.pow((x2-x1),2) + math.pow((y2-y1),2)
    return dis

  def initDistance(self,X):

    distance = np.zeros((X.shape[0],self.clusters))
    centroidsX = self.centroids[:,0]
    centroidsY = self.centroids[:,1]

    
    for i in range(X.shape[0]):
      for j in range(3):
        distance[i,j] = self.distanceCal(centroidsX[j],centroidsY[j],X[i,0],X[i,1]) 
    return distance

  def fit(self,X):
    self.centroids = self.initCentroids(X)

    for k in range(self.iterations):
      print("e %d"%k)
      self.oldCentroids = self.centroids
      self.distance = self.initDistance(X)
      Mincluster = np.argmin(self.distance,axis=1)

      newcluster0 = []
      newcluster1 = []
      newcluster2 = []
      for i in range(Mincluster.shape[0]):
        if(Mincluster[i] == 0):
          newcluster0.append(X[i])
        elif (Mincluster[i] == 1):
          newcluster1.append(X[i])
        elif (Mincluster[i] == 2):
          newcluster2.append(X[i])
      nc0=np.array(newcluster0)
      nc1=np.array(newcluster1)
      nc2=np.array(newcluster2)

      plt.scatter(nc0[:,0], nc0[:,1],2,color='red')
      plt.scatter(nc1[:,0], nc1[:,1],3,color='blue')
      plt.scatter(nc2[:,0], nc2[:,1],5,color='gray')
      plt.scatter(self.oldCentroids[:,0],self.oldCentroids[:,1],20,color='black')
      
      #plt.legend()
      filename=('plt/%d.png'%k)
      plt.savefig(filename)
      #plt.show()
      
      plt.clf()
      plt.cla()
     
      #########

      nc0Sum = 0
      nc1Sum = 0
      nc2Sum = 0
      for i in range(nc0.shape[0]):
        nc0Sum = nc0[i] + nc0Sum
      nc0Mean = nc0Sum / nc0.shape[0]

      for i in range(nc1.shape[0]):
        nc1Sum = nc1[i] + nc1Sum
      nc1Mean = nc1Sum / nc1.shape[0]

      for i in range(nc2.shape[0]):
        nc2Sum = nc2[i] + nc2Sum
      nc2Mean = nc2Sum / nc2.shape[0]

      self.centroids[0]= nc0Mean
      self.centroids[1]= nc1Mean
      self.centroids[2]= nc2Mean


# plt.scatter(dataset[:,0], dataset[:,1])


x = Kmean(clusters,states)
x.fit(dataset)
pp=x.predict(0.5,0.7)
plt.scatter(0.5,0.7,color='yellow')
print("Predicted point lies in Region %d"%pp)
#os.system('convert -delay 80 plt/*.png animated chart.gif')
#ani = animation.FuncAnimation(fig, x.fit, interval=1000) 

# anim = FuncAnimation(fig, animate, init_func=init,frames=200, interval=20, blit=True)
# anim.save('sine_wave.gif', writer='imagemagick')

plt.show()
os.system('ffmpeg -r 25 -f image2 -s 640*480 -i plt/%200d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4')



