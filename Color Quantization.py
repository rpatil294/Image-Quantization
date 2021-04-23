import cv2
import numpy as np

image = cv2.imread('baboon.PNG')
XImg = image.reshape(-1,3)
K = [3,5,10,20]

for k in K:
    Mu=[]
    init=-int(XImg.shape[0]/k)
    for j in range(k):
        Mu.append(np.mean(XImg[init+int(XImg.shape[0]/k):init+2*int(XImg.shape[0]/k)-1],axis=0))
        init=init+int(XImg.shape[0]/k)
    Mu=np.array(Mu)
    itrMu = np.zeros(Mu.shape)
    count = 0
    while ( not (np.array_equal(Mu,itrMu))):
        itrMu = Mu.copy()
        temp=[]
        Mu=Mu.reshape(-1,3)
        for i in range(Mu.shape[0]):
            a=np.linalg.norm((XImg-Mu[i]),axis = 1)
            temp.append(a)
        temp=np.array(temp)
        classification = np.argmin(temp,axis=0)
        for i in range(len(Mu)):
            if i in np.unique(classification):
                Mu[i] = np.round(np.mean(XImg[classification.ravel() == i], axis = 0), 2)
            else:
                r = np.random.randint(1440000, size = (1,1))
                Mu[i]=XImg[r]
        count+= 1
        
    print("No. of iterations to convergence =",count)
    Xout = np.array([Mu[i].ravel() for i in classification])
    Xout = Xout.reshape(image.shape)
    cv2.imwrite('task3_baboon_'+str(k)+'.jpg',Xout)
    print("Image saved for k=",k)
  