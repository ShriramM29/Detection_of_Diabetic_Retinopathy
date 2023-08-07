from PIL import Image ,ImageDraw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray

from skimage.feature import hog,blob_dog
from skimage import data, exposure
from math import sqrt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix
from os import path

def show_classimage(Train,c,cname):

    cls=Train[ Train['level']== c]
    print(cls)
    print( cls['image'].iloc[3] )
    
    fpath0= "dbase/"+ cls['image'].iloc[0] +".jpeg" 
    fpath1= "dbase/"+ cls['image'].iloc[1] +".jpeg" 
    fpath2= "dbase/"+ cls['image'].iloc[2] +".jpeg" 
    fpath3= "dbase/"+ cls['image'].iloc[3] +".jpeg" 
    
    im0 = np.array(Image.open(fpath0),dtype="uint8")
    im1 = np.array(Image.open(fpath1),dtype="uint8")
    im2 = np.array(Image.open(fpath2),dtype="uint8")
    im3 = np.array(Image.open(fpath3),dtype="uint8")

    fig = plt.figure()    
    ax1 = fig.add_subplot(221)
    plt.imshow(im0)
    ax1.title.set_text(cname +" : Sample 1" )
    
    ax2 = fig.add_subplot(222)
    plt.imshow(im1)
    ax2.title.set_text(cname +" :Sample 2" )
    
    ax3 = fig.add_subplot(223)
    plt.imshow(im2)
    ax3.title.set_text(cname +" :Sample 3" )    
    
    ax4 = fig.add_subplot(224)
    plt.imshow(im3)
    ax4.title.set_text(cname +" :Sample 4" )
    plt.show()


def regionofinteret(fpath):

    thresh = 15 
    im = np.array(Image.open(fpath),dtype="uint8")
    rmsk =  im[:,:,0] > thresh  & im[:,:,1] > thresh & im[:,:,2] > thresh   
    img = rgb2gray(im)
    return [img,rmsk]


def hogfeatures2(base,fname):

    fpath= base + fname+".jpeg" 
    im = np.array(Image.open(fpath),dtype="uint8")
    fd, hog_image = hog(im, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
                    
    hfc, hbin = np.histogram(fd, bins=64,range=(0.0,1.0))
    hfp = hfc[1:]/ np.sum(hfc[1:])    
    return hfp
    

def hogfeatures(Train,c):
    
    cls=Train[ Train['level']== c]
    print(cls)
    print( cls['image'].iloc[3] )
    
    fpath0= "dbase/"+ cls['image'].iloc[0] +".jpeg" 
    
    image = np.array(Image.open(fpath0),dtype="uint8")
    
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()



def dogfeatures2(base, fname):
    
    fpath0= base+ fname +".jpeg" 
    image = np.array(Image.open(fpath0),dtype="uint8")
    
    image_gray = rgb2gray(image)
    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    
    image2 = np.zeros(image_gray.shape);
    
    
    for blob in blobs_dog:
        y, x, r = blob
        y1= round(y-r)
        y2= round(y+r)
        x1 = round(x-r)
        x2= round(x+r)        
        image2[y1:y2, x1:x2] = 255
       
    image2 = image2 > 0     
    RGB=image[image2,:] 
    RGB = RGB/255;
    
    
    hfc, bin_edges = np.histogram(RGB[:,0],  bins=22,range=(0,1)) 
    hfc2, bin_edges = np.histogram(RGB[:,1],  bins=22,range=(0,1)) 
    hfc3, bin_edges = np.histogram(RGB[:,2],  bins=22,range=(0,1)) 
        
    hfp = hfc[1:]/ np.sum(hfc[1:])    
    hfp2 = hfc2[1:]/ np.sum(hfc2[1:])    
    hfp3 = hfc3[1:]/ np.sum(hfc3[1:])    
    
    hfp = np.concatenate((hfp, hfp2,hfp3),axis=0)                
    return hfp
    
             

def readfeatures(cls,K):
    
    hfc1 = np.zeros([K,63]);
    hfc2 = np.zeros([K,63]);    
    
    for num,fname in enumerate(cls['image'],start=0):            
        hfc1[num,:]=hogfeatures2("dbase/",fname)      
        hfc2[num,:]=dogfeatures2("dbase/",fname)    
        if ( num+1 >= K ):
            break;
            
    hfc = np.concatenate((hfc1, hfc2),axis=1)            
    return hfc 
    
    
def normalize(vdata):     
    ncols = vdata.shape[1]  
    mx = np.zeros(ncols)
    mn = np.zeros(ncols) 
    for c in range(ncols):
        mx[c]=np.max(vdata[:,c])
        mn[c]=np.min(vdata[:,c])               
        vdata[:,c] = ( np.double(vdata[:,c]) - mn[c] ) / ( mx[c] - mn[c])    
    return vdata,mx,mn



def dogfeatures(Train,c):
    
    cls=Train[ Train['level']== c]
    print(cls)
    print( cls['image'].iloc[3] )
    
    fpath0= "dbase/"+ cls['image'].iloc[0] +".jpeg" 
    image = np.array(Image.open(fpath0),dtype="uint8")
    
    image_gray = rgb2gray(image)
    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    
        

    fig, ax = plt.subplots(1, 1, figsize=(9, 3), sharex=True, sharey=True)
    
    
    #for idx, (blobs, color, title) in enumerate(sequence):
    ax.set_title('Difference of Gaussian')
    ax.imshow(image)
    for blob in blobs_dog:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)

        ax.add_patch(c)
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()
    
    

Labels = pd.read_csv('Labels.csv')
cls = ("No DR", "Mild",  "Moderate",  "Severe", "Proliferative DR")
dcls = {cls[0]:0, cls[1]:1, cls[2]:2, cls[3]:3, cls[4]:4 }

print ('\nDataset :  Retinapathy ')
print('Class \t Count \n')
print( cls[0],"\t", np.sum( Labels['level']== dcls[cls[0]]  ) )
print( cls[1],"\t", np.sum( Labels['level']== dcls[cls[1]])  )
print( cls[2],"\t", np.sum( Labels['level']== dcls[cls[2]])  )

print( cls[3],"\t", np.sum( Labels['level']== dcls[cls[3]]) )
print( cls[4],"\t", np.sum( Labels['level']== dcls[cls[4]]) )



if (  not path.exists('features.npy')  ) : 

    K=10;
    ydata0=np.zeros([1,K]);
    cls0=Labels[ Labels['level']== dcls[cls[0]] ]
    xdata0=readfeatures(cls0,K)


    K=10;
    ydata1=np.ones([1,K]);
    cls1=Labels[ Labels['level']== dcls[cls[1]] ]
    xdata1=readfeatures(cls1,K)

    K=10;
    ydata2=np.ones([1,K])*2;
    cls2=Labels[ Labels['level']== dcls[cls[2]] ]
    xdata2=readfeatures(cls2,K)

    K=10;
    ydata3=np.ones([1,K])*3;
    cls3=Labels[ Labels['level']== dcls[cls[3]] ]
    xdata3=readfeatures(cls3,K)

    K=10;
    ydata4=np.ones([1,K])*4;
    cls4=Labels[ Labels['level']== dcls[cls[4]] ]
    xdata4=readfeatures(cls4,K)

    Xdata = np.concatenate((xdata0,xdata1,xdata2,xdata3,xdata4),axis=0)        
    Ydata = np.concatenate((ydata0,ydata1,ydata2,ydata3,ydata4),axis=1)    
    
    with open('features.npy', 'wb') as fh:
         np.save(fh, Xdata)
         np.save(fh, Ydata)
    
else :
    with open('features.npy', 'rb') as fh:
        Xdata = np.load(fh)
        Ydata= np.load(fh)



print(Xdata.shape)
print(Ydata.shape)

Ydata=np.ravel(Ydata)
Ydata = Ydata.astype(int)
Xtrain,Xtest,Ytrain,Ytest =train_test_split(Xdata,Ydata,test_size=0.3,random_state=1)

print('\n\n Training Dataset ')
print(Xtrain.shape)
print(Ytrain.shape)


print('\n\n Testing Dataset ')
print(Xtest.shape)
print(Ytest.shape)


clf = MLPClassifier(solver='adam', activation='relu',alpha=1e-4,hidden_layer_sizes=[60,40,20], random_state=1,max_iter=100,learning_rate_init=.01).fit(Xtrain, Ytrain)
Yp=clf.predict(Xtrain)
print('\n classification performance : Training');
print(classification_report(Ytrain,Yp,target_names=cls) );


Yp=clf.predict(Xtest)
print('\n classification performance : Testing');
print(classification_report(Ytest,Yp,target_names=cls) );


conf = plot_confusion_matrix(clf, Xtrain, Ytrain,display_labels=cls) 
conf.ax_.set_title('Training Confusion matrix : Retinopathy')
plt.show()  

conf = plot_confusion_matrix(clf, Xtest, Ytest,display_labels=cls) 
conf.ax_.set_title('Testing Confusion matrix : Retinopathy')
plt.show()  




while True :

    fname = input('Enter your image to test retinopathy:' );    
    xdata=np.zeros([1,126])       
    xdata[0,0:63] =hogfeatures2("",fname)      
    xdata[0,63:126] =dogfeatures2("",fname)    
       
    Yp=clf.predict(xdata)        
    print(Yp)
    print('\nPredicted Class :', cls[Yp[0]])
                
    print("\nContinue : 0/1")    
    c = input('Enter 0/1 :');
    if ( c=='0'):
        print('\nFinished')
        break;
