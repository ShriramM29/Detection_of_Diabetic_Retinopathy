from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from skimage.feature import hog,blob_dog
from skimage import data, exposure
from math import sqrt


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

    thresh = 10 
    im = np.array(Image.open(fpath0),dtype="uint8")
    rmsk =  im0[:,:,0] > thresh  & im0[:,:,1] > thresh & im0[:,:,2] > thresh   
    img = rgb2gray(im)
    return [img,rmsk]

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

# training size in percentage
ts =0.7

msk = np.random.rand(len(Labels)) < ts

Train = Labels[msk]
Test = Labels[~msk]


print('Training Dataset ')
print(Train)

print('Testing Dataset ')
print(Test)


show_classimage(Train,dcls[cls[0]],cls[0])
show_classimage(Train,dcls[cls[1]],cls[1])
show_classimage(Train,dcls[cls[2]],cls[2])
show_classimage(Train,dcls[cls[3]],cls[3])
show_classimage(Train,dcls[cls[4]],cls[4])

hogfeatures(Train,dcls[cls[0]])

dogfeatures(Train,dcls[cls[0]])


exit()
