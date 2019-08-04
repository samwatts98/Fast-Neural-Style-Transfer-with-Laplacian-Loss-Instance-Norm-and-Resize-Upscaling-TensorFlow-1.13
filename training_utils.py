import tensorflow as tf
import numpy as np
from scipy.ndimage.filters import laplace

def calcGram(activations) :

   activations = tf.convert_to_tensor(activations)
   b,h,w,c = activations.shape.as_list()  
	
   #size is used for normalisation.
   size = h * w * c 
	
   #If the batch size is > 1 we calculate gram matricies seperately.
   if b > 1: #USED FOR TRAINING BATCH
      temp = []
      for i, img in enumerate(tf.unstack(activations,axis=0)): #Split a batch into lists of tensors.
         x_ = tf.reshape(img,(h*w,c))  #Reshape the tensor to [h*w, c]
         temp.append(tf.matmul(x_,x_,transpose_a=True)) #Add the gram matrix to temp
      return tf.stack(temp) / size #Return the normalised gram matricies stitched back to shape of [B,C,C]
      
   else : #USED FOR STYLE TARGET / SINGLE ITEM 
      activations = tf.squeeze(activations) #Remove the batch dimension
      x = tf.reshape(activations,(h*w,c)) #Reshape tensor to [h*w, c]
      return tf.matmul(x,x,transpose_a=True) / size # Return the normalised gram matrix
		
def calc_squared_loss(generated, target, normalize=True):
  
   #l2_loss is divided by 2, which we don't need, hence *2
   loss = tf.nn.l2_loss(generated-target)*2
   if normalize:
      b,w,h,c = generated.shape.as_list()
      loss = loss / w*h*c
   return loss
	
def get_laplacian_img(img,pool_size,padding='SAME'):
	
   #Convert image to greyscale.
   img = tf.image.rgb_to_grayscale(img)
    
   pool = tf.nn.max_pool(img, ksize=[1,pool_size,pool_size,1], strides=[1,pool_size,pool_size,1], padding=padding)
    
   return np.squeeze(laplace(pool))

