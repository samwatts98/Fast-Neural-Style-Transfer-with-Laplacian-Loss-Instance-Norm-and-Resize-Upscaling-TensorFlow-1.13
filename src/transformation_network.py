
import tensorflow as tf
from tensorflow.nn import conv2d, conv2d_transpose, relu, batch_normalization, l2_loss


#Typical default values used in layer segments for the transformation network are:
#Stride = 2
#Kernel Size = (3,3)
#Contain Relu Layer = True
#When these values are not used they are replaced in the construction of the layers.
def get_network(img, norm_='instance', upscale='resize') :
    #First (9x9) convolution.
    conv1 = conv_layer(img,32,kernel_size=(9,9),stride=1,norm=norm_)
	
    #2 Downsampling convolutions
    conv2 = conv_layer(conv1,64,norm=norm_)
    conv3 = conv_layer(conv2,128,norm=norm_)
    
    #5 Residual blocks
    resid1 = resid_layer(conv3,128,norm=norm_)
    resid2 = resid_layer(resid1,128,norm=norm_)
    resid3 = resid_layer(resid2,128,norm=norm_)
    resid4 = resid_layer(resid3,128,norm=norm_)
    resid5 = resid_layer(resid4,128,norm=norm_)
    
    if upscale == 'resize':
        #2 Upsampling resize convolution layers
        conv4 = resize_conv_layer(resid5,64,norm=norm_)
        conv5 = resize_conv_layer(conv4,32,norm=norm_)
    else:
        conv4 = trans_conv_layer(resid5,64,norm=norm_)
        conv5 = trans_conv_layer(conv4,32,norm=norm_)

    #Final (9x9) convolution, followed by tanh activation and RGB scaling.
    conv6 = conv_layer(conv5,3,stride=1,kernel_size=(9,9),norm=norm_, relu=False)
    return tf.nn.tanh(conv6)*150 +255./2 

#This value initialises a tf.Variable object to be used as kernel filters for convolution layers.
#The shape variable is made up of the 2D kernel filters as the first two dimensions, then the input channel size,
#then the number of filters to use in this convolution layer. (x,y,in_channels,filters)
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype=tf.float32)
	
#The convolution layer generator. This function takes parameters:
#previous_layer = the previous layer feeding into the 2D convolution layer to be created.
#filters = number of kernel filters to apply to the input and number of kernel filters to be learnt by the network
#kernel_size = the 2D shape of each kernel filter in the layer to be generated.
#stride = number of steps to take when sliding the filters over the input (default 2).
#relu = Boolean value, whether to apply a relu activation after normalisation
#padding = the type of padding to apply to the input (default "SAME"). 
#norm = "instance" for instance normalisation, anything else for batch normalisation.
def conv_layer(previous_layer, filters, kernel_size=(3,3),stride=2,relu=True,padding='SAME', norm='instance'):
    
    #Get the output channel size from previous layer
    channels = previous_layer.get_shape().as_list()[3]
   
    #The shape of the output tensor for this convolutional layer
    shape = [kernel_size[0], kernel_size[1],channels, filters]
    
    #Create convolution layer
    conv = conv2d(previous_layer, create_weights(shape), [1,stride,stride,1],padding=padding)
    
    #If using instance norm..
    if norm == 'instance':
        normalised = tf.contrib.layers.instance_norm(conv)
    #Otherwise, we use batch norm..
    else:
        mean,variance = tf.nn.moments(conv,[0,1,2])
        normalised = batch_normalization(conv,mean,variance,None,None,0.0001)
    
    #If we use ReLU, add ReLU layer.
    if relu:
        relu_layer = tf.nn.relu(normalised)
        return relu_layer
    return normalised

#Residual block generator. This function produces a residual block within the network, as explained in the report.
#The network consists of two convolution segments, and the output of which is added to the original input to these convolution segments.
#The idea is if the network doesn't need to apply additional convolution segments, it can bypass this segment and use the original
#input as the output (the 2 convolution layers will need to make weights close to 0).
def resid_layer(previous_layer, filters, kernel_size=(3,3),padding='SAME',norm='instance'):
	
    #Residual blocks add the result of two convolution layers to the input into the block.
    conv1 = conv_layer(previous_layer, filters, kernel_size=kernel_size, stride=1,norm=norm)
    return previous_layer + conv_layer(conv1, filters, kernel_size=kernel_size, stride=1, relu=False,norm=norm)
	
#Implementation of resize convolution. The the function resizes the input proportional to the stride squared using nearest neighbour.
#The network then applies a convolution
#Same parameters as that of conv_layer. 
def resize_conv_layer(previous_layer, filters, kernel_size=(3,3), stride=2, relu=True, padding='SAME', norm='instance' ):
    b,w,h,c = previous_layer.shape.as_list()

    #New shape of activations after resizing:
    new_shape = tf.constant([w*stride**2, h*stride**2])
    #Resize the image by stride^2 as we perform convolution next.
    resized = tf.image.resize_images(previous_layer, new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #Followed by a convolutional layer.
    return conv_layer(resized, filters, kernel_size=kernel_size,stride=stride,padding=padding,relu=relu,norm=norm)

#UNUSED FOR NOW
def trans_conv_layer(previous_layer, filters, kernel_size=(3,3),stride=2,padding='SAME',relu=True, norm='instance'):
    batch,height,width,channels = previous_layer.get_shape().as_list()
    shape = [kernel_size[0], kernel_size[1], filters, channels]
    
    new_shape = tf.convert_to_tensor([batch,(height*stride),(width*stride),filters])
    
    conv = conv2d_transpose(previous_layer, create_weights(shape),new_shape,[1,stride,stride,1],padding=padding)
    
    if norm == 'instance':
        normalised = tf.contrib.layers.instance_norm(conv)
    else:
        mean,variance = tf.nn.moments(conv,[0,1,2])
        normalised = batch_normalization(conv,mean,variance,None,None,0.0001)
    
    if relu:
        relu_layer = tf.nn.relu(normalised)
        return relu_layer
    return normalised
