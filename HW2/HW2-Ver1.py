import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

def build_model_using_depthwiseConv2D():
    ins = keras.layers.Input(shape=(None,None,3),name="Input")
    k   = keras.layers.Input(shape=(1,),name="k")
    
    # Gaussian kernel
    kernel = gaussian_kernel(5, sigma=1)
    kernel = kernel[:, :, np.newaxis, np.newaxis]
    
    # Convolution with Gaussian kernel
    blur = keras.layers.DepthwiseConv2D((5,5), depthwise_initializer=keras.initializers.constant(kernel), use_bias=False, padding='same', activation='linear', name="GaussianBlur")(ins)
    
    # Subtract blurred image from original image
    sharp = keras.layers.Subtract()([ins, blur])
    
    # Add the sharp image with a scaling factor k
    outs = keras.layers.Add()([ins, keras.layers.Multiply()([k, sharp])])
    
    model = keras.Model(inputs=[ins,k],outputs=outs,name="SharpMasking")
    model.trainable = False
    return model

model = build_model_using_depthwiseConv2D()

plt.figure(figsize=(12, 8))
plt.suptitle('Version1')
for j in range(1, 4):
    img = cv2.imread("TestImage{}.jpg".format(j))
    plt.subplot(4, 3, j)
    plt.imshow(img[:,:,[2,1,0]])
    plt.axis(False)
    plt.title('Original')
    
    a = model.predict([img[np.newaxis,:,:,[2,1,0]],np.array([[1.0]])])
    plt.subplot(4, 3, j + 3)
    plt.imshow(np.clip(a,0,255).astype(np.uint8)[0,...])
    plt.axis(False)
    plt.title('k=1')
    
    for i in range(1, 3):
        a = model.predict([img[np.newaxis,:,:,[2,1,0]],np.array([[float(10**i)]])])
        plt.subplot(4, 3, 3*(i+1) + j)
        plt.imshow(np.clip(a,0,255).astype(np.uint8)[0,...])
        plt.axis(False)
        plt.title('k={}'.format(float(10**i)))

plt.tight_layout()
plt.show()
