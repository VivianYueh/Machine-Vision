import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

def build_model():
    ins = keras.layers.Input(shape=(None,None,3),name="Input")
    k   = keras.layers.Input(shape=(1,),name="k")
    kernel = gaussian_kernel(5, sigma=1)
    kernel = kernel[:, :, np.newaxis, np.newaxis]
    yuv_conversion_matrix = np.array([[0.299, 0.587, 0.114],
                                      [-0.14713, -0.28886, 0.436],
                                      [0.615, -0.51499, -0.10001]]).T
    rgb_conversion_matrix = np.linalg.inv(yuv_conversion_matrix)
    # Convert from the RGB color space to the YUV color space
    x = keras.layers.Conv2D(3, (1, 1), kernel_initializer=keras.initializers.constant(yuv_conversion_matrix), padding="same", use_bias=False, name="RGB2YUV")(ins)
    x1  = keras.layers.Conv2D(1,(5,5),kernel_initializer=keras.initializers.constant(kernel),use_bias=False,padding='same',name="GaussianBlur")(x[...,:1])
    x   = keras.layers.Concatenate()([x[...,:1]+(x[...,:1]-x1)*k,x[...,1:]])
    # convert from the YUV color space to the RGB color space
    outs =keras.layers.Conv2D(3,(1,1),kernel_initializer=keras.initializers.constant(rgb_conversion_matrix),padding="same",use_bias=False,name="YUV2RGB")(x)
    model = keras.Model(inputs=[ins,k],outputs=outs,name="Version2")
    model.trainable = False
    return model

model = build_model()

plt.figure(figsize=(12, 8))
plt.suptitle('Version2')
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
