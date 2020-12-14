#importing the libraries
import cv2
import numpy as np
import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization


alpha_str = ["A", "D", "H", "J", "T", "U", "W"]
org = (400, 100)
color = (255,255,255)
thickness = 10
font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 1



def preprocess_img(img: np.array)-> np.array:
    """
    A function to preprocess the RGB image
    
    This function resizes and blurs the image and applies the Canny edge
    detection algorithm. It then converts the image to RGB and returns it.

    Parameters
    ----------
    img : np.array
        A numpy array representation of the RGB image .

    Returns
    -------
    A numpy array of the preprocessed image

    """
    #resizing the image
    img = cv2.resize(img, (299,299))
    
    #adding a blur
    img = cv2.blur(img, (5,5), cv2.BORDER_DEFAULT)
    
    img = cv2.Canny(img, 40, 110)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    return img
    


def load_model(fname: str) -> tf.keras.Model:
    """
    A function to instantiate the deep learning model
    
    Args:
    fname : str
        Filename of the model to be loaded.

    Returns
    -------
    None.

    """
        
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape = (299,299,3)))
    model.add(Conv2D(64, kernel_size=(5,5), strides=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(3,3), strides=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    model.load_weights(fname)
    return model



#capturing the video from the camera
vid = cv2.VideoCapture(0) 

#variable to count the frames
counter = 1

pred_text = "a"

#modify this by passing the path of weights file as the argument
model = load_model("path to weights file")

while True:
    
    #reading the frames from the video
    ret, frame = vid.read()
    
    #drawing the rectangle where the hands should be shown
    cv2.rectangle(frame, (50,150), (350,450), (0,255,0), 0)
    
    #the symbol is processed for every 100 frames
    if counter%100 == 0:
        
        #extracting that part of the image as roi (region of interest)
        roi = frame[150:450, 50:350]
        
        #preprocessing the image
        roi = preprocess_img(roi)

        #displaying the preprocessed image
        cv2.imshow("hand", roi)
                
        re_roi = np.reshape(roi, (1,299,299,3))
        
        #predicting the sign from the image
        pred_class = np.argmax(model.predict(re_roi))
        
        #getting the letter
        pred_text = alpha_str[pred_class]
        print(pred_text)
        
     
    #writing the letter on the frame
    frame = cv2.putText(frame, pred_text, org, fontScale, thickness, cv2.LINE_AA)
    
    #showing the frame
    cv2.imshow("face", frame)
        
    #incrementing the frame counter
    counter+=1
    
    
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
    
    
vid.release()
cv2.destroyAllWindows()












#img = cv2.imread("a.jpg")


#cv2.imshow("image", img)
#cv2.waitKey(0)

