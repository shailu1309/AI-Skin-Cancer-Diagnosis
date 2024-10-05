"""
The 7 classes of skin cancer lesions included in this dataset are:
Melanocytic nevi (nv)
Melanoma (mel)
Benign keratosis-like lesions (bkl)
Basal cell carcinoma (bcc) 
Actinic keratoses (akiec)
Vascular lesions (vas)
Dermatofibroma (df)
"""

import numpy as np
from PIL import Image # PIL.Image if necessary
from sklearn.preprocessing import LabelEncoder
# from tensorflow import keras
from tensorflow.keras.models import load_model
import time


def getPrediction(filename):
    
    classes = ['Actinic keratoses', 'Basal cell carcinoma', 
               'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanoma', 
               'Melanocytic nevi', 'Vascular lesions']
    le = LabelEncoder()
    le.fit(classes)
    le.inverse_transform([2])
    
    
    #Load model
    my_model=load_model("models/skin_lesion_classifier_acc_80.h5")
    
    SIZE = 32 #Resize to same size as training images
    img_path = 'static/images/'+filename
    img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
    
    img = img/255.      #Scale pixel values
    
    img = np.expand_dims(img, axis=0)  #Get it ready as input to the network       
    
    pred = my_model.predict(img) #Predict
    
    #Convert prediction to class name
    pred_class = le.inverse_transform([np.argmax(pred)])[0]
    print("Diagnosis is:", pred_class)
    score = np.round(np.max(pred) *100,2)
    print(f"score is {score}%")
    return pred_class,score

start_time = time.time()
a =getPrediction('ISIC_0026473.jpg')
end_time = time.time()
elapsed_time = end_time -start_time
print(f"Time elapsed is {elapsed_time} seconds")
