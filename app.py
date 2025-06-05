#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
import os

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
# import cv2pip install opencv-python
import cv2

def adjust_brightness_contrast(image, alpha=1.0, beta=0):

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def enhance_contrast(image, min_intensity=0, max_intensity=255):

    return cv2.normalize(image, None, min_intensity, max_intensity, cv2.NORM_MINMAX)

def enhance_sharpness(image, alpha=2.5, beta=-1.5, kernel_size=(5, 5)):

    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    return cv2.addWeighted(image, alpha, blurred, beta, 0)

def processed_image(image):
  adjusted_image = adjust_brightness_contrast(image, alpha=1.4, beta=30)
  stretched_image = enhance_contrast(adjusted_image)
  sharpness_image = enhance_sharpness(stretched_image)
  return sharpness_image
#load model
model =load_model("model/xception_21_10_2023_50ep_98acc.h5")

print('@@ Model loaded')

def prediction(image_file):
  predicted_Output = ['Cashew_anthracnose', 'Cashew_gumosis', 'Cashew_healthy', 'Cashew_leaf miner', 'Cashew_red rust',
                      'Cassava_bacterial blight', 'Cassava_brown spot', 'Cassava_green mite', 'Cassava_healthy', 'Cassava_mosaic',
                      'Maize_fall armyworm', 'Maize_grasshoper', 'Maize_healthy', 'Maize_leaf beetle', 'Maize_leaf blight', 'Maize_leaf spot', 'Maize_streak virus',
                      'Tomato_healthy', 'Tomato_leaf blight', 'Tomato_leaf curl', 'Tomato_septoria leaf spot', 'Tomato_verticulium wilt']
  # input_image_path = input("Give the image path :")
  # img = cv2.imread(image_file)
  # img = cv2.resize(img,(128,128))
  test_image = load_img(image_file, target_size = (150, 150)) # load image 
  print("@@ Got Image for prediction")
  img_array = np.array(test_image)
  img_array = processed_image(img_array)
  img_array = img_array.reshape(1,150,150,3)
  
  print("@@ Got Image for prediction")

  
  a = model.predict(img_array)
  pred = np.argmax(a)

  print("Probability ->")
  for i, probability in enumerate(a[0]):
        print(f"{predicted_Output[i]}: {probability:.4f}")
  indices = a.argmax()
  print(indices)
  print(predicted_Output[indices])
  if pred == 0:
    return "Cashew_anthracnose", 'Thesis project/cashew_Anthracnose.html' 
  elif pred == 1:
      return 'Cashew_gumosis', 'Thesis project/cashew_Gummosis.html' 
  elif pred == 2:
      return 'Cashew_healthy', 'Thesis project/cashew_healthy.html'
  elif pred == 3:
      return 'Cashew_leaf miner', 'Thesis project/cashew_leafMiner.html' 
  elif pred == 4:
      return 'Cashew_red rust', 'Thesis project/cashew_redRust.html' 
  elif pred == 5:
      return 'Cassava_bacterial blight', 'Thesis project/cassava_BacterialBlight.html' 
  elif pred == 6:
      return 'Cassava_brown spot', 'Thesis project/cassava_BrownSpot.html' 
  elif pred == 7:
      return 'Cassava_green mite', 'Thesis project/cassava_GreenMIte.html' 
  elif pred == 8:
      return 'Cassava_healthy', 'Thesis project/cassava_healthy.html' 
  elif pred == 9:
      return 'Cassava_mosaic', 'Thesis project/cassava_Mosaic.html' 
  elif pred == 10:
      return 'Maize_fall armyworm', 'Thesis project/maize_fallArmyworm.html' 
  elif pred == 11:
      return 'Maize_grasshoper', 'Thesis project/maize_Grasshopper.html' 
  elif pred == 12:
      return 'Maize_healthy', 'Thesis project/maize_healthy.html' 
  elif pred == 13:
      return 'Maize_leaf beetle', 'Thesis project/maize_leafBeetles.html' 
  elif pred == 14:
      return 'Maize_leaf blight', 'Thesis project/maize_leafBlight.html' 
  elif pred == 15:
      return 'Maize_leaf spot', 'Thesis project/maize_leafSpot.html' 
  elif pred == 16:
      return 'Maize_streak virus', 'Thesis project/maize_Streak.html' 
  elif pred == 17:
      return 'Tomato_healthy', 'Thesis project/tomato_healthy.html' 
  elif pred == 18:
      return 'Tomato_leaf blight', 'Thesis project/tomato_leafBlight.html' 
  elif pred == 19:
      return 'Tomato_leaf curl', 'Thesis project/tomato_leafCurl.html' 
  elif pred == 20:
      return 'Tomato_septoria leaf spot', 'Thesis project/tomato_septoriaLeafSpot.html' 
  elif pred == 21:
      return 'Tomato_verticulium wilt', 'Thesis project/tomato_verticilliumWilt.html' 
  



def pred_crop_dieas(cott_plant):
  test_image = load_img(cott_plant, target_size = (128, 128)) # load image 
  print("@@ Got Image for prediction")
  
  test_image = img_to_array(test_image)
  test_image = np.expand_dims(test_image, axis = 0)
  
  result = model.predict(test_image)
  print('@@ Raw result = ', result)
  
  pred = np.argmax(result) 

  if pred == 0:
    return "Cashew_anthracnose", 'healthy_plant_leaf.html' # if index 0 burned leaf
  elif pred == 1:
      return 'Cashew_gumosis', 'disease_plant.html' # # if index 1
  elif pred == 2:
      return  'healthy_plant.html'  # if index 2  fresh leaf



app = Flask(__name__)
@app.route('/templates')
def predict_disease():

    return render_template('index_predict.html')
# @app.route("/")
# def about():
#         return render_template('/Thesis project/About_plants.html')

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('/Thesis project/index.html')
@app.route("/about")
def about_plants():
    return render_template('Thesis project/About_plants.html')

# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image']
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = prediction(file_path)
              
        return render_template(output_page, pred_output = pred, user_image = file_path)
    
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False,) 
    
    