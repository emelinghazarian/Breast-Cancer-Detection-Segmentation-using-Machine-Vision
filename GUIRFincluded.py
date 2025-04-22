#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 23:40:54 2025

@author: kamand
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:25:55 2024

@author: kamand
"""
import cv2
import numpy as np
import sys
import h5py
import joblib
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.decomposition import PCA

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QWidget,
    QMessageBox,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class ImageDisplayWindow(QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Segmentation Output")
        self.setGeometry(200, 200, 600, 600)
        self.setStyleSheet("background-color: black; color: white")
        self.layout = QVBoxLayout()

        # Display the image
        self.image_label = QLabel(self)
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))
        self.layout.addWidget(self.image_label)

        # Set the layout
        self.setLayout(self.layout)

class GetImageAndAddModule(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Add your patient's Image")
        self.setGeometry(100, 100, 500, 500)
        self.setStyleSheet("background-color: black; color: white")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.images = QHBoxLayout()

        # Title of loading
        self.image_label = QLabel("Load an image of your patient")
        self.image_label.setStyleSheet(
            """
            background-color: #1b2b2b; 
            color: white;
            border: 2px solid white;
            """
        )
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.clicked_thetopmessage
        self.layout.addWidget(self.image_label)

        # Load Image Button
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold;
                border: 2px solid white;
                padding: 5px;
            }
            """
        )
        self.load_image_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_image_button)

        # Vertical buttons layout
        self.buttons_layout = QVBoxLayout()

        # SVM Button
        self.button1 = QPushButton("SVM")
        self.button1.setStyleSheet(
            """
            QPushButton {
                background-color: #2b2b2b;
                color: white;
                font-weight: bold;
                border: 2px solid white;
                padding: 5px;
            }
            """
        )
        self.button1.clicked.connect(self.SVM)
        self.buttons_layout.addWidget(self.button1)

        # DT Button
        self.button2 = QPushButton("DT")
        self.button2.setStyleSheet(
            """
            QPushButton {
                background-color: #2b2b2b;
                color: white;
                font-weight: bold;
                border: 2px solid white;
                padding: 5px;
            }
            """
        )
        self.button2.clicked.connect(self.DT)
        self.buttons_layout.addWidget(self.button2)

        # RF Button
        self.button3 = QPushButton("RF")
        self.button3.setStyleSheet(
            """
            QPushButton {
                background-color: #2b2b2b;
                color: white;
                font-weight: bold;
                border: 2px solid white;
                padding: 5px;
            }
            """
        )
        self.button3.clicked.connect(self.RF)
        self.buttons_layout.addWidget(self.button3)

        # AdaBoost Button
        self.button4 = QPushButton("AdaBoost")
        self.button4.setStyleSheet(
            """
            QPushButton {
                background-color: #2b2b2b;
                color: white;
                font-weight: bold;
                border: 2px solid white;
                padding: 5px;
            }
            """
        )
        self.button4.clicked.connect(self.AdaBoost)
        self.buttons_layout.addWidget(self.button4)

        # Horizontal Segmentation Button
        self.segmentation_button = QPushButton("Segmentation")
        self.segmentation_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold;
                border: 2px solid white;
                padding: 5px;
            }
            """
        )
        self.segmentation_button.clicked.connect(self.segmentation)
        self.images.addLayout(self.buttons_layout)
        self.images.addWidget(self.segmentation_button)

        self.layout.addLayout(self.images)
        self.central_widget.setLayout(self.layout)

        # Load the trained models
        self.svm_model = joblib.load('model.joblib')
        self.rf_model = joblib.load('RF_model.joblib')
        self.dt_model = joblib.load('decision_tree_model_2.joblib')
        self.ada_model = joblib.load('best_ada_model_random.joblib')
        self.pca = joblib.load('pca_model.joblib')
        self.pca_2 = joblib.load('pca_model_2.joblib')
        self.vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.vgg2_model = load_model('vgg16_model.h5')
        
        # Update with your model path RF is an example here

    def show_message(self, message):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(message)
        msg_box.setWindowTitle("Hows The Import Going? ")
        msg_box.setStyleSheet("background-color: #1b2b2b; color: white")
        msg_box.exec_()

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
            options=options,
        )
        if file_path:
            self.file_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(400, 400))
            self.show_message("Image Imported Successfully!")
        else:
            self.show_message("Couldn't Get The Image From This Destination")

    def clicked_thetopmessage(self, event):
        self.load_image()

    def ask_continue_or_quit(self, action_name):
        reply = QMessageBox.question(
            self,
            "continue or quit?",
            f"{action_name} executed successfully! \n\n Do you want to continue? or may I quit.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply == QMessageBox.Yes:
            self.show_message(
                "You chose to continue you may choose another algorithm\n press Ok and start again!"
            )
        else:
            QApplication.quit()

    def segmentation(self):
        if hasattr(self, "file_path") and self.file_path:
            # Load the image
            original_image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)

            # Perform basic segmentation (e.g., edge detection)
            segmented_image = cv2.Canny(original_image, 100, 200)

            # Save the segmented output to a temporary file
            segmented_image_path = "segmented_output.jpg"
            cv2.imwrite(segmented_image_path, segmented_image)

            # Open a new window to display the segmented image
            self.segmentation_window = ImageDisplayWindow(segmented_image_path)
            self.segmentation_window.show()
        else:
            self.show_message("No image loaded. Please load an image first!")


            
    
    def SVM(self):
        if hasattr(self, "file_path") and self.file_path:
            image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
        
            # Preprocess image
            image_resized = cv2.resize(image, (224, 224))
            image_resized = np.expand_dims(image_resized, axis=0)
            image_resized = np.expand_dims(image_resized, axis=-1)
            image_resized = np.repeat(image_resized, 3, axis=-1)
            image_resized = preprocess_input(image_resized)

            # Extract VGG features
            vgg_features = self.vgg_model.predict(image_resized)
            vgg_features_flattened = vgg_features.flatten().reshape(1, -1)
            
            print(f"Input shape for PCA: {vgg_features_flattened.shape}")  # Should be (1, 25088)

            # Reduce dimensions with PCA
            print(f"PCA components shape: {self.pca.components_.shape}")  # Should be (100, 25088)

            vgg_features_reduced = self.pca.transform(vgg_features_flattened)
            
            # Predict using SVM
            prediction_svm = self.svm_model.predict(vgg_features_reduced)
            
            # Add prediction logic
            if prediction_svm == 1:
                self.show_message("Your patient is suspicious of malignancy")
            elif prediction_svm == 0:
                self.show_message("Your patient is suspicious of benign cancer")
            else:
                self.show_message("Everything is normal")
            

            self.ask_continue_or_quit("SVM")
        else:
            self.show_message("No image loaded. Please load an image first!")

            
    
   
            
    def DT(self):
      #  self.vgg_model = load_model('vgg_features.h5')

        if hasattr(self, "file_path") and self.file_path:
            
            image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
    

            # Apply a binary threshold to create a mask
            _, mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

            # Display the original image and the mask
            #cv2.imshow("Original Image", image)
            #cv2.imshow("Mask", mask)

            #preprocessing part with vgg or cnn add a image_flattened here used vgg
            # Preprocess image for VGG model
            image_resized = cv2.resize(mask, (224, 224))  
            image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension
            image_resized = np.expand_dims(image_resized, axis=-1)
            image_resized = np.repeat(image_resized, 3, axis=-1)
            image_resized = preprocess_input(image_resized)  # Preprocess for VGG
            vgg_features = self.vgg2_model.predict(image_resized)
            vgg_features_flattened = vgg_features.flatten().reshape(1, -1)
            
            prediction_dt = self.dt_model.predict(vgg_features_flattened)
            
            bool_2 = prediction_dt
            
            """adding a value to the num with the model_prediction"""
            if bool_2 == 1 :
                self.show_message("your patient is suspicious of malignancy")
            elif bool_2 == 0:
                self.show_message("your patient is suspicious of Benign cancer")
            else :
                self.show_message("everything is normal")
                
                
            self.ask_continue_or_quit("DT")
        
        else :
            self.show_message("No image loaded. Please load an image first!")
        

    def RF(self):
     #   self.vgg_model = load_model('vgg_features.h5')

        if hasattr(self, "file_path") and self.file_path:
            
            image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
            #preprocessing part with vgg or cnn add a image_flattened here used vgg
            # Preprocess image for VGG model
            image_resized = cv2.resize(image, (224, 224))  
            image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension
            image_resized = np.expand_dims(image_resized, axis=-1)
            image_resized = np.repeat(image_resized, 3, axis=-1)
            image_resized = preprocess_input(image_resized)  # Preprocess for VGG
            vgg_features = self.vgg_model.predict(image_resized)

            #that will be in the prediction part
            vgg_features_flattened = vgg_features.flatten().reshape(1, -1)
            # Make prediction using the Random Forest model
            prediction_RF = self.rf_model.predict(vgg_features_flattened)

            # Display the prediction result
            bool_3 = prediction_RF
            """adding a value to the num with the model_prediction"""
            if bool_3 == 1 :
                self.show_message("your patient is suspicious of malignancy")
            elif bool_3 == 0:
                self.show_message("your patient is suspicious of Benign cancer")
            else :
                self.show_message("everything is normal")
                
            self.ask_continue_or_quit("RF")                
        else:
            self.show_message("No image loaded. Please load an image first!")
            

    def AdaBoost(self):
     #   self.vgg_model = load_model('vgg_features.h5')

        if hasattr(self, "file_path") and self.file_path:
            
            image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
            #preprocessing part with vgg or cnn add a image_flattened here used vgg
            # Preprocess image for VGG model
            image_resized = cv2.resize(image, (224, 224))  
            image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension
            image_resized = np.expand_dims(image_resized, axis=-1)
            image_resized = np.repeat(image_resized, 3, axis=-1)
            image_resized = preprocess_input(image_resized)  # Preprocess for VGG
            vgg_features = self.vgg_model.predict(image_resized)
            vgg_features_flattened = vgg_features.flatten().reshape(1, -1)



            vgg_features_reduced = self.pca_2.transform(vgg_features_flattened)
            # Predict using Adaboost
            prediction_ada = self.svm_model.predict(vgg_features_reduced)

            # Display the prediction result
            bool_3 = prediction_ada
            """adding a value to the num with the model_prediction"""
            if bool_3 == 1 :
                self.show_message("your patient is suspicious of malignancy")
            elif bool_3 == 0:
                self.show_message("your patient is suspicious of Benign cancer")
            else :
                self.show_message("everything is normal")
         
            
            self.ask_continue_or_quit("AdaBoost")
            
        else:
            self.show_message("No image loaded. Please load an image first!")

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Exit Application?",
            "Are you sure you want to exit?",
            QMessageBox.No | QMessageBox.Yes,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            event.accept()
            QApplication.quit()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GetImageAndAddModule()
    window.show()
    sys.exit(app.exec_())