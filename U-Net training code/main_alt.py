import os
import cv2
from keras.utils.np_utils import normalize
from PIL import Image
import numpy as np
import random
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
import imageio
import tensorflow as tf
import glob

############################################################################################################

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if I normalize our inputs beforehand
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)

    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
    c6 = Dropout(0.3)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    # Expansive path
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c5])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.3)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c4])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c3])
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    u10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c2])
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u10)
    c10 = Dropout(0.1)(c10)
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)

    u11 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c10)
    u11 = concatenate([u11, c1], axis=3)
    c11 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u11)
    c11 = Dropout(0.1)(c11)
    c11 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c11)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c11)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model
########################################################################################################### 
   
SIZE = 512
image_dataset = []  # Can also use pandas. Using a list format.
mask_dataset = []  # Place holders to define add labels.


image_directory = "a"  #Create directory in folder of project containing images 
mask_directory = "b"   #Create directory in folder of project containing masks 

#Could probably improve the two functions below using os.join
images = os.listdir(image_directory)
for i, image_name in enumerate(images):
    if (image_name.split('.')[1] == 'png'):
        image = Image.open(image_name)
        image = image.resize((512,512))
        image_dataset.append(np.array(image))
        print(image)

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'png'):
        image = Image.open(image_name)
        image = image.convert('L')
        image = image.resize((512,512))
        mask_dataset.append(np.array(image))
        print(image)

    # Normalize images
    image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
    # D not normalize masks, just rescale to 0 to 1.
    mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) 

    #Create random set of test images (10% of training dataset)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.10, random_state=0)

    # Sanity check, view few images

    image_number = random.randint(0, len(X_train))
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(np.reshape(X_train[image_number], (512, 512)), cmap='gray')
    plt.subplot(122)
    plt.imshow(np.reshape(y_train[image_number], (512, 512)), cmap='gray')
    plt.show()

    ###############################################################
    
    #MAIN MODEL
    IMG_HEIGHT = image_dataset.shape[1]
    IMG_WIDTH = image_dataset.shape[2]
    IMG_CHANNELS = image_dataset.shape[3]

    def get_model():
        return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    model=get_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # If starting with pre-trained weights.
    # model.load_weights('sciatic.hdf5')
    
    #CURRENT ERROR: AttributeError: 'numpy.ndarray' object has no attribute '_assert_compile_was_called'
    
    history = model.fit(X_train, y_train,
                        batch_size=1,
                        verbose=1,
                        epochs=1,
                        validation_data=(X_test, y_test),
                        shuffle=False)

    model.save('test.hdf5')

    ############################################################
    # Evaluate the model

    # evaluate model
    _, acc = model.evaluate(X_test, y_test)
    print("Accuracy = ", (acc * 100.0), "%")

    # plot the training and validation accuracy and loss at each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['acc']
    # acc = history.history['accuracy']
    val_acc = history.history['val_acc']
    # val_acc = history.history['val_accuracy']

    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    ###################################################
    # IOU - I kinda found this better than dice loss for semantic segmentation
    y_pred = model.predict(X_test)
    y_pred_thresholded = y_pred > 0.5

    intersection = np.logical_and(y_test, y_pred_thresholded)
    union = np.logical_or(y_test, y_pred_thresholded)
    iou_score = np.sum(intersection) / np.sum(union)
    print("IoU socre is: ", iou_score)

    #######################################################################
    # Predict on a few images
    #model = get_model()
    #model.load_weights('sciatic_50_plus_100_epochs.hdf5')  # Trained for 50 epochs and then additional 100
    model.load_weights('test.hdf5')  #Trained for 50 epochs

    test_img_number = random.randint(0, len(X_test))
    test_img = X_test[test_img_number]
    ground_truth = y_test[test_img_number]
    test_img_norm = test_img[:, :, 0][:, :, None]
    test_img_input = np.expand_dims(test_img_norm, 0)
    prediction = (model.predict(test_img_input)[0, :, :, 0] > 0.2).astype(np.uint8)

    test_img_other = cv2.imread('data/test_images/02-1_256.png', 0)
    test_img_other = cv2.imread('data/test_images/img8.png', 0)
    test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1), 2)
    test_img_other_norm = test_img_other_norm[:, :, 0][:, :, None]
    test_img_other_input = np.expand_dims(test_img_other_norm, 0)

    # Predict and threshold for values above 0.5 probability
    # Change the probability threshold to low value (e.g. 0.05) for watershed demo.
    prediction_other = (model.predict(test_img_other_input)[0, :, :, 0] > 0.2).astype(np.uint8)

    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :, 0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:, :, 0], cmap='gray')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(prediction, cmap='gray')
    plt.subplot(234)
    plt.title('External Image')
    plt.imshow(test_img_other, cmap='gray')
    plt.subplot(235)
    plt.title('Prediction of external Image')
    plt.imshow(prediction_other, cmap='gray')
    plt.show()
