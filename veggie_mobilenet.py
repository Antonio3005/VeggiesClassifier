import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from PIL import ImageEnhance
from skimage.io import imread
import matplotlib.pyplot as plt

import os, random, pathlib, warnings, itertools, math
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix

from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

K.clear_session()

# Loading the dataset
dataset_folder = 'Vegetable Images'
train_folder = os.path.join(dataset_folder, "train")
test_folder = os.path.join(dataset_folder, "test")
validation_folder = os.path.join(dataset_folder, "validation")

def count_files(rootdir):
    '''counts the number of files in each subfolder in a directory'''
    for path in pathlib.Path(rootdir).iterdir():
        if path.is_dir():
            print("There are " + str(len([name for name in os.listdir(path) \
                  if os.path.isfile(os.path.join(path, name))])) + " files in " + \
                  str(path.name))

count_files(os.path.join(test_folder))

# Image Processing
selected_vegetable = "Tomato"  
num_of_images = 2       

def preprocess_images():
    j = 1
    for i in range(num_of_images):
        folder = os.path.join(test_folder, selected_vegetable)
        random_image = random.choice(os.listdir(folder))

        img = Image.open(os.path.join(folder, random_image))
        img_duplicate = img.copy()
        plt.figure(figsize=(10, 10))

        plt.subplot(num_of_images, 2, j)
        plt.title(label='Original', size=17, pad='7.0', loc="center", fontstyle='italic')
        plt.imshow(img)
        j += 1

        img_processed = ImageEnhance.Color(img_duplicate).enhance(1.35)
        img_processed = ImageEnhance.Contrast(img_processed).enhance(1.45)
        img_processed = ImageEnhance.Sharpness(img_processed).enhance(2.5)

        plt.subplot(num_of_images, 2, j)
        plt.title(label='Processed', size=17, pad='7.0', loc="center", fontstyle='italic')
        plt.imshow(img_processed)
        j += 1

preprocess_images()

# Data Visualization
# Start exploring the dataset and visualize any class label (for instance, Capsicum)
selected_vegetable_vis = "Capsicum"
rows, columns = 1, 5

display_folder = os.path.join(train_folder, selected_vegetable_vis)
total_images_vis = rows * columns
fig = plt.figure(1, figsize=(20, 10))

for i, j in enumerate(os.listdir(display_folder)):
    img_vis = plt.imread(os.path.join(train_folder, selected_vegetable_vis, j))
    fig = plt.subplot(rows, columns, i + 1)
    fig.set_title(selected_vegetable_vis, pad=11, size=20)
    plt.imshow(img_vis)

    if i == total_images_vis - 1:
        break

#%% Visualize the whole dataset by picking a random image from each class inside the training dataset
images_vis = []

for food_folder_vis in sorted(os.listdir(train_folder)):
    food_items_vis = os.listdir(train_folder + '/' + food_folder_vis)
    food_selected_vis = np.random.choice(food_items_vis)
    images_vis.append(os.path.join(train_folder, food_folder_vis, food_selected_vis))

fig = plt.figure(1, figsize=(15, 10))

for subplot_vis, image_vis in enumerate(images_vis):
    print(image_vis)
    category_vis = image_vis.replace("\\", '/').split('/')[-2]
    imgs_vis = plt.imread(image_vis)
    a, b, c = imgs_vis.shape
    fig = plt.subplot(3, 5, subplot_vis + 1)
    fig.set_title(category_vis, pad=10, size=18)
    plt.imshow(imgs_vis)

plt.tight_layout()
#%%
# Model Building
IMAGE_SIZE = [224, 224]

mobilenet = MobileNetV2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in mobilenet.layers:
    layer.trainable = False

x = mobilenet.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

prediction = Dense(10, activation='softmax')(x)  # Changed to 10 classes

model_vegetable = Model(inputs=mobilenet.input, outputs=prediction)

model_vegetable.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Model Training
train_datagen_vegetable = image.ImageDataGenerator(rescale=1./255,
                                                   shear_range=0.2,
                                                   zoom_range=0.2,
                                                   horizontal_flip=True)

test_datagen_vegetable = image.ImageDataGenerator(rescale=1./255)

training_set_vegetable = train_datagen_vegetable.flow_from_directory(
    train_folder,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical')

test_set_vegetable = test_datagen_vegetable.flow_from_directory(
    test_folder,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical')

class_map_vegetable = training_set_vegetable.class_indices
class_map_vegetable

history_vegetable = model_vegetable.fit_generator(
    training_set_vegetable,
    validation_data=test_set_vegetable,
    epochs=5,
    steps_per_epoch=len(training_set_vegetable),
    validation_steps=len(test_set_vegetable)
)

# Saving model
model_vegetable.save('model_mobilenetV2_vegetable.h5')

# Accuracy model
def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='train accuracy')
    print(history.history['accuracy'])
    print(history.history['val_accuracy'])
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('Accuracy_v1_MobilenetV2_vegetable.png')
    plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'], label="train loss")
    plt.plot(history.history['val_loss'], label="validation loss")
    print(history.history['loss'])
    print(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('Loss_v1_MobilenetV2_vegetable.png')
    plt.show()

plot_accuracy(history_vegetable)
plot_loss(history_vegetable)

# Model layers
print("Total layers in the model : ", len(model_vegetable.layers), "\n")

layers_vegetable = [layer.output for layer in model_vegetable.layers[0:]]
layer_names_vegetable = []
for layer in model_vegetable.layers[0:]:
    layer_names_vegetable.append(layer.name)

print("First layer : ", layer_names_vegetable[0])
print("MobilenetV2 layers : Layer 2 to Layer 311")
print("Our fine-tuned layers : ", layer_names_vegetable[311:314])
print("Final Layer : ", layer_names_vegetable[314])

# Predictions
#%%
import h5py
K.clear_session()
path_to_model_vegetable = 'model_mobilenetV2_vegetable.h5'
print("Loading the model..")
try:
    with h5py.File(path_to_model_vegetable, 'r') as file_h5:
        model_vegetable = load_model(file_h5)
except Exception as e:
    print(f"Error loading the model: {e}")

print("Done!")

#%%
# Validation data accuracy
validation_data_dir_vegetable = 'Vegetable Images/test'

validation_datagen_vegetable = image.ImageDataGenerator(rescale=1. / 255)

validation_generator_vegetable = validation_datagen_vegetable.flow_from_directory(
    validation_data_dir_vegetable,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical')

scores_vegetable = model_vegetable.evaluate_generator(validation_generator_vegetable)
print("Test Accuracy: {:.3f}".format(scores_vegetable[1]))

# Predict output for the model
category_vegetable = {
    0: 'Bean', 1: 'Broccoli', 2:'Cabbage' , 3: 'Capsicum', 4: 'Carrot' , 
    5: 'Cauliflower', 6: 'Cucumber', 7: 'Papaya', 8: 'Potato', 9:'Tomato' 
}

def predict_image_vegetable(filename, model_veg):
    img_veg = image.load_img(filename, target_size=(224, 224))
    img_array_veg = image.img_to_array(img_veg)
    img_processed_veg = np.expand_dims(img_array_veg, axis=0)
    img_processed_veg /= 255.

    prediction_veg = model_veg.predict(img_processed_veg)
    index_veg = np.argmax(prediction_veg)

    plt.title("Prediction - {}".format(category_vegetable[index_veg]))
    plt.imshow(img_array_veg)

def predict_dir_vegetable(filedir, model_veg):
    cols_veg = 3
    pos_veg = 0
    images_veg = []
    total_images_veg = len(os.listdir(filedir))
    rows_veg = total_images_veg // cols_veg + 1

    true_veg = filedir.replace("\\", '/').split('/')[-1]

    for i_veg in sorted(os.listdir(filedir)):
        images_veg.append(os.path.join(filedir, i_veg))

    for subplot_veg, imggg_veg in enumerate(images_veg):
        img_veg = image.load_img(imggg_veg, target_size=(224, 224))
        img_array_veg = image.img_to_array(img_veg)
        img_processed_veg = np.expand_dims(img_array_veg, axis=0)
        img_processed_veg /= 255.
        prediction_veg = model_veg.predict(img_processed_veg)
        index_veg = np.argmax(prediction_veg)

        pred_veg = category_vegetable.get(index_veg)
        if pred_veg == true_veg:
            pos_veg += 1

    acc_veg = pos_veg / total_images_veg
    print("Accuracy for {original}: {:.2f} ({pos}/{total})".format(acc_veg, pos=pos_veg, total=total_images_veg,
                                                                     original=true_veg))

# Single image prediction
predict_image_vegetable(os.path.join(validation_folder, 'Cauliflower/1064.jpg'), model_vegetable)

# Directory accuracy prediction
for i_veg in os.listdir(validation_folder):
    predict_dir_vegetable(os.path.join(validation_folder, i_veg), model_vegetable)

# Confusion matrix
def labels_confusion_matrix_vegetable(validation_folder_veg):
    folder_path_veg = validation_folder_veg

    mapping_veg = {}
    for i_veg, j_veg in enumerate(sorted(os.listdir(folder_path_veg))):
        mapping_veg[j_veg] = i_veg

    files_veg = []
    real_veg = []
    predicted_veg = []

    for i_veg in os.listdir(folder_path_veg):

        true_veg = os.path.join(folder_path_veg, i_veg)
        true_veg = true_veg.replace("\\", '/').split('/')[-1]
        true_veg = mapping_veg[true_veg]

        for j_veg in os.listdir(os.path.join(folder_path_veg, i_veg)):
            img_veg = image.load_img(os.path.join(folder_path_veg, i_veg, j_veg), target_size=(224, 224))
            img_array_veg = image.img_to_array(img_veg)
            img_processed_veg = np.expand_dims(img_array_veg, axis=0)
            img_processed_veg /= 255.
            prediction_veg = model_vegetable.predict(img_processed_veg)
            index_veg = np.argmax(prediction_veg)

            predicted_veg.append(index_veg)
            real_veg.append(true_veg)

    return real_veg, predicted_veg

def print_confusion_matrix_vegetable(real_veg, predicted_veg):
    total_output_labels_veg = 10
    cmap_veg = "turbo"
    cm_plot_labels_veg = [i_veg for i_veg in range(10)]

    cm_veg = confusion_matrix(y_true=real_veg, y_pred=predicted_veg)
    df_cm_veg = pd.DataFrame(cm_veg, cm_plot_labels_veg, cm_plot_labels_veg)
    sns.set(font_scale=1.2)  # for label size
    plt.figure(figsize=(10, 7))  # Adjusted figure size
    s_veg = sns.heatmap(df_cm_veg, fmt="d", annot=True, cmap=cmap_veg)  # font size

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_vegetable.png')
    plt.show()

# ...

# Show matrix
print(validation_folder)
y_true_veg, y_pred_veg = labels_confusion_matrix_vegetable(validation_folder)
print_confusion_matrix_vegetable(y_true_veg, y_pred_veg)
#%%
#predict_image_vegetable(os.path.join(validation_folder, 'Potato/1064.jpg'), model_vegetable)
plot_accuracy(history_vegetable)
plot_loss(history_vegetable)

