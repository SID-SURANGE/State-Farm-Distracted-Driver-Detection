# coding = utf-8

# IMPORTS---------------------------------------------------------------------------------------------------------------
import imports
import load_data
import Preprocessing
import config

# LOAD TRAIN AND TEST DATA----------------------------------------------------------------------------------------------
# IMAGE PATH ON DISK FOR TRAIN AND TEST IMAGES
directory = 'D:\\Springboard\\state-farm-distracted-driver-detection\\imgs\\train'
test_directory = 'D:\\Springboard\\state-farm-distracted-driver-detection\\imgs\\test\\'
classes = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

# CALL LOAD FUNCTIONS FOR IMAGES
training_data = load_data.create_training_data(directory, classes)
# testing_data = load_data.create_testing_data(test_directory)

for i in classes:
    path = imports.os.path.join(directory, i)
    for img in imports.os.listdir(path):
        img_array = imports.cv2.imread(imports.os.path.join(path,img), imports.cv2.IMREAD_COLOR)
        imports.plt.imshow(img_array, cmap='gray')
        imports.plt.show()
        break
    break

#training_data = imports.random.shuffle(training_data)

# PRE PROCESS IMAGES
X_train, X_test, y_train, y_test = Preprocessing.pre_process(training_data)


# MODEL ARCHITECTURE-------------------------------------------------------------------------------------------------
'''
# Sequential CNN model with below architecture with
# 6 CONV layers
# 7 Batch Norm layers
# 3 Maxpool layers
# 5 Dropout layers
# 3 Dense Layers
model = models.Sequential()

# CNN 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(240, 240, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.2))

# CNN 2
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))

# CNN 3
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.5))

# Dense & Output
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
'''

# LOAD MODEL FROM THE SAVED MODEL AND WRIGHT FILE-----------------------------------------------------------------------
# Model reconstruction from JSON file
with open('D:\\Springboard\\Capstone\\Flask\\static\\Model.json', 'r') as f:
    model = imports.model_from_json(f.read())

# Load weights into the new model
model.load_weights('D:\\Springboard\\Capstone\\Flask\\static\\Model-weights.h5')


# SUMMARY OF MODEL LAYERS AND PARAMETERS--------------------------------------------------------------------------------
model.summary()


# COMPILE MODEL---------------------------------------------------------------------------------------------------------
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks = [imports.EarlyStopping(monitor='val_acc', patience=5)]


# FIT MODEL WITH EPOCHS 12 and using CALLBACKS--------------------------------------------------------------------------
results = model.fit(X_train, y_train, batch_size= config.BATCH_SIZE, epochs= config.EPOCHS, verbose=1,
                    validation_data=(X_test,y_test), callbacks=callbacks)
