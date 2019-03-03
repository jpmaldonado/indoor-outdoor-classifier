from keras import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

model = Sequential([
    Flatten(input_shape=(100, 100,3)),
    Dense(128, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
    ])

model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Data generators: read files as needed

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )              


test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

train_generator = train_datagen.flow_from_directory(
    "../data/img/train",
    target_size=(100,100),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    "../data/img/test",
    target_size=(100,100),
    batch_size=32,
    class_mode='binary'
)


## Train the model
model.fit_generator(
    train_generator, 
    steps_per_epoch=2500//32,
    validation_data = test_generator,
    validation_steps = 5
    )


## Final model
model.save_weights("../model.h5")



