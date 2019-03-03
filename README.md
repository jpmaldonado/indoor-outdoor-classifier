# Classifying indoor and outdorr scenes

Pipeline for image classifier.

- `create_train_test.py` generates images from video frames to produce the classifier. 
    - 2000 frames are read from each video. This amounts to around 1 minute.
    - Skip first 500 frames. This is because some videos have introductory text.
    - Sample 300 from the remaining 1500 frames. We introduce random sampling to avoid correlation between scenes.

- `train_model.py` trains the model in keras. Everything is flattened and passed to a neural network. No CNNs for simplicity. The model obtained has an accuracy of 0.78

- `cli.py` produces the label for a given image, using the trained model.
