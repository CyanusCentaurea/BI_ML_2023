# BI_ML_2023
Neural network for predicting dog breeds in pictures.

## Project:    
Building a neural network for classifying pictures, training it and designing it as a telegram bot, to which you can send a picture and receive a prediction in response.

## Dataset:   
[Dog Breed Identification (Kaggle)](https://www.kaggle.com/c/dog-breed-identification/data) (120 classes - dog breeds).    

## Pre-trained model:   
VIT_L_16 [(google-research/vision_transformer (github))](https://github.com/google-research/vision_transformer).

ViT by Google Research Brain Team is a transformer-based visual model originally designed for text tasks. VIT_L_16 represents the input image as a series of patches and receives vector representations for classification. This makes it possible to achieve high classification accuracy when using a relatively small number of parameters (in our case, 123000).
The weights of the pretrained model were frozen, and the classifier was changed to a linear classifier with 120 output classes.

## Data:

* train: 10222 .jpg files
* test: 10357 .jpg files
* .csv with the correspondence of file names to the breed

## Data preprocessing:
* distribution of training images in breed-appropriate directories for applying the datasets.ImageFolder method;   
* selection of a validation set (1000 random images from training data);   
* transforms:   
  * Resize(256) # based on the selected model
  * CenterCrop(224)   
  * ToTensor()   
  * Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # mean and std of millions of Imagenet images that the model has been trained on (works extremely poorly on its own calculations);
  for test data - augmentation (RandomHorizontalFlip, RandomVerticalFlip, RandomRotation)

### Data slice + augmentation with a given probability

![./Screenshot 2023-06-26 154148.jpg](./Screenshot 2023-06-26 154148.jpg)

## Results (5 epochs):
Phase: train, Loss: 0.011821450976235983, Accuracy: 0.9411190631099544, Epoch: 5/5   
Phase: val, Loss: 0.007666406750679016, Accuracy: 0.98, Epoch: 5/5

## [Telegram bot](http://t.me/ram_dog_bot)   
![./Screenshot 2023-06-26 154503.jpg](./Screenshot 2023-06-26 154503.jpg)

### Added:
* bot description;   
* avatar;   
* welcome message (/start) and help (/help);  
* response to an inappropriate data type (text, document);   
* 3 degrees of “confidence” in the answer:   
  * up to 50% (“It’s tricky! I'm not sure, but maybe it is a pug!”)   
  * 51 - 79% (“I think it's a pug!”)   
  * 80 - 100% (“I'm pretty sure it's a pug!”)   