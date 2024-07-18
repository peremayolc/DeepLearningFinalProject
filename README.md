[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/jPcQNmHU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=14965960&assignment_repo_type=AssignmentRepo)
# Handwriting Recognition
## 1 - Context
In this project, we are going to train a neural network in order to recognize words from handwritten data. The input dataset is composed of labelled images (that one can find in the "Inputs" folder, or via this [link](https://www.kaggle.com/code/samfc10/handwriting-recognition-using-crnn-in-keras/input)). The data is composed of 2 types of files : .zip files containing the images, and .csv files linking the images to their label. The folder "Inputs" is then composed of the 3 .csv files, as well as 3 sub-folders, which correspond to the extracted .zip files. 

## 2 - Preprocessing
The data provided is not cleaned yet. In fact, one can find remaining NA values, but also some images labelled 'UNREADABLE', which can't be used for our training. As a consequence, the first step of our work was to preprocess the 3 datasets provided (Training, Validation and Test) : delete the NA and 'UNREADABLE' labels, and turn all of them into upper case. For the images, we defined a standardized size of 64x256, obtained by cropping the current images.

After this cleaning operation, we converted the labels into vectors. To do that, we first looked at the sizes of the labels, and found out that the maximum length in the dataset is 34. But, this value is really far from the rest of the labels. In fact, all the others are of size < 24.
We decided to put this special value apart, and consider a maximum length of 24, to save memory and computing time. 

Now, each label can be converted into a vector of size 24, with numbers corresponding to the characters of the word (and with a padding value of -1 for shorter words). For this convertion, we used 30 different characters : " ABCDEFGHIJKLMNOPQRSTUVWXYZ-'" (+1 for the CTC pseudo blank).

## 3 - Models
Considering the model, we have implemented 1 model, with 2 versions : 
  -  [CRNN model using LSTM](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/blob/main/best_model_LSTM.pth)
  -  [CRNN model using GRU](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/blob/main/best_model_GRU.pth)

For each Convolutional and Recurrent Neural Networks, we first apply 3 convolutional layers with 3x3 filters, ReLU activation, same padding, and batch normalization, followed by a linear layer, with ReLU activation, in order to obtain a final activation map of 64x64. This map will then pass through 2 LSTM bi-directional recurrent layers, of 512 neurons. 
The final activation map obtained has a size 64x1024, which leads to an output of shape 64x30, by applying a linear transformation, and a Log-Softmax.

The final map is finally compared to the label using the CTC Loss. Here, the 64x30 output corresponds to 64 probability distributions (over the 30 possible characters). Using those probability distributions, we can obtain a predicted word. For example, with the label 'DIMOS', considering the maximum probability character for each line, one could obtain 'DDDDDDDDD_IIIII_MMMMMMMMMM_OOOO_SSSS______' (vector of size 64, '_' being the blank symbol), which leads to 'DIMOS' after the collapse operation.

For the training of those models, we used those parameters :
- Size of the training sample : 64 0000 images
- Size of the validation sample : 6 400 images
- Batch size of 128
- Adam optimizer
- Drop rate of 0.4 on the 2 last convolutional layers, and the recurrent ones
- Learning rate of 0.001
- 10 epochs

Afterwards, we wanted to see if by training a bigger number of samples in the training and validation images, we could improve the results. So for the final model we took 300800 as the train size and 30080 as the validation size. This allowed us to obtain better results (that we will see in the tests). 

For the training of the final model, we used these parameters :
- Size of the training sammple : 300 800 images
- Size of the validation sample : 30 080 images
- Batch size of 128
- Adam optimizer
- Drop rate of 0.4 on the 2 last convolutional layers, and the recurrent ones
- Learning rate of 0.001
- 10 epochs



## 4 - Test of the models

In this part, we have applied the models implemented on the test dataset. With the first training, we obtain good results since the Letter Accuracy is around 90%, and the Word Accuracy, around 75-78%. But, we can see an improvement when using more data. In fact, with the second training, we increase those accuracies to 82% accuracy for the words, and 92% for the letters, which is a good improvement. 

In that part, we also made some particular tests, such as ploting some predictions with its image, or looking at mispredicted images to identify any recurrent aspect of those images.
We also looked at the top-3 most mispredicted letters, and their top predicted letters mistakes. In this part, we observe that, for both versions, the most frequent mistakes are: 'H' often predicted as 'M', 'N' often predicted as 'M', 'O' predicted as 'U', and 'A' predicted as 'R'. 
What we can see here is that, in all the cases, the model mispredicts a letter for a letter having some shape similarities, which is a good observation. This shows us that even when the model fails to predict well the letter, its prediction is almost close to the desired one : the model have learnt something.

At the end, we also applied the models on own written images. To do so, we have written our names on a paper, and applied the model on them. We encountered some problems regarding the format of the images and their sizes. This is because in the dataset our model was trained on, all images have a similar shape (around 256x80) and colour (binarized images). 
To fix this changes and make our predictions good, we used the functions inside the cv2 library, which allow us to binarize them and resize them in a way that the predictions could be done. This happened because we saw poor performance for images that had very large sizes and also that where taken in poor light and not preprocessed as they should be. 

After improving this, we were able to get very good predictions, taking into account the necessities of the function, even if in some cases there are still errors, since the accuracy is not 100%,  if the words are cleary written, it will perform fairly good. 

We tried this with the first model trained, and the predictions were good. Some letter predictions missed, still the accuracy was fine. Afterwards, we tried it with the final model, where we took in almost all the images for training, and we obtain quite similar results, with some letters mispredicted. But again, the overall result is clearly close to the desired one.


## Code structure
The code is composed of different .py files :
- [TEST_Data_Preprocessing.py](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/blob/main/TEST_Data_Preprocessing.py) : containing all the preprocessing functions
- [TEST_models.py](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/blob/main/TEST_models.py) : containing the definition of the models
- [TEST_train.py](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/blob/main/TEST_train.py) : containing training/validation and some ploting functions
- [TEST_test.py](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/blob/main/TEST_test.py) : containing test and ploting functions
- [TEST_main.py](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/blob/main/TEST_main.py) : managing all the previous files

To run the code, first, we have to run the Data_Preprocessing.py file. This will extract the .zip files if it's not already done. If it's done, it will not do anything else. The only thing that has to be changed are the paths "path_zip" and "destination", which correspond, respectively, to the location of the .zip files and the place where to put the extracted folders. But, normally, the inputs are already extracted in the "Inputs" folder, so this extraction is simply here in case one want to use the code, only downloading the .zip files. 

Then, simply running the main.py file will carry out the pre-processing operations, create the models, train and test them. 
This run will print some information in the terminal, such as the number of parameters of each model, the results of the training/test; as well as some plots (that will be stored in a folder "Plots"). At the end, we also print a comparative table of both versions of the model.

Again, some paths may need changes : "path_csv", which has to be the location of the 3 .csv files / "path_images", which has to be the location of the extracted folders (="destination") / and "save_plots", which corresponds to the place where we will save the plots during the training.

Note that for the long training, all the results are stored in the folder "Long Training Results" (plots and terminal results as well). Then, the main.py is written to achieve the first training that we presented (64000 images for training and 6400 for validation), which took around 15 minutes to train on the VM that we used.

As a consequence, the plots and images stored in the "Plots" folder corresponds to the results of this first training. In this folder, one can also find a .txt file containing the terminal results of our last run of this model.

## Contributors
Andreu Gascón (1670919@uab.cat)

Mathias Lommel (Mathias.Lommel@autonoma.cat / mathias.lommel@insa-rennes.fr) 

Pere Mayol (1669503@uab.cat / peremayolc@gmail.com)

---------------------------------------

Neural Networks and Deep Learning

Degree of __Artificial Intelligence__

UAB, 2023-2024
