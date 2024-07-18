import os
import random
import cv2
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from IPython import display
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid


from TEST_train import *
from TEST_test import *
from TEST_Data_Preprocessing import *
from TEST_models import *


# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
assert torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":   
    print("Using device:", device)
    ###########################################################################
    ############################# DATA PROCESSING #############################
    ###########################################################################
    print("Pre-processing...")
    # Definition of the paths
    ## Those paths CAN BE CHANGED depending on where are the .csv files and the image folders
    path_csv = '/home/xnmaster/github-classroom/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/Inputs/'
    path_images = '/home/xnmaster/github-classroom/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/Inputs/'
    
    # Sizes of the datasets
    train_size =  64000
    valid_size = 6400
    test_size = valid_size
    batch_size = 128
    n_valid_batch = valid_size / batch_size
    
    # Used alphabet
    alphabet = u" ABCDEFGHIJKLMNOPQRSTUVWXYZ-'"
    # Maximum length of input labels
    max_str_len = 24 
    # Number of characters (+1 for ctc pseudo blank)
    num_of_characters = len(alphabet) + 1 
    # Maximum length of predicted labels
    num_of_timestamps = 64
    # Folder where we will store the plots (CAN BE CHANGED)
    save_plots = '/home/xnmaster/github-classroom/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/Plots/'
    
    train, valid, test, train_loader, valid_loader, test_loader = data_preprocessing(path_csv,path_images,train_size,valid_size,test_size,batch_size,max_str_len,alphabet,save_plots,'Visualize_Images.png')
    
    train_label_len, train_input_len = ctc_inputs(train, train_size, num_of_timestamps)
    valid_label_len, valid_input_len = ctc_inputs(valid, valid_size, num_of_timestamps)
    test_label_len, test_input_len = ctc_inputs(test, test_size, num_of_timestamps)
    
    print("Pre-processing done successfully.")

    ###########################################################################
    ############################### CRNN MODEL ################################
    ###########################################################################
    print("Building CRNN Models...")
    # Parameters of the model
    ## Size of the RNN inputs
    rnn_input_dim = 64
    ## Number of neurons of the RNN layers
    rnn_hidden_dim = 512
    ## Number of RNN layers
    n_rnn_layers = 2
    ## Dimension of the output
    output_dim = num_of_characters
    ## Drop rate
    drop_prob = 0.4

    # Loss function
    criterion = torch.nn.CTCLoss()
    # Initialisation of the model
    model_LSTM = CRNN(rnn_input_dim, rnn_hidden_dim, n_rnn_layers, output_dim, drop_prob, "LSTM").to(device)
    model_GRU = CRNN(rnn_input_dim, rnn_hidden_dim, n_rnn_layers, output_dim, drop_prob, "GRU").to(device)
    # Optimizer
    optimizer_LSTM = optim.Adam(model_LSTM.parameters(), lr=0.001) 
    optimizer_GRU = optim.Adam(model_GRU.parameters(), lr=0.001) 
    print("Models successfully created.")
    print(f"--> LSTM : {get_n_params(model_LSTM)} parameters")
    print(f"--> GRU : {get_n_params(model_GRU)} parameters")
    
    ###########################################################################
    ############################# CRNN TRAINING ###############################
    ###########################################################################
    num_epochs = 10
        
    print("LSTM Training...")
    t = time.time()
    best_LSTM, train_loss_LSTM, valid_loss_LSTM, words_acc_val_LSTM, letters_acc_val_LSTM = train_CRNN(train_loader, model_LSTM, batch_size, 
                                                                        criterion, optimizer_LSTM, num_epochs, valid_loader, 
                                                                        train_label_len, train_input_len, valid_label_len, 
                                                                        valid_input_len, max_str_len, device,n_valid_batch)
    
    time_LSTM = time.time() - t
    torch.save(best_LSTM,'best_model_LSTM.pth')
    model_LSTM.load_state_dict(torch.load('best_model_LSTM.pth'))
    
    visualize_results(train_loss_LSTM,valid_loss_LSTM,words_acc_val_LSTM,letters_acc_val_LSTM,save_plots,'LSTM_Training.png')
    print("Training successfully completed.")
    
    print("GRU Training...")
    t = time.time()
    best_GRU, train_loss_GRU, valid_loss_GRU, words_acc_val_GRU, letters_acc_val_GRU = train_CRNN(train_loader, model_GRU, batch_size, 
                                                                        criterion, optimizer_GRU, num_epochs, valid_loader, 
                                                                        train_label_len, train_input_len, valid_label_len, 
                                                                        valid_input_len, max_str_len, device,n_valid_batch)
    
    time_GRU = time.time() - t
    torch.save(best_GRU,'best_model_GRU.pth')
    model_GRU.load_state_dict(torch.load('best_model_GRU.pth'))
                                                                                                                                
    visualize_results(train_loss_GRU,valid_loss_GRU,words_acc_val_GRU,letters_acc_val_GRU,save_plots,'GRU_Training.png')
    print("Training successfully completed.")
    
    ###########################################################################
    ############################### CRNN TEST #################################
    ###########################################################################
    print("LSTM Test...")
    test_loss_LSTM, test_accuracy_words_LSTM, test_accuracy_letters_LSTM, n_letters_LSTM, mispred_prop_letters_LSTM, mispred_images_LSTM, mispred_pred_LSTM ,mispred_target_LSTM = test_CRNN(criterion, model_LSTM, test_loader, 
                                                                                                                                                                                             batch_size, test_label_len, 
                                                                                                                                                                                             test_input_len, max_str_len, device,alphabet)
    print("Test successfully applied.")
    print(f"--> Accuracy of the model on the {test_size} test images: {test_accuracy_words_LSTM:%}")
    print(f"--> Accuracy of the model on the {n_letters_LSTM} test letters: {test_accuracy_letters_LSTM:%}")
    print(f"--> Average word's proportion well predicted on mispredicted words : {mispred_prop_letters_LSTM:%}")
    
    test_some_images(model_LSTM, batch_size, test_loader, max_str_len, alphabet, device, save_plots, 'Some_predictions_LSTM.png')                                                           
    plot_misclassified(mispred_images_LSTM, mispred_pred_LSTM, mispred_target_LSTM, alphabet,save_plots,'Misclassified_images_LSTM.png')
    
    print("GRU Test...")
    test_loss_GRU, test_accuracy_words_GRU, test_accuracy_letters_GRU, n_letters_GRU, mispred_prop_letters_GRU, mispred_images_GRU, mispred_pred_GRU ,mispred_target_GRU = test_CRNN(criterion, model_GRU, test_loader, 
                                                                                                                                                                                             batch_size, test_label_len, 
                                                                                                                                                                                             test_input_len, max_str_len, 
                                                                                                                                                                                             device, alphabet)
    print("Test successfully applied.")
    print(f"--> Accuracy of the model on the {test_size} test images: {test_accuracy_words_GRU:%}")
    print(f"--> Accuracy of the model on the {n_letters_GRU} test letters: {test_accuracy_letters_GRU:%}")
    print(f"--> Average word's proportion well predicted on mispredicted words : {mispred_prop_letters_GRU:%}")
    
    test_some_images(model_GRU, batch_size, test_loader, max_str_len, alphabet, device, save_plots, 'Some_predictions_GRU.png')  
    plot_misclassified(mispred_images_GRU, mispred_pred_GRU, mispred_target_GRU, alphabet, save_plots,'Misclassified_images_GRU.png')
    
    ###########################################################################
    ############################## OWN IMAGES #################################
    ###########################################################################
    path = '/home/xnmaster/github-classroom/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/Inputs/'
    print("Testing our own images...")
    names = ['name_trial_andreu.jpg','name_trial_mathias.jpg','name_trial_pere.jpg']
    targets = ['ANDREU','MATHIAS','PERE']
    test_own_image(model_LSTM,path,names,alphabet,max_str_len,device,save_plots,'Own_images_LSTM.png',True)
    test_own_image(model_GRU,path,names,alphabet,max_str_len,device,save_plots,'Own_Images_GRU.png',True)
    print("Test successfully done.")
    
    ###########################################################################
    ################################ SUMMUP ###################################
    ###########################################################################
    print("Comparison of the models : ")
    summary = pd.DataFrame({
        "LSTM" : [time_LSTM, 100*test_accuracy_words_LSTM, 100*test_accuracy_letters_LSTM, 100*mispred_prop_letters_LSTM],
        "GRU" : [time_GRU, 100*test_accuracy_words_GRU, 100*test_accuracy_letters_GRU, 100*mispred_prop_letters_GRU]
    })
    
    summary.index = ["Time of training (s)","Word accuracy (%)", "Letter accuracy (%)","(%) of letters in a mispredicted word"]
    print(summary)