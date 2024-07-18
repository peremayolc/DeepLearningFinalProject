import torch
import numpy as np
import matplotlib.pyplot as plt

from TEST_train import decode
from TEST_Data_Preprocessing import num_to_label, preprocess_image
import cv2
import os
from collections import Counter


#from bottle_neck_handling_model.TEST_missclassifications import *


@torch.no_grad()  # prevent this function from computing gradients
def test_CRNN(criterion, model, loader, batch_size, test_label_len, test_input_len, max_str_len, device, alphabet):
    """
    Applies the given model on the test set

    Parameters
    ----------
    criterion : torch.nn - Loss function used
    model : CRNN - The model applied
    loader : Dataloader - Test values
    batch_size : Int - Size of a batch
    test_label_len : torch.tensor - Real lengths of the labels
    test_input_len : torch.tensor - Lengths of the outputs of the model
    max_str_len : Int - maximum label length
    device : torch.device - GPU or CPU
    alphabet : String - Alphabet used for decoding

    Returns
    -------
    test_loss : Float - Value of the loss 
    accuracy_words : Float - Number of well predicted words / Total number of words
    accuracy_letters : Float - Number of well predicted letters / Total number of letters
    n_letters : Int - The total number of letters over the whole dataset
    mispred_prop_letters : Float - Proportion of well predicted letters on a failed word prediction
    mispred_images : array - Misclassified images
    mispred_pred : array - Their predicted labels 
    mispred_target : array - Their true labels 
    """
    
    # Initialisations
    test_loss = 0
    correct_words = 0
    correct_letters = 0
    n_letters = 0
    
    mispred_prop_letters = 0
    mispred_nb_letters = 0
    letter_misclassifications = Counter()

    # Gradients are not needed
    model.eval()
    
    for batch, (data, target) in enumerate(loader):
        # Initialisation of the hidden states of the RNN part of the model
        h_state, c_state = model.init_hidden(batch_size)

        # We put the variables to the GPU's memory
        h_state = h_state.to(device)
        if c_state is not None:
            c_state = c_state.to(device)

        data, target = data.to(device), target.to(device)

        # Application of the model
        output, h_state, c_state = model(data, h_state, c_state)

        # Inputs of the CTC Loss
        target_lengths = test_label_len[(batch*batch_size):((batch+1)*batch_size)]
        input_lengths = test_input_len[(batch*batch_size):((batch+1)*batch_size)]
        # Application of the loss function
        loss = criterion(output.transpose(0, 1), target, input_lengths, target_lengths)
        # Upgrade the loss value
        test_loss += loss.item()

        # Computation of the accuracies
        _, pred = torch.max(output.data,dim=2)
        pred = decode(pred,batch_size,max_str_len)

        target = target.cpu().numpy()

        correct_words += np.sum(np.sum((abs(target-pred)),axis=1)==0)
        correct_letters += np.sum(abs(target-pred)==0, where=(target!=-1))
        
        # We keep in memory the misclassified images
        mispred_index = np.sum((abs(target-pred)),axis=1)!=0
        mispred_images = data[mispred_index,:,:,:]
        
        mispred_pred = pred[mispred_index]
        mispred_target = target[mispred_index]
        
        # Proportion of well predicted letters in a wrong word prediction
        mispred_prop_letters += np.sum(abs(mispred_target-mispred_pred)==0, where=(mispred_target!=-1))
        mispred_nb_letters += np.sum(mispred_target!=-1)
        
        n_letters += np.sum(target!=-1)
        
        # Carry out the misclassification analysis
        batch_misclassifications = analyze_misclassifications(pred, target)

        # update letter_misclassifications counter with the missclassified pairs of all the batches
        letter_misclassifications.update(batch_misclassifications)
        
    # Average loss over each batch (25 batches in the test set) 
    test_loss /= 25
    # Average accuracies over each batch
    accuracy_words = correct_words / len(loader.dataset)
    accuracy_letters = correct_letters / n_letters
    mispred_prop_letters = mispred_prop_letters/mispred_nb_letters
    
    # Now display the top errors per letter
    display_common_misclassifications(letter_misclassifications, alphabet)
    display_top_letter_errors(letter_misclassifications, alphabet, top_n=3)

    

    return test_loss, accuracy_words, accuracy_letters, n_letters, mispred_prop_letters, mispred_images, mispred_pred,mispred_target


def plot_misclassified(mispred_images, mispred_pred,mispred_target,alphabet,save_path,name):
    """
    Plots 6 mispredicted images, with the true and predicted label.
    
    Parameters
    ----------
    mispred_images : array - Some mispredicted images
    mispred_pred : array - Their predicted label
    mispred_target : array - Their true label
    alphabet : String - Alphabet used for decoding
    save_path : String - The path where to save the plot
    name : String - Name of the plot file
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
     
    plt.figure(figsize=(15, 10))

    for i in range(6):
        ax = plt.subplot(2, 3, i+1)
        image = mispred_images[i]
        image = image.squeeze(0)
        image = image.cpu().numpy()
        pred = num_to_label(mispred_pred[i],alphabet)
        target = num_to_label(mispred_target[i],alphabet)

        plt.imshow(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), cmap = 'gray')
        plt.title(target+"/"+pred, fontsize=12)
        plt.axis('off')

    plt.subplots_adjust(wspace=0.2, hspace=-0.8)
    plt.savefig(os.path.join(save_path,name))
    plt.clf()
    
def test_some_images(model, batch_size, loader, max_str_len, alphabet, device, save_path, name):
    """
    Print somes images, with their label and the prediction of the model.
    
    Parameters
    ----------
    model : CRNN - Model to apply
    batch_size : Int - Size of a batch
    loader : Dataloader - Test values
    max_str_len : Int - Maximum length of a label
    alphabet : String - Alphabet sed for decoding
    device : torch.device - GPU or CPU
    save_path : String - Path of the folder to save the plot
    name : Name of the plot
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
     
    data, targets = next(iter(loader))
    data, targets = data.to(device), targets.to(device)
    
    h, c = model.init_hidden(128)
    h = h.to(device)
    if c is not None:
        c = c.to(device)
        
    output, h, c = model(data,h,c)

    _,predictions = torch.max(output.data,dim=2)
    predictions = decode(predictions,batch_size,max_str_len)
    targets = targets.cpu().numpy()
      
    plt.figure(figsize=(16, 6))

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        image = data[i].cpu().numpy().reshape((256,64))
        pred = predictions[i]
        target = targets[i]
        
        pred = num_to_label(pred,alphabet)
        target = num_to_label(target,alphabet)
            
        plt.imshow(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), cmap = 'gray')
        plt.title(target+"/"+pred, fontsize=12)
        plt.axis('off')
        
    plt.subplots_adjust(wspace=0.2, hspace=-0.8)
    plt.savefig(os.path.join(save_path,name))
    plt.clf()
        
def test_own_image(model,dir,names,alphabet,max_str_len,device, save_path,save_name, print_pred:bool = False):
    """
    Apply the model on 3 images made by ourselves.
    
    Parameters 
    ----------
    model : CRNN - Model to apply
    dir : list of paths to images
    names : List - List of the file names
    alphabet : String - Alphabet used for decoding
    max_str_len : Int - Maximum label length
    device : torch.device - GPU or CPU
    save_path : String - Folder to store the plot
    save_name : String - Name of the plot
    """
    k=0
    for name in names:
        # Get the image
        path = os.path.join(dir,name)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # Process the image
        resized_image = cv2.resize(image, (256, 71))
        ret, otsu_thresholded = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = preprocess_image(otsu_thresholded)/255.
        image = image.astype(np.float32)
        
        # Apply the model
        h, c = model.init_hidden(1)
        h = h.to(device)
        if c is not None:
            c = c.to(device)
            
        input = torch.tensor(image)
        input = input.reshape((1, 1, input.shape[0], input.shape[1]))
        input = input.to(device)
            
        pred, h, c = model(input,h,c)

        # Get and decode the prediction
        _, pred = torch.max(pred,dim=2)
        pred = decode(pred,1,max_str_len)
        pred = num_to_label(pred[0],alphabet)
        
        plt.subplot(1, 3, k+1)
        plt.imshow(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), cmap = 'gray')
        plt.title(pred, fontsize=12)
        plt.axis('off')
        k+=1
        if print_pred == True:
            print(f'For the image {name}, the prediction is {pred}.')
            print('--------------------------------------------------------')
    
    plt.subplots_adjust(wspace=0.2, hspace=-0.8)
    plt.savefig(os.path.join(save_path,save_name))
    plt.clf()


def analyze_misclassifications(predictions, true_labels):
    """
    comment: char or letter means the integer returned by the decoder than is then transformed 
    to the actual 'char' or 'letter' based on the alphabet
    """

    letter_misclassifications = Counter()

    # iterate over the the batch sequences from the decoded predictions and targets
    for pred_seq, true_seq in zip(predictions, true_labels):
        pred_seq = [p for p in pred_seq if p != -1] # don't append placeholders indicating unfilled positions (-1) from the decoder
        true_seq = [t for t in true_seq if t != -1]

        # iterate over the letters of the given sequence
        for pred_char, true_char in zip(pred_seq, true_seq):
            if pred_char != true_char:

                # add to the counter class both the true letter and the predicted letter and add +1 to the counter.
                letter_misclassifications[(true_char, pred_char)] += 1 
    
    # return the counter of all the misclasified pair in the given batch
    return letter_misclassifications

def display_common_misclassifications(misclassifications, alphabet,top_n=10):
    """
    comment: char or letter means the integer returned by the decoder than is then transformed 
    to the actual 'char' or 'letter' based on the alphabet
    """

    print("Top Misclassifications:")

    # iterate over the top n pairs with more counts from the whole data
    for (true_char, pred_char), count in misclassifications.most_common(top_n):

        # display the integers of the predictions and targets depending on the position in the alphabet
        true_display = alphabet[true_char] if true_char < len(alphabet) else f"Unknown: {true_char}" # unknown if not in the alphabet
        pred_display = alphabet[pred_char] if pred_char < len(alphabet) else f"Unknown: {pred_char}"
        print(f"True: '{true_display}', Predicted: '{pred_display}', Count: {count}")



def display_top_letter_errors(letter_misclassifications, alphabet, top_n=8):
    """
    comment: char or letter means the integer returned by the decoder than is then transformed 
    to the actual 'char' or 'letter' based on the alphabet
    """
    # Create a dictionary to hold data organized by true letter
    organized_misclassifications = {}

    # iterate over the pairs of letters in the counter
    for (true_letter, pred_letter), count in letter_misclassifications.items():

        #add to the dictionary the true letter if is not in it   
        if true_letter not in organized_misclassifications:
            organized_misclassifications[true_letter] = []

        #append for each letter (key in the dictionary) the predicted letter and the count of the pair
        organized_misclassifications[true_letter].append((pred_letter, count))


    # iterate through the keys of the dictionary
    for true_letter in sorted(organized_misclassifications.keys()):

        mispredictions = organized_misclassifications[true_letter]

        # Sort by count the given key in descending order
        mispredictions.sort(key=lambda x: x[1], reverse=True)

        # display the integers of the targets depending on the position in the alphabet 
        true_display = alphabet[true_letter] if true_letter < len(alphabet) else f"Unknown: {true_letter}"
        print(f"Top mispredictions for '{true_display}':")

        # display the top_n counts of integers of the predictions depending on the position in the alphabet 
        for pred_letter, count in mispredictions[:top_n]:
            pred_display = alphabet[pred_letter] if pred_letter < len(alphabet) else f"Unknown: {pred_letter}"
            print(f"  Predicted: '{pred_display}', Count: {count}")
        print()  # Adds a newline for better readability between letters


    



