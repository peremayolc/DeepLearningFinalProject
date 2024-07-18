import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os

def decode(pred, batch_size, str_len):
    """
    Decode the outputs of the model into the label shape

    Parameters
    ----------
    pred : torch.tensor - Output of the model
    batch_size : Int - Size of a batch
    str_len : Int - Maximum length of a label

    Returns
    -------
    decoded_batch : np.array - Decoded predictions

    """
    # We initialize the decoded predictions
    decoded_batch = np.ones((batch_size, str_len), dtype=int) * (-1)
    # We define the blank symbol
    blank = 0
    # Convert the prediction into array numpy
    pred = pred.cpu().numpy() 
    
    for b in range(batch_size):
        # For each prediction
        previous_letter = blank
        index = 0
        # For each character of the prediction
        for k in range(64):
            letter = pred[b, k]  
            # If the letter is different, and not blank, we add it to the decoded prediction
            if letter != blank:
                if letter != previous_letter:
                    decoded_batch[b, index] = letter.item()
                    previous_letter = letter
                    index += 1
            previous_letter = letter

    return decoded_batch


@torch.no_grad()  # prevent this function from computing gradients
def validate_CRNN(criterion, model, loader, batch_size, valid_label_len, valid_input_len, max_str_len, device,n_valid_batch):
    """
    Applies the given model on the validation set

    Parameters
    ----------
    criterion : torch.nn - Loss function used
    model : CRNN - The model applied
    loader : Dataloader - Training values
    batch_size : Int - Size of a batch
    valid_label_len : torch.tensor - Real lengths of the labels
    valid_input_len : torch.tensor - Lengths of the outputs of the model
    max_str_len : Int - Maximum label length
    device : torch.device - GPU or CPU

    Returns
    -------
    val_loss : Float - Value of the loss 
    accuracy_words : Float - Number of well predicted words / Total number of words
    accuracy_letters : Float - Number of well predicted letters / Total number of letters
    """
    
    # Initialisations
    val_loss = 0
    correct_words = 0
    correct_letters = 0
    n_letters = 0
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
        target_lengths = valid_label_len[(batch*batch_size):((batch+1)*batch_size)]
        input_lengths = valid_input_len[(batch*batch_size):((batch+1)*batch_size)]
        # Application of the loss function
        loss = criterion(output.transpose(0, 1), target, input_lengths, target_lengths)
        
        # Upgrade the loss value
        val_loss += loss.item()
        
        # Computation of the accuracies
        _, pred = torch.max(output.data,dim=2)
        target = target.cpu().numpy()

        pred = decode(pred,batch_size,max_str_len)

        correct_words += np.sum(np.sum((abs(target-pred)),axis=1)==0)
        correct_letters += np.sum(abs(target-pred)==0, where=(target!=-1))

        n_letters += np.sum(target!=-1)
    
    # Average loss over each batch 
    val_loss /= n_valid_batch
    # Average accuracies over each batch
    accuracy_words = correct_words / len(loader.dataset)
    accuracy_letters = correct_letters / n_letters
    
    return val_loss, accuracy_words, accuracy_letters

def train_CRNN(dataloader, model, batch_size, criterion, optimizer, num_epochs, valid_loader, train_label_len, train_input_len, valid_label_len, valid_input_len, max_str_len, device, n_valid_batch):
    """
    Trains the model on the training set

    Parameters
    ----------
    dataloader : Dataloader - Training values
    model : CRNN - Model to apply
    batch_size : Int - Size of a batch
    criterion : torch.nn - Loss function used
    optimizer : torch.optim - Optimizer used
    num_epochs : Int - Number of epochs
    valid_loader : Dataloader - Validation values
    train_label_len : torch.tensor - Real lengths of the training labels
    train_input_len : torch.tensor - Lengths of the output of the model
    valid_label_len : torch.tensor - Real lengths of the validation labels
    valid_input_len : torch.tensor - Lengths of the output of the model
    device : torch.device - GPU or CPU

    Returns
    -------
    best_model : Best state of the model trained
    train_losses : array - Training losses over the epochs
    valid_losses : array - Validation losses over the epochs
    words_acc_val : array - Words accuracies on the validation set
    letters_acc_val : array - Letters accuracies on the validation set
    """
    # Initialisations
    train_losses = []
    valid_losses = []
    words_acc_val = []
    letters_acc_val = []
    best_model = None
    best_validation_loss = 30

    for epoch in range(num_epochs):
        # We need the gradients here
        model.train()

        for batch, (x, y) in enumerate(dataloader):
            # Initialisation of the hidden states of the RNN part
            h_state, c_state = model.init_hidden(batch_size)
            
            # We put the variables into the device memory
            h_state = h_state.to(device)
            if c_state is not None:
                c_state = c_state.to(device)
            
            x = x.to(device)
            y = y.to(device)
            
            # We vanish the gradients
            optimizer.zero_grad()
            # We apply the current model
            y_pred, h_state, c_state = model(x, h_state, c_state)

            # Inputs for the CTC Loss
            target_lengths = train_label_len[(batch*batch_size):((batch+1)*batch_size)]
            input_lengths = train_input_len[(batch*batch_size):((batch+1)*batch_size)]
            # Application of the loss function
            loss = criterion(y_pred.transpose(1,0), y, input_lengths, target_lengths) #.transpose(1, 2)
            # Backward pass
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            #gradient_clipping(model, 0.0001, 1)

            # We update our model
            optimizer.step()

            if batch%500 == 0:
                print({ 'batch': batch, 'epoch': epoch, 'training loss': loss.item()})
        
        # Application of the model on the validation set
        val_loss, accuracy_words, accuracy_letters = validate_CRNN(criterion, model, valid_loader, batch_size, valid_label_len, valid_input_len, max_str_len, device, n_valid_batch)
        
        if val_loss < best_validation_loss:
            best_model = model.state_dict().copy()
            best_validation_loss = val_loss
            
        # Add the needed values to the lists
        train_losses.append(loss.item())
        valid_losses.append(val_loss)
        words_acc_val.append(accuracy_words)
        letters_acc_val.append(accuracy_letters)

        print({ 'epoch': epoch, 'training loss': loss.item(), 'validation loss':val_loss, "Words accuracy":accuracy_words, "Letters accuracy":accuracy_letters})

    return best_model, train_losses, valid_losses, words_acc_val, letters_acc_val


def visualize_results(train_loss,valid_loss,words_acc_val,letters_acc_val,save_path,name):
    """
    Plots the results of the training

    Parameters
    ----------
    train_loss : array - Training losses over the epochs
    valid_loss : array - Validation losses over the epochs
    words_acc_val : array - Words accuracies on the validation set
    letters_acc_val : array - Letters accuracies on the validation set
    save_path : String - Path to save the plot
    name : String - Name of the plot file
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
     
    plt.subplot(1,3,1)
    plt.plot(train_loss, label='Training')
    plt.plot(valid_loss, label='Validation')
    plt.title("Losses")
    plt.legend()
    
    plt.subplot(1,3,2)
    plt.plot(words_acc_val)
    plt.title("Words accuracy")
    
    plt.subplot(1,3,3)
    plt.plot(letters_acc_val)
    plt.title("Letters accuracy")
    
    plt.savefig(os.path.join(save_path,name))
    plt.clf()