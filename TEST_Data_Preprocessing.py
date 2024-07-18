import zipfile
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def import_csv(path):
    """
    Imports the csv files using the given path.

    Parameters
    ----------
    path : STRING - Location of the .csv files

    Returns
    -------
    train : DataFrame - Names of the training images
    valid : DataFrame - Names of the validation images
    test : DataFrame - Names of the test images
    """
    train = pd.read_csv(path+'written_name_train_v2.csv')
    valid = pd.read_csv(path+'written_name_validation_v2.csv')
    test = pd.read_csv(path+'written_name_test_v2.csv')
    
    return train, valid, test

def extract_zip(path, destination):
    """
    Extracts the .zip files, containing the images
    
    Parameters
    ----------
    path : String - Location of the .zip files
    destination : String - Location to store the extracted files
    """
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(destination)

    folder_images = os.listdir(destination)
    
def visualize(images_path,train,save_path,name):
    """
    Visualize the 6 first images of the training set
    
    Parameters
    ----------
    images_path : String - Location of the images
    train : DataFrame - Table containing the names of the training images
    save_path : String - Path to save the plot
    name : String - Name of the plot file
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
     
    plt.figure(figsize=(15, 10))

    for i in range(6):
        plt.subplot(2, 3, i+1)
        # We select the pah of the image
        img_dir = images_path+'/train/'+train.loc[i, 'FILENAME']
        # We read the image
        image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        # We show the image and add the label
        plt.imshow(image, cmap = 'gray')
        plt.title(train.loc[i, 'IDENTITY'], fontsize=12)
        plt.axis('off')
    
    plt.subplots_adjust(wspace=0.2, hspace=-0.8)
    plt.show()
    plt.savefig(os.path.join(save_path,name))
    plt.clf()
    
def clean(data):
    """
    Cleans the given DataFrame

    Parameters
    ----------
    data : DataFrame - Table to clean

    Returns
    -------
    data : DataFrame - The cleaned table
    """
    # Delete the NA values
    data.dropna(axis=0, inplace=True)      
    # Delete the unreadable images
    data = data[data['IDENTITY'] != 'UNREADABLE']
    # Put all the labels into lower case letters
    data.loc[:,('IDENTITY')] = data.loc[:,('IDENTITY')].str.upper()
    # Reset the indexes of the datasets
    data.reset_index(inplace = True, drop=True)
    
    return data

def preprocess_image(img):
    """
    Images preprocessing : cropping a shape 256x64 
    
    Parameters
    ----------
    img : np.array - The image to preprocess
    
    Returns
    ----------
    final_img : np.array - The processed image
    """
    # We get the shape of the current image
    (h, w) = img.shape
    # We initialize our new image
    final_img = np.ones([64, 256])*255

    # If the current image is too big (bigger height/width), we crop it
    if w > 256:
        img = img[:, :256]
    if h > 64:
        img = img[:64, :]

    # We put the cropped image into our new image
    final_img[:h, :w] = img
    # We apply the rotation
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

def create_dataset(train,valid,test,train_size,valid_size,test_size,images_path):
    """
    Create the training/validation datasets
    
    Parameters
    ----------
    train : DataFrame - Names of the training images
    valid : DataFrame - Names of the validation images
    train_size : Int - Size of the training set
    valid_size : Int - Size of the validation set
    images_path : String - Location of the images

    Returns
    -------
    train_images : np.array - Training images
    valid_images : np.array - Validation images

    """
    # We initialize the lists
    train_images = []
    valid_images = []
    test_images = []
    
    for i in range(train_size):
        # We get the image
        img_train = images_path+'/train/'+train.loc[i, 'FILENAME']
        image_train = cv2.imread(img_train, cv2.IMREAD_GRAYSCALE)
        # We crop if to the right shape
        image_train = preprocess_image(image_train)
        # We normalize the pixel's values
        image_train = image_train/255.
        # We add the image to the training set
        train_images.append(image_train)

        if i < valid_size:
            # We get the image
            img_valid = images_path+'/validation/'+valid.loc[i, 'FILENAME']
            image_valid = cv2.imread(img_valid, cv2.IMREAD_GRAYSCALE)
            # We crop it to the right shape
            image_valid = preprocess_image(image_valid)
            # We normalize the pixel's values
            image_valid = image_valid/255.
            # We add th image to the validation set
            valid_images.append(image_valid)
        
        if i < test_size:
            # We get the image
            img_test = images_path+'/test/'+test.loc[i, 'FILENAME']
            image_test = cv2.imread(img_test, cv2.IMREAD_GRAYSCALE)
            # We crop it to the right shape
            image_test = preprocess_image(image_test)
            # We normalize the pixel's values
            image_test = image_test/255.
            # We add th image to the validation set
            test_images.append(image_test)

    # We convert the lists to numpy and reshape them
    train_images = np.array(train_images).reshape(-1, 256, 64, 1)
    valid_images = np.array(valid_images).reshape(-1, 256, 64, 1)
    test_images = np.array(test_images).reshape(-1, 256, 64, 1)

    return train_images, valid_images, test_images


def label_to_num(label,alphabet):
    """
    Encodes a natural language label to a vector of numbers

    Parameters
    ----------
    label : String - Label to convert

    Returns
    -------
    num : np.array - Encoded label

    """
    # We initialize the output vector
    label_num = []
    # For each character of the label, we add the corresponding number
    for ch in label:
        label_num.append(alphabet.find(ch))
    
    num = np.array(label_num)
    return num

def num_to_label(num,alphabet):
    """
    Decodes a vector into a natural language label

    Parameters
    ----------
    num : np.array - Vector to decode

    Returns
    -------
    ret : String - Decoded label

    """
    # We initialize the output string
    ret = ""
    # For each number, we add the element of the alphabet at that position
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabet[ch]
    return ret

def preprocess_labels(data,size,max_str_len,alphabet):
    """
    Encode the labels of the given data into vectors

    Parameters
    ----------
    data : DataFrame - Images labels
    size : Int - Number of labels
    max_str_len : Int - Maximum length for a label (used for padding)

    Returns
    -------
    y : np.array - Encoded labels
    """
    # We initialize the matrix
    y = np.ones([size, max_str_len]) * -1
    # We encode each label
    for i in range(size):
        y[i, 0:len(data.loc[i, 'IDENTITY'])]= label_to_num(data.loc[i, 'IDENTITY'],alphabet)
    # We convert to int
    y = y.astype(int)

    return y

def ctc_inputs(data, size, num_of_timestamps):
    """
    Generate the inputs for the CTC Loss function

    Parameters
    ----------
    data : DataFrame - Images labels
    size : Int - Number of labels in the dataset
    num_of_stamp : Int - Maximum length for predicted labels 

    Returns
    -------
    label_len : torch.tensor - Real lengths of the labels
    input_len : torch.tensor - Lengths of the predicted labels
    """
    # Initialization of the arrays
    label_len = np.zeros([size, 1])
    input_len = np.ones([size, 1]) * (num_of_timestamps-2)
    # We add the lengths to label_len
    for i in range(size):
        label_len[i] = len(data.loc[i, 'IDENTITY'])
    # Convertion into int
    label_len = label_len.astype(int)
    input_len = input_len.astype(int)
    # Convertion into torch.tensor
    label_len = torch.from_numpy(label_len)
    input_len = torch.from_numpy(input_len)
    
    return label_len, input_len
    

class HandWrittenDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def data_preprocessing(path_csv,path_images,train_size,valid_size,test_size,batch_size,max_str_len,alphabet,save_path,name):
    """
    Pre-process the whole data, using the previous functions

    Parameters
    ----------
    path_csv : String - Location of the .csv files
    path_zip : String - Location of the .zip files
    destination : String - Location to store the extracted .zip files
    train_size : Int - Size of the training set
    valid_size : Int - Size of the validation set
    batch_size : Int - Size of a batch
    max_str_len : Int - Maximum length of a label
    alphabet : String - Used alphabet for decoding the predictions
    save_path : String - Path to save the visualize plot

    Returns
    -------
    train_images : np.array - Training images
    valid_images : np.array - Validation images
    train_loader : Dataloader - Containing the training values
    valid_loader : Dataloader - Containing the validation values
    """
    # Importation of the csv files
    print("- CSV Files importation")
    train, valid, test = import_csv(path_csv)

    # Plot some training images
    visualize(path_images,train,save_path,name)
    
    # Labels pre-processing 
    print("- Labels cleaning")
    train = clean(train)
    valid = clean(valid)
    test = clean(test)
    
    # The maximum length of a label is 34, and obtained once and the second maximum length is 24
    #  --> to save memory, we delete the label of size 34 (index=39128) from the training set
    train = train.drop(39128)
    train.reset_index(inplace = True, drop = True)
    
    # Images datasets
    print("- Images datasets creation")
    train_images, valid_images, test_images = create_dataset(train,valid,test,train_size, valid_size,test_size,path_images)
    train_images = train_images.astype(np.float32)
    valid_images = valid_images.astype(np.float32)
    test_images = test_images.astype(np.float32)
    
    # Building the dataloader
    print("- Labels preprocessing")
    train_y = preprocess_labels(train,train_size,max_str_len,alphabet)
    valid_y = preprocess_labels(valid,valid_size,max_str_len,alphabet)
    test_y = preprocess_labels(test,test_size,max_str_len,alphabet)
    
    print("- Dataloaders creation")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = HandWrittenDataset(train_images, train_y, transform=transform)
    valid_dataset = HandWrittenDataset(valid_images, valid_y, transform=transform)
    test_dataset = HandWrittenDataset(test_images, test_y, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train, valid, test, train_loader, valid_loader, test_loader

if __name__ == "__main__": 
    # Paths to change in case the files are in another place
    path_zip = '/home/Test Machine'
    destination = '/home/xnmaster/github-classroom/DCC-UAB/deep-learning-project-2024-ai_nndl_group_14/Inputs/'
    print("Extraction of the ZIP files...")
    # Extraction of the zip files ("if" to make sure that we add them once)
    if not os.path.exists(destination+'/validation'):
        extract_zip(path_zip+"validation_v2.zip", destination)
        print("- Validation file successfully extracted.")
    else:
        print("- Validation file already extracted.")
        
    if not os.path.exists(destination+'/train'):
        extract_zip(path_zip+"train_v2.zip", destination)
        print("- Training file successfully extracted.")
    else:
        print("- Training file already extracted.")
        
    if not os.path.exists(destination+'/test'):
        extract_zip(path_zip+"test_v2.zip", destination)
        print("- Test file successfully extracted.")
    else:
        print("- Test file already extracted.")
        
    print("Extraction succesfully completed.")



