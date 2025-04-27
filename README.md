Sentiment Analysis with RNN - README
This project implements a sentiment analysis model for Vietnamese text using a Recurrent Neural Network (RNN). The model consists of three main components: data preprocessing (Phụ lục A), model architecture (Phụ lục B), and training (Phụ lục C). Below are the instructions to set up and run the program.
Prerequisites
Software Requirements

Python 3.8 or higher
pip (Python package manager)

Required Libraries
You need to install the following Python libraries:

pandas: For reading and processing the dataset.
torch: For building and training the RNN model.
pyvi: For Vietnamese text tokenization.

Install the required libraries using the following command:
pip install pandas torch pyvi

Project Structure

data_preprocessing.py: Code for data loading and preprocessing (Phụ lục A).
model.py: Code for the RNN model architecture (Phụ lục B).
train.py: Code for training the model (Phụ lục C).
sentiment_data.csv: The dataset file (not included; you need to provide your own CSV file with text and label columns).

Setup and Running the Program
Step 1: Prepare the Dataset

Ensure you have a sentiment_data.csv file in the project directory.
The CSV file should have two columns: text (Vietnamese text) and label (one of Positive, Negative, Neutral).

Step 2: Run Data Preprocessing (Phụ lục A)
This script loads the dataset, tokenizes the text, and prepares a PyTorch DataLoader.

Save the code from Phụ lục A as data_preprocessing.py.
Run the script to verify that the data is loaded correctly:python data_preprocessing.py


This script will load the dataset and create a DataLoader. No output is generated, but it will raise an error if the dataset is not found or improperly formatted.



Step 3: Define the Model (Phụ lục B)
This script defines the RNN model architecture with an Embedding layer, RNN layer, and Dense layer.

Save the code from Phụ lục B as model.py.
No need to run this file directly; it will be imported in the training step.

Step 4: Train the Model (Phụ lục C)
This script trains the model using the DataLoader and saves the trained model.

Save the code from Phụ lục C as train.py.
Ensure data_preprocessing.py and model.py are in the same directory.
Run the training script:python train.py


The script will train the model for 10 epochs and print the loss for each epoch.
After training, the model will be saved as sentiment_model.pth in the project directory.



Expected Output

During training, you will see the loss printed for each epoch, e.g.:Epoch 1, Loss: 1.098
Epoch 2, Loss: 0.987
...


After training, a file named sentiment_model.pth will be created, containing the trained model weights.

Notes

Ensure the sentiment_data.csv file exists and follows the correct format (columns: text, label).
If you encounter a FileNotFoundError, double-check the path to sentiment_data.csv.
The model assumes a vocabulary size based on the dataset. If your dataset is very large, you may need to adjust the vocab_size in model.py.
Training on CPU is supported by default. To use GPU, ensure CUDA is installed and PyTorch is configured to use GPU.

Troubleshooting

ModuleNotFoundError: Ensure all required libraries (pandas, torch, pyvi) are installed.
CUDA-related errors: If using GPU, ensure your PyTorch installation supports CUDA. Otherwise, set device = torch.device('cpu') in train.py.

