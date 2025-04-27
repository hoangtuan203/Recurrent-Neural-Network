Sentiment Analysis with RNN for Vietnamese Text

A simple yet effective sentiment analysis model for Vietnamese text using a Recurrent Neural Network (RNN). The model processes text through an Embedding layer, an RNN layer, and a Dense layer to classify sentiments as Positive, Negative, or Neutral.
📖 Project Overview
This project implements a sentiment analysis pipeline for Vietnamese text using PyTorch. It consists of three main components:

Data Preprocessing (Phụ lục A): Loads, tokenizes, and prepares the dataset for training.
Model Architecture (Phụ lục B): Defines the RNN model with Embedding, RNN, and Dense layers.
Training (Phụ lục C): Trains the model using SGD and saves the trained weights.

The model architecture is as follows:

Embedding: vocab_size × 100
RNN: 100 → 128
Dense: 128 → 3 (output classes: Positive, Negative, Neutral)

🚀 Getting Started
Prerequisites
Ensure you have the following installed:

Python: Version 3.8 or higher
pip: Python package manager

Required Libraries
Install the necessary Python libraries using the command below:
pip install pandas torch pyvi


pandas: For dataset handling
torch: For building and training the model
pyvi: For Vietnamese text tokenization

Project Structure
sentiment-analysis-rnn/
│
├── data_preprocessing.py   # Phụ lục A: Data loading and preprocessing
├── model.py                # Phụ lục B: RNN model architecture
├── train.py                # Phụ lục C: Model training script
├── sentiment_data.csv      # Dataset (not included; provide your own)
├── sentiment_model.pth     # Trained model weights (generated after training)
└── README.md               # Project documentation

Dataset

The dataset should be a CSV file named sentiment_data.csv.
Required columns:
text: Vietnamese text
label: Sentiment label (Positive, Negative, or Neutral)



Example:

text  label



Tối nay hoàn thành dự án, Positive


Tôi không thích cái này, Negative


Thời tiết hôm nay bình thường, Neutral


🛠️ Installation and Setup
Step 1: Clone the Repository
git clone https://github.com/hoangtuan203/Recurrent-Neural-Network.git
cd Recurrent-Neural-Network

Step 2: Install Dependencies
Install the required libraries:
pip install -r requirements.txt

Note: If requirements.txt is not provided, manually install the libraries listed in the Prerequisites section.
Step 3: Prepare the Dataset
Place your sentiment_data.csv file in the project directory. Ensure it follows the format described above.
▶️ Running the Program
1. Data Preprocessing (Phụ lục A)
This script prepares the dataset for training.
python data_preprocessing.py


What it does: Loads the CSV file, tokenizes the text using pyvi, and creates a PyTorch DataLoader.

2. Model Definition (Phụ lục B)
The model architecture is defined in model.py. No need to run this file directly; it will be imported during training.
3. Training the Model (Phụ lục C)
Train the model and save the weights.
python train.py


What it does: Trains the model for 10 epochs using SGD, prints the loss per epoch, and saves the trained model as sentiment_model.pth.

Expected Output
During training, you’ll see the loss for each epoch:
Epoch 1, Loss: 1.098
Epoch 2, Loss: 0.987
...
Epoch 10, Loss: 0.654

After training, a file sentiment_model.pth will be created in the project directory.
📊 Model Architecture
The model consists of three layers:

Embedding Layer: Converts text to vectors (vocab_size × 100)
RNN Layer: Processes sequences (100 → 128 hidden units)
Dense Layer: Outputs sentiment classes (128 → 3)

🖥️ Usage Notes

GPU Support: The script defaults to CPU. To use GPU, ensure CUDA is installed and modify train.py to set device = torch.device('cuda').
Dataset: If sentiment_data.csv is missing or incorrectly formatted, the program will raise a FileNotFoundError or KeyError.
Vocabulary Size: The model dynamically builds the vocabulary from the dataset. For very large datasets, you may need to adjust vocab_size in model.py.

🔧 Troubleshooting



Issue
Solution



ModuleNotFoundError
Ensure all libraries (pandas, torch, pyvi) are installed.


FileNotFoundError
Verify sentiment_data.csv exists in the project directory.


CUDA errors
Set device = torch.device('cpu') in train.py if GPU is unavailable.


📜 License
This project is licensed under the MIT License.
🙌 Contributing
Feel free to open issues or submit pull requests if you have suggestions or improvements!

Built with ❤️ by Hoang Tuan
