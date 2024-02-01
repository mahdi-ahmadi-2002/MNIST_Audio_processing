import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import scipy.io.wavfile as wav
import os
from sklearn.metrics import accuracy_score
import time
import torchaudio
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
from torch.utils.data import Dataset
import scipy.io.wavfile as wav
import librosa
from torch.nn.functional import pad
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


class VoiceMNISTTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads, num_layers):
        super(VoiceMNISTTransformer, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          nhead=num_heads,
                                          batch_first=True).encoder
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.bn = nn.LazyBatchNorm1d()

    def forward(self, x):
        x = self.bn(self.embedding(x)).relu()
        output = self.transformer(x)
        output = self.fc(output)
        return output


def load_model():
    input_dim = 16384
    hidden_dim = 512
    num_classes = 10
    num_heads = 8
    num_layers = 4
    num_epochs = 1
    model = VoiceMNISTTransformer(input_dim, hidden_dim, num_classes, num_heads, num_layers)
    model.load_state_dict(torch.load('path/to/saved/model.pth'))
    model.eval()
    return model


def preprocess_audio(audio_path):
    data = []
    temp = librosa.load(audio_path)
    temp = temp[0]
    data.append(temp)

    paddedDataa = pad_sequences(data, padding='post', dtype='float32')

    ToSpectrogram = torchaudio.transforms.MelSpectrogram()
    ToDB = torchaudio.transforms.AmplitudeToDB()
    paddedDataa = pad_sequences(paddedDataa, padding='post', dtype='float32')
    file = paddedDataa
    audio_padded = torch.zeros((1, 25500))
    audio_padded[0, :len(file)] = torch.Tensor(file)
    padded = audio_padded

    file = padded
    spectrogram = ToSpectrogram(file)
    spectrogram = ToDB(spectrogram)
    audio_seq = spectrogram[0]

    normalized_data = (audio_seq - (-100)) / (20 - (-100))
    normalized_data = torch.FloatTensor(normalized_data)


    test_loader = DataLoader(normalized_data.flatten(), batch_size=32)

    return test_loader

# Prediction function
def predict(audio_path):
    processed_audio = preprocess_audio(audio_path)

    model = load_model()
    with torch.no_grad():
        output = model(processed_audio)

    predicted_class = torch.argmax(output).item()
    return predicted_class

# Define the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs='audio',
    outputs='text'  # Assuming classification output as text
)

# Launch the Gradio app
iface.launch()