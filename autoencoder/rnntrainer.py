import os
import argparse
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from autoencoder.model import RNNModel
from torch.utils.data import TensorDataset, DataLoader

def make_timetables(data, size1, size2):
    dlen = data.shape[0]
    size = size1 * size2
    maxsize = np.floor_divide(dlen - size, batch_size) * batch_size
    result = torch.zeros((maxsize, size1, size2))
    for i in range(0, maxsize):
        result[i] = data[i:i + size].view(size1, size2)

    return result

def check_anomaly(inputs, outputs, verbose=False):
    criterion = nn.MSELoss()
    for i in range(batch_size):
        loss = criterion(inputs[i], outputs[i])
        if loss > anomaly_threshold:
            if verbose:
                print('Anomaly: %f' % loss.item())
            else:
                print('Anomaly found')
                break

def read_file(filename, scaler, history_dim, input_dim, shuffle=True, fit=False):
    csv = pd.read_csv(filename, parse_dates=['timestamp'], dtype={'value':'float'})

    if fit:
        csv.loc[:, 'value'] = scaler.fit_transform(csv.loc[:, 'value'].values.reshape(-1, 1))
    else:
        csv.loc[:, 'value'] = scaler.transform(csv.loc[:, 'value'].values.reshape(-1, 1))
    x = torch.from_numpy(csv.loc[:, 'value'].values)
    x = make_timetables(x, history_dim, input_dim)

    data = TensorDataset(x, x)
    return DataLoader(data, shuffle=shuffle, batch_size=batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--history_dim', type=int, default=50)
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=30)
    parser.add_argument('--encoder_dim', type=int, default=10)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--clip', type=int, default=5)  # gradient clipping
    parser.add_argument('--learning_rate', type=float, default=0.0008)
    parser.add_argument('--anomaly_threshold', type=float, default=5.0)

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    parser.add_argument('--train_file', type=str)
    parser.add_argument('--test_file', type=str)

    args = parser.parse_args()

    batch_size = args.batch_size
    anomaly_threshold = args.anomaly_threshold

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    scaler = StandardScaler()
    train_loader = read_file(args.train_file, scaler, args.history_dim, args.input_dim, fit=True)
    test_loader = read_file(args.test_file, scaler, args.history_dim, args.input_dim,
                            shuffle=False, fit=False)

    model = RNNModel(args.input_dim, args.hidden_dim, args.encoder_dim, args.layers)
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_on_gpu=torch.cuda.is_available()

    if train_on_gpu:
        print('Training on GPU.')
        model = model.cuda()
    else:
        print('No GPU available, training on CPU.')

    for epoch in range(1, args.epochs+1):
        train_loss = 0.0
        test_loss = 0.0
        hidden1, hidden2 = model.init_hidden(batch_size, train_on_gpu)

        model.train()
        for inputs, _ in train_loader:
            if train_on_gpu:
                inputs = inputs.cuda()

            if len(inputs) != batch_size:
                break

            hidden1 = tuple([each.data for each in hidden1])
            hidden2 = tuple([each.data for each in hidden2])
            optimizer.zero_grad()
            outputs, hidden1, hidden2 = model(inputs, hidden1, hidden2)
            loss = criterion(outputs, inputs)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        model.eval()
        hidden1, hidden2 = model.init_hidden(batch_size, train_on_gpu)
        for inputs, _ in test_loader:
            if train_on_gpu:
                inputs = inputs.cuda()

            if len(inputs) != batch_size:
                break

            hidden1 = tuple([each.data for each in hidden1])
            hidden2 = tuple([each.data for each in hidden2])
            outputs, hidden1, hidden2 = model(inputs, hidden1, hidden2)
            loss = criterion(outputs, inputs)

            check_anomaly(inputs, outputs)

            test_loss += loss.item() * inputs.size(0)

        # print avg training statistics
        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch,
                                                                                   train_loss,
                                                                                   test_loss))
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir,
                                                    'checkpoint-%d.pth' % epoch))
