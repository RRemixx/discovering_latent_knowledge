import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import numpy as np
from tqdm import tqdm


class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)

class CCS(object):
    def __init__(self, x0, x1, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
                 verbose=False, device="cuda", linear=True, weight_decay=0.01, var_normalize=False):
        # data
        self.var_normalize = var_normalize
        self.x0 = self.normalize(x0)
        self.x1 = self.normalize(x1)
        self.d = self.x0.shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # probe
        self.linear = linear
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

        
    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            self.probe = MLPProbe(self.d)
        self.probe.to(self.device)    


    def normalize(self, x):
        """
        Mean-normalizes the data x (of shape (n, d))
        If self.var_normalize, also divides by the standard deviation
        """
        normalized_x = x - x.mean(axis=0, keepdims=True)
        if self.var_normalize:
            normalized_x /= normalized_x.std(axis=0, keepdims=True)

        return normalized_x

        
    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x0 = torch.tensor(self.x0, dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.x1, dtype=torch.float, requires_grad=False, device=self.device)
        return x0, x1
    

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (torch.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        return informative_loss + consistent_loss


    def get_acc(self, x0_test, x1_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        x0 = torch.tensor(self.normalize(x0_test), dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.normalize(x1_test), dtype=torch.float, requires_grad=False, device=self.device)
        with torch.no_grad():
            p0, p1 = self.best_probe(x0), self.best_probe(x1)
        avg_confidence = 0.5*(p0 + (1-p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        acc = (predictions == y_test).mean()
        acc = max(acc, 1 - acc)

        return acc
    
        
    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = self.get_tensor_data()
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]
        
        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            for j in range(nbatches):
                x0_batch = x0[j*batch_size:(j+1)*batch_size]
                x1_batch = x1[j*batch_size:(j+1)*batch_size]
            
                # probe
                p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

                # get the corresponding loss
                loss = self.get_loss(p0, p1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()
    
    def repeated_train(self):
        best_loss = np.inf
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss

        return best_loss
    
    def supervised_train(self, labels, batch_size=-1, verbose=False, num_data_points=None, nepochs=0):
        """
        Trains the probe using the provided labels (0/1)
        Args:
            num_data_points: Number of data points to use for training. If None, use all data points.
        Returns both final loss and loss history
        """
        x0, x1 = self.get_tensor_data()
        x = torch.cat([x0, x1], dim=0)
        y = torch.tensor(np.concatenate([labels, labels]), dtype=torch.float, requires_grad=False, device=self.device).reshape(-1, 1)
        
        if num_data_points is not None and num_data_points < len(x):
            # Randomly select num_data_points data points
            indices = torch.randperm(len(x))[:num_data_points]
            x = x[indices]
            y = y[indices]
        
        # Check x and y values
        # Print a few examples to check data
        print("First 5 x values:", x[:5])
        print("First 5 y values:", y[:5])
        print("Shape of x:", x.shape)
        print("Shape of y:", y.shape)
        print("Unique y values:", torch.unique(y))
        
        # check the distributio of y
        print("Distribution of y values:", torch.bincount(y.flatten().long()))
        
        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if batch_size == -1:
            batch_size = len(x)
        nbatches = len(x) // batch_size
        print(len(x), batch_size, nbatches)

        # Track loss history
        loss_history = []

        # Start training
        for epoch in tqdm(range(nepochs)):
            epoch_loss = 0.0
            # Randomly shuffle the data at each epoch
            perm = torch.randperm(len(x))
            x_shuffled = x[perm]
            y_shuffled = y[perm]
            
            for j in range(nbatches):
                x_batch = x_shuffled[j*batch_size:(j+1)*batch_size]
                y_batch = y_shuffled[j*batch_size:(j+1)*batch_size]

                # probe
                p = self.probe(x_batch)
                
                # check some of the model outputs in a batch
                # print("Model outputs (probabilities):", p[:5])
                # print("Corresponding labels:", y_batch[:5])

                # get the corresponding loss
                loss = F.binary_cross_entropy(p, y_batch)
                epoch_loss += loss.item()

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Record average loss for this epoch
            loss_history.append(epoch_loss / nbatches)
            # Print loss for this epoch
            if verbose:
                print(f"Epoch {epoch+1}/{self.nepochs}, Loss: {epoch_loss / nbatches:.4f}")

        # visualize the loss_history
        # if verbose:
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss History")
        plt.show()