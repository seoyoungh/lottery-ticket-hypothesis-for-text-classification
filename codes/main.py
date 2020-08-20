# Importing Libraries
import os
import copy
import time
import torch
import torchtext
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

# Custom Libraries
import DataLoader
import Model

# Define functions
# load dataset
def load_dataset(dataset, architecture, batch_size, device, path):
    if dataset == "imdb":
      if architecture == "cnn":
        data = DataLoader.IMDB_CNN_CUSTOM(batch_size, device, path)
      elif architecture == "lstm":
        data = DataLoader.IMDB_LSTM(batch_size, device, path)

    elif dataset == "agnews":
        data = DataLoader.AGNEWS(batch_size, device, path)

    else:
        raise ValueError(dataset + "is not supported")

    return data

# load model and set hyperparameters
def load_model(architecture, data_choice, batch_size):

    if architecture == "cnn":
      # hyperparameters
      vocab_size = len(dataset.TEXT.vocab)
      embedding_dim = 100
      n_filters = 100
      filter_sizes = [3,4,5]
      dropout = 0.5
      pad_idx = dataset.TEXT.vocab.stoi[dataset.TEXT.pad_token]

      if data_choice == "imdb":
        # binary-class
        output_dim = 1
        model = Model.binaryCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx)

      elif data_choice == "agnews":
        # multi-class
        output_dim = 4
        model = Model.multiCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx)

      unk_idx = dataset.TEXT.vocab.stoi[dataset.TEXT.unk_token]
      model.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
      model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)

      return model

    elif architecture == "lstm":
      # hyperparameters
      vocab_size = len(dataset.TEXT.vocab)
      embedding_dim = 100
      hidden_dim = 256
      output_dim = 1
      n_layers = 2
      bidirectional = True
      dropout = 0.5
      pad_idx = dataset.TEXT.vocab.stoi[dataset.TEXT.pad_token]

      if data_choice == "imdb":
        # binary-class
        output_dim = 1

      elif data_choice == "agnews":
        # multi-class
        output_dim = 4

      model = Model.LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx)

      unk_idx = dataset.TEXT.vocab.stoi[dataset.TEXT.unk_token]
      model.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
      model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)

      return model

# weight initialization
# weight initializtion
def initialize_xavier_normal(m):

  """
	Function to initialize a layer by picking weights from a xavier normal distribution
	Arguments
	---------
	m : The layer of the neural network
	Returns
	-------
	None
	"""

  if type(m) == torch.nn.Conv2d:
    torch.nn.init.xavier_normal_(m.weight.data)
    m.bias.data.fill_(0)

  elif isinstance(m, torch.nn.Linear):
    torch.nn.init.xavier_normal_(m.weight.data)
    m.bias.data.fill_(0)

  elif type(m) in [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]:
    for name, param in m.named_parameters():
        if 'weight_ih' in name:
          torch.nn.init.xavier_normal_(param.data)
        elif 'weight_hh' in name:
          torch.nn.init.orthogonal_(param.data)
        elif 'bias' in name:
          param.data.fill_(0)

# train and test functions
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum().float() / torch.FloatTensor([y.shape[0]]).to(device)

# binary
def train(model, iterator, optimizer, criterion):
    # EPS = 1e-6

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

      optimizer.zero_grad()
      text, text_lengths = batch.text
      target = batch.label
      predictions = model(text, text_lengths).squeeze(1)
      loss = criterion(predictions, target)
      acc = binary_accuracy(predictions, target)
      loss.backward()
      step = 0

      # Freezing Pruned weights by making their gradients Zero
      for name, p in model.named_parameters():
        weight_dev = param.device
        tensor = p.data.cpu().numpy()
        grad_tensor = p.grad.data.cpu().numpy()
        grad_tensor = np.where(mask[step] == 0, 0, grad_tensor)
        p.grad.data = torch.from_numpy(grad_tensor).to(device)
        step += 1

      step = 0

      optimizer.step()

      epoch_loss += loss.item()
      epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:
          text, text_lengths = batch.text
          target = batch.label

          predictions = model(text, text_lengths).squeeze(1)

          loss = criterion(predictions, batch.label)

          acc = binary_accuracy(predictions, batch.label)

          epoch_loss += loss.item()
          epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# multi
def train(model, iterator, optimizer, criterion):
    # EPS = 1e-6

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

      optimizer.zero_grad()

      predictions = model(batch.text[0])

      loss = criterion(predictions, batch.label)

      acc = categorical_accuracy(predictions, batch.label)

      loss.backward()
      step = 0

      # Freezing Pruned weights by making their gradients Zero
      for name, p in model.named_parameters():
        weight_dev = param.device
        tensor = p.data.cpu().numpy()
        grad_tensor = p.grad.data.cpu().numpy()
        grad_tensor = np.where(mask[step] == 0, 0, grad_tensor)
        p.grad.data = torch.from_numpy(grad_tensor).to(device)
        step += 1

      step = 0

      optimizer.step()

      epoch_loss += loss.item()
      epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:
            predictions = model(batch.text[0])
            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Pruning Functions
# Prune by Percentile module
def prune_by_percentile(percent):
  global step
  global mask
  global model
  # Calculate percentile value
  step = 0
  for name, param in model.named_parameters():
    # if 'weight' in name:
    tensor = param.data.cpu().numpy()
    nz_count = np.count_nonzero(tensor)

    # bias가 all pruned 되는 경우 발생
    if nz_count == 0:
      step += 1

    else:
      alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
      percentile_value = np.percentile(abs(alive), percent)

      # Convert Tensors to numpy and calculate
      weight_dev = param.device
      new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

      # Apply new weight and mask
      param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
      mask[step] = new_mask
      step += 1

  step = 0

 # Function to make an empty mask of the same size as the model
def make_mask(model):
  global step
  global mask

  step = 0
  for name, param in model.named_parameters():
    # if 'weight' in name:
    step += 1
  mask = [None]* step
  step = 0
  for name, param in model.named_parameters():
    # if 'weight' in name:
    tensor = param.data.cpu().numpy()
    mask[step] = np.ones_like(tensor)
    step += 1
  step = 0

  def original_initialization(mask_temp, initial_state_dict):
    global step
    global model

    step = 0
    for name, param in model.named_parameters():
        # if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        # if "bias" in name:
            # param.data = initial_state_dict[name]
    step = 0

# ANCHOR Print table of zeros and non-zeros count
def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    return (round(((total-nonzero)/total)*100,1))

reinit = True # random이면 True, lt면 False
mode = 'random' # random / lt / lr_l

# load model and set hyperparameters
model = load_model(arch_choice, data_choice, batch_size)

# model initialization and save the state
model.apply(initialize_xavier_normal)

# initial state 저장
torch.save(model.state_dict(), f'{data_choice}-{arch_choice}-{trial}-{mode}-initial.pt')

model = model.to(device)
make_mask(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# binary
criterion = torch.nn.BCEWithLogitsLoss()
# multi
criterion = torch.nn.CrossEntropyLoss()

criterion = criterion.to(device)

print("Iterative Pruning started")

performance = [] # 결과값을 담을 리스트
valid_losses = []

for pruning_iter in range(0,iteration+1):
  print(f"Running pruning iteration {pruning_iter}/{iteration}")

  # 첫 iter에는 no model compression
  if not pruning_iter == 0:

    # pruning
    prune_by_percentile(percent)

    # random initialization
    if reinit:
      model.apply(initialize_xavier_normal) # random initialization
      model.embedding.weight = embedding_pretrained_weight # embedding은 공통적으로 초기 임베딩으로 초기화

      step = 0
      for name, param in model.named_parameters():
      # if 'weight' in name:
         weight_dev = param.device
         param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
         step = step + 1
      step = 0

    # lt initialization
    else:
      original_initialization(mask, initial_state_dict)

  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  # train
  for epoch in range(N_EPOCH):
    train_loss, train_acc = train(model, dataset.train_iter, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, dataset.valid_iter, criterion)
    print(epoch, valid_loss, valid_acc)

  test_loss, test_acc = evaluate(model, dataset.test_iter, criterion)
  torch.save(model.state_dict(), f'{data_choice}-{arch_choice}-{trial}-{mode}-{pruning_iter}.pt')
  print(test_acc)

  valid_losses.append(valid_loss)
  performance.append(100*test_acc)

  # late rewinding after the last epoch
  if pruning_iter == 0:
     initial_state_dict = copy.deepcopy(model.state_dict())

  print_nonzeros(model)

print(performance)
print(valid_losses)
