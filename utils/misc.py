import copy
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from skimage import io
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_datetime():
  dt_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
  dt_now = dt_now.strftime('%Y%m%d-%H%M%S')
  return dt_now

def get_unnormalize():
    unmean = [-x for x in (0.485, 0.456, 0.406)]
    unstd = [1.0/x for x in (0.229, 0.224, 0.225)]
    unnormalize = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                           std = unstd),
                                      transforms.Normalize(mean = unmean,
                                                           std = [ 1., 1., 1. ]),])
    return unnormalize

def visualize_batch(images, ncols=4):
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(3*ncols, 3))
    for n in range(ncols):
        img = images[n].cpu().numpy()
        img = img.transpose((1, 2, 0))
        ax[n].imshow(img)
        ax[n].axis('off')

def set_requires_grad_toFalse(model):
    for param in model.parameters():
        param.requires_grad = False

def plot_learning_trajectory(losses, val_losses, ylabel = 'loss'):
    n_epochs = losses.shape[0]
    plt.figure()
    plt.plot(np.arange(n_epochs), np.array(losses), label='training')
    plt.plot(np.arange(n_epochs), np.array(val_losses), label='validation')
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

def plot_learning_trajectory_semilogy(losses, val_losses, ylabel = 'loss'):
    n_epochs = losses.shape[0]
    plt.figure()
    plt.semilogy(np.arange(n_epochs), np.array(losses), label='training')
    plt.semilogy(np.arange(n_epochs), np.array(val_losses), label='validation')
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

def save_model_results(results_path, dt_now, 
                       model, train_losses, train_accs, val_losses, val_accs, test_loss, test_acc):
  ### save model, [train/val] [losses/accs]
  torch.save(model.to('cpu').state_dict(), results_path + 'ResNet50_' + dt_now + '_' + str(np.argmax(val_accs)) + 'epoch.pt')
  np.savetxt(results_path + 'train_losses.csv', train_losses, delimiter=',')
  np.savetxt(results_path + 'train_accs.csv', train_accs, delimiter=',')
  np.savetxt(results_path + 'val_losses.csv', val_losses, delimiter=',')
  np.savetxt(results_path + 'val_accs.csv', val_accs, delimiter=',')

  ### save test loss/acc of best weights
  np.savetxt(results_path + 'test_loss.csv', np.array([test_loss]), delimiter=',')
  np.savetxt(results_path + 'test_acc.csv', np.array([test_acc]), delimiter=',')

def save_data_indices(results_path, train_indices, val_indices, test_indices):
  ### save data indices
  np.savetxt(results_path + 'indices_train.csv', train_indices, delimiter=',')
  np.savetxt(results_path + 'indices_val.csv', val_indices, delimiter=',')
  np.savetxt(results_path + 'indices_test.csv', test_indices, delimiter=',')
    

def display_loss_acc(path):
  fig, ax = plt.subplots(1,2, figsize=(12,6))
  ax[0].imshow(io.imread(path+ 'learning_trajectory_acc.png'))
  ax[0].axis('off')
  ax[1].imshow(io.imread(path+ 'learning_trajectory_loss.png'))
  ax[1].axis('off')

def predict_testloader(model, test_loader, dev):
    model.eval()
    inputs_test, labels_test = [] , []
    outputs_test = []
    preds_test = []
    for inputs, labels in test_loader:
        inputs_test.append(inputs)
        inputs = inputs.to(dev)

        labels_test.append(labels)
        labels = labels.to(dev)

        with torch.no_grad():
            outputs = model(inputs)
        outputs_test.append(outputs.detach().cpu().numpy())
        _, preds = torch.max(outputs, 1)
        preds_test.append(preds.detach().cpu().numpy())

    inputs_test = np.vstack(inputs_test)
    labels_test = np.hstack(labels_test)
    outputs_test = np.vstack(outputs_test)
    preds_test = np.hstack(preds_test)
    return inputs_test, labels_test, outputs_test, preds_test



def get_loss_func(outputs, labels, criterion, multilabel=False):
    if multilabel:
        return criterion(outputs, F.one_hot(labels, num_classes=2).to(torch.float32))
    elif not multilabel:
        return criterion(outputs[:, 1] - outputs[:, 0], labels.to(torch.float32))


def train_model_pytorch_tutorial(model, dataloaders, dev, criterion, optimizer, num_epochs=5, multilabel=False):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses = np.zeros((num_epochs))
    train_accs = np.zeros((num_epochs))
    val_losses = np.zeros((num_epochs))
    val_accs = np.zeros((num_epochs))
    
    for epoch in range(num_epochs):
        if epoch%10 == 0:
            print(f'Epoch {epoch}/{num_epochs-1}')
            print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(dev)
                labels = labels.to(dev)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) #labelsと同じ形

                    loss = get_loss_func(outputs, labels, criterion, multilabel)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # add batch loss & corrects
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # epoch
            datanum = len(dataloaders["train"].dataset) if phase == 'train' else len(dataloaders['val'].dataset)
            epoch_loss = running_loss / datanum
            epoch_acc = running_corrects.double() / datanum
            if phase == 'train':
                train_losses[epoch] = epoch_loss
                train_accs[epoch] = epoch_acc
            else:
                val_losses[epoch] = epoch_loss
                val_accs[epoch] = epoch_acc
            
            if epoch%10 == 0: print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        if epoch%10 == 0: print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m')
    print(f'Best val Acc: {best_acc:4f}')
    
    model.load_state_dict(best_model_wts)
    return model, train_losses, train_accs, val_losses, val_accs

def compute_loss(model, loader, dev, multilabel=False):
    running_loss = 0.0
    running_corrects = 0.0
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    for inputs, labels in loader:
        inputs = inputs.to(dev)
        labels = labels.to(dev)
        with torch.no_grad():
            outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = get_loss_func(outputs, labels, criterion, multilabel)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    datanum = len(loader.dataset)
    total_loss = running_loss / datanum
    total_acc = running_corrects.detach().cpu().numpy() / datanum
    return total_loss, total_acc

def test_loadedModel(model_path, model, val_loader, dev, multilabel=False):
    ## saved best val acc & loss when training
    val_accs = pd.read_csv(model_path + 'val_accs.csv', header=None)
    val_acc = np.max(val_accs)
    val_losses = pd.read_csv(model_path + 'val_losses.csv', header=None)
    val_loss = val_losses.values[np.argmax(val_accs)]
    print('epoch with max validation acc. : ' + str(np.argmax(val_accs)))

    ## calculate val acc & loss with newly loaded model
    loss, acc = compute_loss(model, val_loader, dev, multilabel)
    print(f'loss : {loss}, acc : {acc}')

    assert int((val_acc - acc)*100)==0, 'The accuracy mismatch'
    assert int((val_loss - loss)*100)==0, 'The loss mismatch'
