from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch import nn
import copy
import csp_densenet
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import pandas as pd
import random

from sklearn.metrics import classification_report

def seed_torch(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True





# normalize input for faster computations
transform = transforms.Compose([
                                transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def train(model, device, train_loader, optimizer, scheduler, criterion):
    running_loss = 0.0
    avg_loss = 0
    correct = 0
    total_imges = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, _ = data
        total_imges += len(labels)
#         print(i, labels, 'datame')
        inputs=inputs.to(device)
        labels=labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
#         print(outputs, labels)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        avg_loss += loss.item()
        
    correct = correct/total_imges
    print('Finished Training with an accuracy of ', correct*100, '%')
    print ('avg loss', avg_loss/(i+1))
    return avg_loss/(i+1), correct*100

def evaluate(model_eval, device, test_loader, criterion, nb_classes):
    correct = 0
    total = 0
    running_loss = 0
    avg_loss = 0
    model_eval.eval()
    confusion_matrix = np.zeros((nb_classes, nb_classes))
    incorrect_identified = []
    correct_identified = []
    report_actual = []
    report_predicted = []
    probabilities = []
    outputs_ = []
    with torch.no_grad():
        for i,data in enumerate(test_loader):
            images, labels, path = data
            report_actual += list(labels.tolist())
            ima = copy.deepcopy(images)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_eval(images)
            outputs_ += [outputs]
            prob, predicted = torch.max(outputs.data, 1)
            report_predicted += list(predicted.view(-1).tolist())
            probabilities += prob.view(-1).tolist()
            total += len(labels)
            correct += (predicted == labels).sum().item()
                    
            for t, p, im_,pa_ in zip(labels.view(-1), predicted.view(-1), ima, path):
                confusion_matrix[t.long(), p.long()] += 1
                if t.long()==p.long():
                    correct_identified.append((t.long(), p.long(), im_, pa_))
                    continue
                incorrect_identified.append((t.long(), p.long(), im_, pa_))
                
            loss = criterion(outputs, labels)
              # print statistics
            running_loss += loss.item()
            avg_loss += loss.item()      
    if nb_classes==3:
        target_names = ['0','1','2']
    if nb_classes==2:
        target_names = ['0','1']
    report = classification_report(report_actual, report_predicted, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report)
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    print ('avg loss', avg_loss/(i+1))
    return {'avg_loss':avg_loss/(i+1), 'accuracy':(correct*100) / total, 'confusion_matrix':confusion_matrix, 'incorrect_identified':incorrect_identified, 'correct_identified':correct_identified, 'report':report, 'report_df':report_df, 'probabilities':probabilities, 'report_actual':report_actual, 'report_predicted':report_predicted, 'outputs_':outputs_}


def training(train_dataset_path, val_dataset_path, batch_size, epochs, classification, model, device, optimizer, scheduler, criterion, seed, step=''):
    seed_torch(seed)
    train_dataset = ImageFolderWithPaths(train_dataset_path, transform=transform)
    val_dataset = ImageFolderWithPaths(val_dataset_path, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    if classification=='binary':
        num_classes = 2
    elif classification=='multi':
        num_classes = 3
    else:
        raise Exception("invalid classification")

    train_losses=[]
    train_accuracies=[]
    val_losses=[]
    val_accuracies=[]
    lr_data=[]
    save_model_paths = []

    weights_sub_folder = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    model_save_path = os.path.join(os.getcwd(), 'weights', classification, weights_sub_folder)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    res_path = os.path.join(os.getcwd(), 'results', classification, weights_sub_folder)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    print(model_save_path, "model_save_path")
#     EPOCHS = 100
    
    cols = ['epoch', 'train_accuracies', 'train_losses', 'val_accuracies', 'val_losses', 'save_model_paths']
    with open(os.path.join(res_path, 'res_temp.txt'), 'w') as f_w:
        f_w.write(','.join(map(str,cols))+'\n')
    for epoch in range(epochs):
        print("EPOCH:", epoch, 'started at:', datetime.datetime.now())
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, scheduler, criterion)
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)

        val_result = evaluate(copy.deepcopy(model), device, val_loader, criterion, num_classes)
        val_loss, val_accuracy, matrix, incorrect = val_result['avg_loss'], val_result['accuracy'], val_result['confusion_matrix'], val_result['incorrect_identified']
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)
        save_model_path = os.path.join(model_save_path, str(epoch)+'.pt')
        save_model_paths.append(save_model_path)
        torch.save(model, save_model_path)
        if step=='val_loss':
            scheduler.step(val_loss)
        else:
            scheduler.step()
    #     
        print(optimizer.state_dict()['param_groups'][0]['lr'], 'lr')
        data_ = [epoch, train_accuracy, train_loss, val_accuracy, val_loss, save_model_path]
        with open(os.path.join(res_path, 'res_temp.txt'), 'a') as f_w:
            f_w.write(','.join(map(str,data_))+'\n')

    
        
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    print(res_path, "path")
    df = pd.DataFrame({'epoch':list(range(len(train_accuracies))), 'train_accuracies':train_accuracies, 'val_accuracies':val_accuracies , 'train_losses':train_losses , 'val_losses':val_losses, 'save_model_paths':save_model_paths})
    df.to_csv(os.path.join(res_path, 'results.csv'), index=False)
    results_df = df.copy()
    model_info = os.path.join(res_path, 'model_info.txt')
    with open(model_info, 'w', encoding='utf-8') as f_w:
        f_w.write('optimizer is : ' + str(optimizer) + '\n')
        f_w.write('scheduler is : ' + str(scheduler) + '\n')
        f_w.write('criterion is : ' + str(criterion) + '\n')    
        f_w.write('model  is : ' + str(model) + '\n')

    df = df.loc[df['train_accuracies']>df['val_accuracies']]
    df = df.loc[df['train_accuracies']-df['val_accuracies'] < 7]
    df = df.loc[(df['train_accuracies']>90) & (df['val_accuracies']>87) ]
    df.sort_values(['train_accuracies', 'val_accuracies',], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    if len(df)>0:
        best_model_path = df['save_model_paths'][0]
    else:
        results_df.sort_values(['train_accuracies', 'val_accuracies',], ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)
        best_model_path = results_df['save_model_paths'][0]
        print("model did not provide satisfactory results such as train_accuracies>94, val_accuracies>92,  train_accuracies>val_accuracies and difference between train and test accuracies is less than 4")
    return {'model_save_path':model_save_path, 'results_df':results_df, 'best_weights_df':df, 'train_accuracies':train_accuracies, 'train_losses':train_losses, 'val_accuracies':val_accuracies, 'val_losses':val_losses, 'model_info_path':model_info, 'best_model_path':best_model_path}

def testing(test_dataset_path, batch_size, model_path, device, criterion, nb_classes):
    test_dataset = ImageFolderWithPaths(test_dataset_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    bestmodel = torch.load(model_path)
    result = evaluate(copy.deepcopy(bestmodel), device, test_dataloader, criterion, nb_classes)
    return result

def inference(img_path):
    pass