import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from convnet import Convnet
import csv
import random
import numpy as np
import pandas as pd
from utils import count_acc, Averager, euclidean_metric
from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def predict(args, model, data_loader):
    ave_acc = Averager()
    
    with torch.no_grad():
        prediction_results = pd.DataFrame()
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):
            
            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot,:,:,:] 
            query_input   = data[args.N_way * args.N_shot:,:,:,:]

            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])

            # TODO: extract the feature of support and query data
            x = model(support_input.cuda()) 
            y = model(query_input.cuda())

            # TODO: calculate the prototype for each class according to its support data
            x = x.reshape(args.N_shot, args.N_way, -1).mean(dim=0)

            # TODO: classify the query data depending on the its distense with each prototype
            logits = euclidean_metric(y, x)
            prediction = torch.argmax(logits, dim=1)
            pred_per_episode = pd.DataFrame(prediction).astype("int").T
            prediction_results = prediction_results.append(pred_per_episode,ignore_index=True)
            acc = count_acc(logits, query_label)
            ave_acc.add(acc)
            ave_acc.item()
            #print(acc)

    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str,default = "./save/proto-1/max-acc.pth", help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    
    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # TODO: load your model
    model = Convnet().cuda()
    model.load_state_dict(torch.load(args.load))
    model.eval()

    prediction_results = predict(args, model, test_loader)
    # TODO: output your prediction to csv
    prediction_results.columns=['query0', 'query1', 'query2', 'query3', 'query4', 'query5', 'query6', 'query7', 'query8', 'query9', 'query10','query11', 'query12', 'query13', 'query14', 'query15', 'query16', 'query17', 'query18', 'query19', 'query20', 'query21', 'query22', 'query23', 'query24', 'query25','query26', 'query27', 'query28', 'query29', 'query30', 'query31', 'query32', 'query33', 'query34', 'query35', 'query36', 'query37', 'query38', 'query39', 'query40', 'query41', 'query42', 'query43', 'query44', 'query45', 'query46', 'query47', 'query48', 'query49', 'query50', 'query51', 'query52', 'query53', 'query54', 'query55', 'query56', 'query57', 'query58', 'query59', 'query60', 'query61', 'query62', 'query63', 'query64', 'query65', 'query66', 'query67', 'query68', 'query69', 'query70', 'query71', 'query72', 'query73', 'query74'] 
    index = prediction_results.index
    index.name = "episode_id"
    prediction_results.to_csv(args.output_csv)


