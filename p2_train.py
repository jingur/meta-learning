import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.autograd as autograd
from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from generator import Generator
from utils import pprint, ensure_path, Averager, Timer, count_acc, euclidean_metric
from torch.distributions import uniform, normal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=50)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/proto-1_p2_50')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--M', default='50')
    args = parser.parse_args()
    pprint(vars(args))

    ensure_path(args.save_path)

    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    model = Convnet().cuda()
    net_G = Generator(1728).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optim_G = torch.optim.Adam(net_G.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
    def save_model_G(name):
        torch.save(net_G.state_dict(), osp.join(args.save_path, name + '_G.pth'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()
        net_G.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            #proto plus noise input to a generator
            for k in range(0,int(args.M)):
                Z = normal.Normal(0, 1).sample([p, 128]).cuda()
                new_X = torch.cat([Z, proto], dim=1).float()
                X_gen = net_G(new_X)
                proto = proto*(k+1) + X_gen
                proto = proto/(k+2)

            #calculate mean of th output of the generator
            #proto = torch.cat([X_gen, proto], dim=1).float()
            #proto = torch.mean(torch.stack([proto, X_gen(k)] for k in range(0, int(args.M))), 0)
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            L_gen = loss
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            #bp for conv4
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()

            #bp for generator
            optim_G.zero_grad()
            L_gen.backward()
            optim_G.step() 

            proto = None; logits = None; loss = None

        tl = tl.item()
        ta = ta.item()

        model.eval()
        net_G.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            for k in range(0,int(args.M)):
                Z = normal.Normal(0, 1).sample([p, 128]).cuda()
                new_X = torch.cat([Z, proto], dim=1).float()
                X_gen = net_G(new_X)
                proto = proto*(k+1) + X_gen
                proto = proto/(k+2)
                
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)
            
            proto = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')
            save_model_G('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog_p2'))


        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

