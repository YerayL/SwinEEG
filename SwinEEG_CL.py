import torch
from torch.autograd import Variable

# 对比学习一轮，预训练浅层的CNN编码器
# 所有被试数据平均
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def cl_loss(model, eeg_features, img_features):

    device =  eeg_features.device

    labels = torch.arange(eeg_features.shape[0])  # used for the loss
    labels = Variable(labels.cuda().type(torch.cuda.LongTensor)).to(device)

    # eeg_features = model.Proj_eeg(eeg_features)
    img_features = model.projector(img_features)

    eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
    img_features = img_features / img_features.norm(dim=1, keepdim=True)

    # cosine similarity as the logits
    logit_scale = model.logit_scale.exp()
    logits_per_eeg = logit_scale * eeg_features @ img_features.t()
    logits_per_img = logits_per_eeg.t()

    loss_eeg = model.criterion_cls(logits_per_eeg, labels)
    loss_img = model.criterion_cls(logits_per_img, labels)

    loss_cos = (loss_eeg + loss_img) / 2

    # total loss
    loss = loss_cos

    return loss

import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import warnings
import copy
import os
from torch.nn import TransformerEncoder
from transformers import ViTModel
from transformers import AutoImageProcessor
from torchvision.transforms import Resize, Compose, Normalize, ToTensor
from PIL import Image
from MMSwin2SR import MMSwin2SR
import torch.nn.functional as F
warnings.filterwarnings('ignore')
EXP_NAME = 'cl_pretrain_4000'
torch.manual_seed(1024)

# New parameters
recon_criterion1 = nn.CosineEmbeddingLoss(0.5)
recon_criterion2 = nn.MSELoss()
lambda_recon1 = 1
lambda_recon2 = 1

n_epochs = 4000
input_dim = 17
real_dim = 17
display_step = 1000
batch_size = 4096
lr = 0.0002  # 2e-4 , 2e-3
#target_shape = 100
device = 'cuda'
early_stop = True
patience = 10


def get_image_data():
    train_img_feature = np.load('/home/user/lanyinyu/EEG2EEG/dnn_feature/Vit_train_features.npy', allow_pickle=True)
    test_img_feature = np.load('/home/user/lanyinyu/EEG2EEG/dnn_feature/Vit_test_features.npy', allow_pickle=True)

    train_img_feature = np.squeeze(train_img_feature)
    test_img_feature = np.squeeze(test_img_feature)

    train_img_feature = torch.from_numpy(train_img_feature).float()
    test_img_feature = torch.from_numpy(test_img_feature).float()
    # print(train_img_feature.shape, test_img_feature.shape) # torch.Size([16540, 768]) torch.Size([200, 768])
    return train_img_feature, test_img_feature




def get_gen_loss(gen, real, condition, recon_criterion1, recon_criterion2, lambda_recon1, lambda_recon2, img_feature):
    condition = condition.unsqueeze(1)
    fake = gen(condition, img_feature)
    fake = fake.squeeze(1)
    gen_rec_loss1 = recon_criterion1(real.flatten(start_dim=1).to(device), fake.flatten(start_dim=1).to(device), torch.ones(fake.shape[0]).to(device))
    gen_rec_loss2 = recon_criterion2(real, fake)
    gen_loss = lambda_recon1 * gen_rec_loss1 + lambda_recon2 * gen_rec_loss2
    return gen_loss


class GetData(torch.utils.data.Dataset):
    
    def __init__(self, data1, data2, data3):
        self.data1 = data1   # source eeg
        self.data2 = data2   # target eeg   
        self.data3 = data3   # image 

    def __getitem__(self, index):
        data1 = self.data1[index]
        data2 = self.data2[index]
        data3 = self.data3[index]
        return data1, data2, data3
    
    def __len__(self):
        return len(self.data1)
import torch.nn.init as init



def train_and_test(torch_data, input_dim, real_dim, lr, testsub1data, testsub2data, subid1, subid2, test_img_feature, early_stop=False, patience=10):
    testsub1data = testsub1data.unsqueeze(1)
    f_path = f"/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/"
    if not os.path.exists(f_path):
        os.makedirs(f_path)

    upscale = 1
    window_size = 8

    height = (17 // upscale // window_size + 1) * window_size
    width = (200 // upscale // window_size + 1) * window_size

    gen = MMSwin2SR(upscale=1, img_size=(height, width), in_chans=1,
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pretrain').to(device)
    
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))

    gen = gen.apply(weights_init_normal)
    
    epochs_loss = np.zeros([n_epochs])
    
    dataloader = DataLoader(torch_data, batch_size=batch_size, shuffle=True)

    best_loss = 9999

    gen.train()
    
    for epoch in range(n_epochs):
        
        loss = 0
        # Dataloader returns the batches
        cur_step = 0
        for data1, data2, img_feature in tqdm(dataloader):
            condition = data1
            real = data2
            condition = condition.to(device)
            real = real.to(device)
            img_feature = img_feature.to(device)

            gen_opt.zero_grad()
            condition = condition.unsqueeze(1)
            eeg = gen(condition,img_feature)
            loss = cl_loss(gen, eeg, img_feature)

            loss.backward() # Update gradients
            gen_opt.step() # Update optimizer
            loss += loss.item()
            
            cur_step += 1
        

        with torch.no_grad():   # 节约内存加上
            testsub1data = testsub1data.to(device)
            test_img_feature = test_img_feature.to(device)
            testeeg = gen(testsub1data, test_img_feature)

        tloss = cl_loss(gen, testeeg, test_img_feature)

        # Keep track of the average generator loss
        mean_loss = loss / cur_step
        
        print('Sub' + str(subid1+1).zfill(2) + ' -> ' + 'Sub' + str(subid2+1).zfill(2) + ': ' + f"Epoch {epoch+1}: cl loss: {mean_loss} Test cl loss {tloss}" )
        #show(condition, fake, real)
        loss = 0
        epochs_loss[epoch] = mean_loss

        if tloss < best_loss:
            best_loss = tloss
            best_gen = copy.deepcopy(gen.state_dict())
            best_gen_opt = copy.deepcopy(gen_opt.state_dict())

        torch.cuda.empty_cache()
    
    torch.save({'gen':  gen.state_dict(),
                'gen_opt': gen_opt.state_dict()
                }, f"/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/final_model_{tloss}.pth")
    torch.save({'gen':  best_gen,
                'gen_opt': best_gen_opt
                }, f"/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/best_model_{best_loss}.pth")
    np.save(f'/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/gloss.npy', epochs_loss)


source_train_data = []
target_train_data = []
source_test_data = []
target_test_data = []
train_img_feature, test_img_feature = get_image_data()


for subid1 in range(10):
    for subid2 in range(10):
        if subid1 != subid2 and subid1 != 8 and subid2 != 8:
            data1 = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/train/sub' + str(subid1+1).zfill(2) + '.npy')
            mean1 = np.average(data1)
            std1 = np.std(data1)
            data1 = (data1-mean1)/std1
            data1 = np.transpose(data1, (1, 0, 2))
            data1 = torch.from_numpy(data1).float()
            data2 = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/train/sub' + str(subid2+1).zfill(2) + '.npy')
            mean2 = np.average(data2)
            std2 = np.std(data2)
            data2 = (data2-mean2)/std2
            data2 = np.transpose(data2, (1, 0, 2))
            data2 = torch.from_numpy(data2).float()
            source_train_data.append(data1)
            target_train_data.append(data2)
            testsub1data = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/test/sub' + str(subid1+1).zfill(2) + '.npy')
            testsub1data = (testsub1data-mean1)/std1
            testsub1data = np.transpose(testsub1data, (1, 0, 2))
            testsub1data = torch.from_numpy(testsub1data).float()
            source_test_data.append(testsub1data)
            testsub2data = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/test/sub' + str(subid2+1).zfill(2) + '.npy')
            testsub2data = (testsub2data-mean2)/std2
            testsub2data = np.transpose(testsub2data, (1, 0, 2))
            testsub2data = torch.from_numpy(testsub2data).float()
            target_test_data.append(testsub2data)



# 预训练所有数据~~~
source_train_data = torch.from_numpy(np.mean(source_train_data, axis= 0)).float()
target_train_data = torch.from_numpy(np.mean(target_train_data, axis= 0)).float()
source_test_data = torch.from_numpy(np.mean(source_test_data, axis= 0)).float()
target_test_data = torch.from_numpy(np.mean(target_test_data, axis= 0)).float()
print(source_train_data.shape, target_train_data.shape, source_test_data.shape, target_test_data.shape)
torch_data = GetData(source_train_data, target_train_data, train_img_feature)

train_and_test(torch_data, input_dim, real_dim, lr, source_test_data, target_test_data, subid1, subid2, test_img_feature, early_stop, patience)

            # torch_data = GetData(data1, data2, train_img_feature)
            # testsub1data = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/test/sub' + str(subid1+1).zfill(2) + '.npy')
            # testsub1data = (testsub1data-mean1)/std1
            # testsub1data = np.transpose(testsub1data, (1, 0, 2))
            # testsub1data = torch.from_numpy(testsub1data).float()
            # testsub2data = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/test/sub' + str(subid2+1).zfill(2) + '.npy')
            # testsub2data = (testsub2data-mean2)/std2
            # testsub2data = np.transpose(testsub2data, (1, 0, 2))
            # testsub2data = torch.from_numpy(testsub2data).float()
            
            # train_and_test(torch_data, input_dim, real_dim, lr, testsub1data, testsub2data, subid1, subid2, test_img_feature, early_stop, patience)

# for subid1 in range(10):
#     for subid2 in range(10):
#         if subid1 != subid2 and subid1 != 8 and subid2 != 8:
#             # data1 = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/train/sub' + str(subid1+1).zfill(2) + '.npy')
#             # mean1 = np.average(data1)
#             # std1 = np.std(data1)
#             # data1 = (data1-mean1)/std1
#             # data1 = np.transpose(data1, (1, 0, 2))
#             # data1 = torch.from_numpy(data1).float()
#             # data2 = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/train/sub' + str(subid2+1).zfill(2) + '.npy')
#             # mean2 = np.average(data2)
#             # std2 = np.std(data2)
#             # data2 = (data2-mean2)/std2
#             # data2 = np.transpose(data2, (1, 0, 2))
#             # data2 = torch.from_numpy(data2).float()
#             # torch_data = GetData(data1, data2)
#             testsub1data = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/test/st_sub' + str(subid1+1).zfill(2) + '.npy')
#             testsub1data = (testsub1data-mean1)/std1
#             testsub1data = np.transpose(testsub1data, (1, 2, 0, 3))
#             gen = MMSwin2SR(input_dim, real_dim).to(device)
#             gen.load_state_dict(torch.load(f"/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/best_model.pth")['gen'])
#             gen.eval()
#             testsub2fakedata = np.zeros([200, 80, 17, 200])
#             for i in range(200):
#                 test = torch.from_numpy(testsub1data[i]).float()
#                 test = test.to(device)
#                 testsub2fakedata[i] = gen(test).detach().cpu().numpy()
#             np.save(f"/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/generated_fullmodel/st_Sub{subid1+1}ToSub{subid2+1}.npy", testsub2fakedata)

# for subid1 in range(10):
#     for subid2 in range(10):
#         if subid1 != subid2 and subid1 != 8 and subid2 != 8:
#             # data1 = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/train/sub' + str(subid1+1).zfill(2) + '.npy')
#             # mean1 = np.average(data1)
#             # std1 = np.std(data1)
#             # data1 = (data1-mean1)/std1
#             # data1 = np.transpose(data1, (1, 0, 2))
#             # data1 = torch.from_numpy(data1).float()
#             # data2 = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/train/sub' + str(subid2+1).zfill(2) + '.npy')
#             # mean2 = np.average(data2)
#             # std2 = np.std(data2)
#             # data2 = (data2-mean2)/std2
#             # data2 = np.transpose(data2, (1, 0, 2))
#             # data2 = torch.from_numpy(data2).float()
#             # torch_data = GetData(data1, data2)
#             testsub1data = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/test/sub' + str(subid1+1).zfill(2) + '.npy')
#             testsub1data = (testsub1data-mean1)/std1
#             testsub1data = np.transpose(testsub1data, (1, 0, 2))
#             gen = MMSwin2SR(input_dim, real_dim).to(device)
#             gen.load_state_dict(torch.load(f"/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/best_model.pth")['gen'])
#             gen.eval()
#             testsub1data = torch.from_numpy(testsub1data).float()
#             testsub1data = testsub1data.to(device)
#             testsub2fakedata = gen(testsub1data).detach().cpu().numpy()
#             np.save(f"/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/generated_fullmodel/Sub{subid1+1}ToSub{subid2+1}.npy", testsub2fakedata)
