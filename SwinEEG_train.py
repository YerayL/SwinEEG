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
EXP_NAME = 'swin2sr_cl'
torch.manual_seed(1024)

# New parameters
recon_criterion1 = nn.CosineEmbeddingLoss(0.5)
recon_criterion2 = nn.MSELoss()
lambda_recon1 = 1
lambda_recon2 = 1

n_epochs = 100
input_dim = 17
real_dim = 17
display_step = 1000
batch_size = 48
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
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect').to(device)
    
    # cl pretrain model weight
    weights_path = '/home/user/lanyinyu/EEG2EEG/Exp/cl_pretrain_4000/Sub10ToSub10_AllChannels/best_model_5.24.pth'
    checkpoint = torch.load(weights_path)
    gen.load_state_dict(checkpoint['gen'])

    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

    # gen = gen.apply(weights_init)
    
    epochs_loss = np.zeros([n_epochs])
    epochs_corr = np.zeros([n_epochs])
    
    dataloader = DataLoader(torch_data, batch_size=batch_size, shuffle=True)

    patience_cnt = 0
    best_corr = 0

    gen.train()
    
    for epoch in range(n_epochs):
        if early_stop and patience_cnt >= patience:  
            print(f"epoch {epoch+1} early stop, best corr is {best_corr}")
            torch.save({'gen':  gen.state_dict(),
                'gen_opt': gen_opt.state_dict()
                }, f"/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/final_model.pth")
            torch.save({'gen':  best_gen,
                        'gen_opt': best_gen_opt
                        }, f"/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/best_model.pth")
            np.save(f'/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/gloss.npy', epochs_loss)
            np.save(f'/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/corr.npy', epochs_corr)
            break
        
        loss = 0
        # Dataloader returns the batches
        cur_step = 0
        for data1, data2, img_feature in tqdm(dataloader):
            condition = data1
            real = data2
            cur_batch_size = len(condition)
            condition = condition.to(device)
            real = real.to(device)
            img_feature = img_feature.to(device)

            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, real, condition, recon_criterion1, recon_criterion2, lambda_recon1, lambda_recon2, img_feature)
            gen_loss.backward() # Update gradients
            gen_opt.step() # Update optimizer
            loss += gen_loss.item()
            
            cur_step += 1
        
        gen.eval()
        gen_opt.zero_grad()
        
        with torch.no_grad():   # 节约内存加上
            testsub1data = testsub1data.to(device)
            testsub2data = testsub2data.to(device)
            test_img_feature = test_img_feature.to(device)
            testsub2fakedata = gen(testsub1data, test_img_feature)
            testsub2fakedata = testsub2fakedata.squeeze(1)

        arr1 = testsub2fakedata.detach().cpu().numpy()
        arr2 = testsub2data.detach().cpu().numpy()
        
        corr = spearmanr(arr1.flatten(), arr2.flatten())[0]
            
        # Keep track of the average generator loss
        mean_loss = loss / cur_step
        
        print('Sub' + str(subid1+1).zfill(2) + ' -> ' + 'Sub' + str(subid2+1).zfill(2) + ': ' + f"Epoch {epoch+1}: EEG2EEG loss: {mean_loss}, Corr : {corr}")
        #show(condition, fake, real)
        loss = 0
        epochs_loss[epoch] = mean_loss
        epochs_corr[epoch] = corr
        if corr > best_corr:
            best_corr = corr
            best_gen = copy.deepcopy(gen.state_dict())
            best_gen_opt = copy.deepcopy(gen_opt.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            # torch.save({'gen':  gen.state_dict(),
            #     'gen_opt': gen_opt.state_dict()
            #     }, f"/home/user/lanyinyu/EEG2EEG/Exp/MMTransformer/Sub{subid1+1}ToSub{subid2+1}_AllChannels/best_model.pth")
        torch.cuda.empty_cache()
    
    torch.save({'gen':  gen.state_dict(),
                'gen_opt': gen_opt.state_dict()
                }, f"/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/final_model.pth")
    
    torch.save({'gen':  best_gen,
                'gen_opt': best_gen_opt
                }, f"/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/best_model.pth")

    np.save(f'/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/gloss.npy', epochs_loss)
    np.save(f'/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/corr.npy', epochs_corr)


train_img_feature, test_img_feature = get_image_data()
train_img_feature = train_img_feature.to(device)
test_img_feature = test_img_feature.to(device)

# for subid1 in range(10):
#     for subid2 in range(10):
#         if subid1 != subid2 and subid1 != 8 and subid2 != 8:
#             data1 = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/train/sub' + str(subid1+1).zfill(2) + '.npy')
#             mean1 = np.average(data1)
#             std1 = np.std(data1)
#             data1 = (data1-mean1)/std1
#             data1 = np.transpose(data1, (1, 0, 2))
#             data1 = torch.from_numpy(data1).float()
#             data2 = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/train/sub' + str(subid2+1).zfill(2) + '.npy')
#             mean2 = np.average(data2)
#             std2 = np.std(data2)
#             data2 = (data2-mean2)/std2
#             data2 = np.transpose(data2, (1, 0, 2))
#             data2 = torch.from_numpy(data2).float()
#             torch_data = GetData(data1, data2, train_img_feature)
#             testsub1data = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/test/sub' + str(subid1+1).zfill(2) + '.npy')
#             testsub1data = (testsub1data-mean1)/std1
#             testsub1data = np.transpose(testsub1data, (1, 0, 2))
#             testsub1data = torch.from_numpy(testsub1data).float()
#             testsub2data = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/test/sub' + str(subid2+1).zfill(2) + '.npy')
#             testsub2data = (testsub2data-mean2)/std2
#             testsub2data = np.transpose(testsub2data, (1, 0, 2))
#             testsub2data = torch.from_numpy(testsub2data).float()
            
#             train_and_test(torch_data, input_dim, real_dim, lr, testsub1data, testsub2data, subid1, subid2, test_img_feature, early_stop, patience)

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
            # torch_data = GetData(data1, data2)
            testsub1data = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/test/st_sub' + str(subid1+1).zfill(2) + '.npy')
            testsub1data = (testsub1data-mean1)/std1
            testsub1data = np.transpose(testsub1data, (1, 2, 0, 3))
            upscale = 1
            window_size = 8
            height = (17 // upscale // window_size + 1) * window_size
            width = (200 // upscale // window_size + 1) * window_size
            gen = MMSwin2SR(upscale=1, img_size=(height, width), in_chans=1,
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect').to(device)
            gen.load_state_dict(torch.load(f"/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/best_model.pth")['gen'])
            gen.eval()
            testsub2fakedata = np.zeros([200, 80, 17, 200])

            with torch.no_grad(): 
                for i in range(200):
                    test = torch.from_numpy(testsub1data[i]).float()
                    test = test.to(device)
                    test = test.unsqueeze(1)
                    out = gen(test,test_img_feature[i])
                    out = out.squeeze(1)
                    testsub2fakedata[i] = out.detach().cpu().numpy()
                
            np.save(f"/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/generated_fullmodel/st_Sub{subid1+1}ToSub{subid2+1}.npy", testsub2fakedata)

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
            # torch_data = GetData(data1, data2)
            testsub1data = np.load('/home/user/lanyinyu/EEG2EEG/eeg_data/test/sub' + str(subid1+1).zfill(2) + '.npy')
            testsub1data = (testsub1data-mean1)/std1
            testsub1data = np.transpose(testsub1data, (1, 0, 2))
            upscale = 1
            window_size = 8
            height = (17 // upscale // window_size + 1) * window_size
            width = (200 // upscale // window_size + 1) * window_size
            gen = MMSwin2SR(upscale=1, img_size=(height, width), in_chans=1,
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect').to(device)
            gen.load_state_dict(torch.load(f"/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/Sub{subid1+1}ToSub{subid2+1}_AllChannels/best_model.pth")['gen'])
            gen.eval()
            with torch.no_grad(): 
                testsub1data = torch.from_numpy(testsub1data).float()
                testsub1data = testsub1data.to(device)
                testsub1data = testsub1data.unsqueeze(1)
                testsub2fakedata = gen(testsub1data,test_img_feature).detach().cpu().numpy()
                testsub2fakedata = testsub2fakedata.squeeze(1)
            np.save(f"/home/user/lanyinyu/EEG2EEG/Exp/{EXP_NAME}/generated_fullmodel/Sub{subid1+1}ToSub{subid2+1}.npy", testsub2fakedata)