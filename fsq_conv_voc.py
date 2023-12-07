import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision
# from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
# from torchvision.utils import save_image
import cv2 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import json 
from copy import deepcopy 
import os 


class Residual_Block(nn.Module):
    def __init__(self, x_channels, residual_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=x_channels, out_channels=residual_channels, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=residual_channels, out_channels=x_channels, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        h = self.conv1(x)
        h = F.relu(h)
        h = self.conv2(h)
        h = h + x 
        h = F.relu(h)
        return h 
    

class Residual_Stack(nn.Module):
    def __init__(self, num_blocks, x_channels, residual_channels):
        super().__init__()
        self.blocks = nn.ModuleList([ deepcopy( Residual_Block(x_channels, residual_channels) ) for _ in range(num_blocks) ])

    def forward(self, x):
        for block in self.blocks: 
            x = block(x) 
        return x 


class FSQ_Conv(nn.Module):
    def __init__(self, device, num_blocks, num_quantized_values):
        super().__init__()
        self.L = num_quantized_values
        self.embed_dim = len(num_quantized_values) # d in fsq paper
        self.codebook = self.get_implicit_codebook(num_quantized_values).to(device) 
        num_embed = self.codebook.shape[0]
        self.codebook_usage = torch.zeros(num_embed) # like eligibility traces to measure codebook usage

        self.enc_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1) # [b,3,128,128] -> [b,6,64,64]
        self.enc_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1) # [b,6,64,64] -> [b,12,32,32]
        self.enc_residual_stack1 = Residual_Stack(num_blocks, 128, 32) # [b,12,32,32] -> [b,12,32,32]
        self.enc_conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1) # [b,12,32,32] -> [b,24,16,16]
        self.enc_conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1) # [b,24,16,16] -> [b,48,8,8]
        self.enc_residual_stack2 = Residual_Stack(num_blocks, 512, 128)
        self.enc_conv5 = nn.Conv2d(in_channels=512, out_channels=self.embed_dim, kernel_size=3, stride=1, padding='same') # [b,48,8,8] -> [b,embed_dim,8,8]; so z_dim = embed_dim with z_seqlen = 8*8=64

        self.dec_conv5 = nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=512, kernel_size=3, stride=1, padding=1) # [b,d,8,8] -> [b,48,8,8]
        self.dec_residual_stack2 = Residual_Stack(num_blocks, 512, 128) 
        self.dec_conv4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) # [b,48,8,8] -> [b,24,16,16]
        self.dec_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) # [b,24,16,16] -> [b,12,32,32]
        self.dec_residual_stack1 = Residual_Stack(num_blocks, 128, 32) # [b,12,32,32] -> [b,12,32,32]
        self.dec_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) # [b,12,32,32] -> [b,6,64,64]
        self.dec_conv1 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1) # [b,6,64,64] -> [b,3,128,128]
        self.device = device 

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x)) 
        x = self.enc_residual_stack1(x)
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = self.enc_residual_stack2(x)
        x = self.enc_conv5(x) # no relu here to allow negative values 
        x = x.permute(0, 2, 3, 1) # x.shape: [b,d,8,8] -> [b,8,8,d]
        z_e = x.flatten(start_dim=0, end_dim=-2) # x.shape: [b,8,8,d] -> [b*64,d]
        return z_e 
    
    # recursive function to build codebook
    def build_codebook_recursive(self, L, p):
        x = L[p]
        # base case 
        if p == len(L)-1:
            return torch.tensor([i for i in range(-int(x/2), int(x/2)+1)]).unsqueeze(-1)
        # recursive case 
        s = self.build_codebook_recursive(L, p+1)
        new_s = []
        for i in range(-int(x/2), int(x/2)+1):
            i_vec = torch.tensor([i]).unsqueeze(0).expand(s.shape[0],-1)
            new_s.append( torch.cat([i_vec, s], dim=1) )
        return torch.cat(new_s, dim=0)

    # function to explicitly prepare the implicit codebook
    def get_implicit_codebook(self, L):
        # recursive function to build codebook
        codebook = self.build_codebook_recursive(L, 0)
        return codebook.int()
    
    def get_codebook_usage(self, idx):
        with torch.no_grad():
            unique = torch.unique(idx).shape[0]
            # increment time elapsed for all codebook vectors 
            self.codebook_usage += 1
            # reset time for matched codebook vectors 
            self.codebook_usage[idx] = 0
            # measure usage 
            usage = torch.sum(torch.exp(-self.codebook_usage))
            usage /= self.codebook.shape[0]
            return usage, unique  

    def quantize(self, z_e): # z_e.shape: [b*64, d]
        L = torch.tensor(self.L).unsqueeze(0).to(self.device) # [1, d]
        z_e_squashed = (L/2).int() * torch.tanh(z_e) # [b*64, d]
        z_q = torch.round(z_e_squashed)
        z = (z_q - z_e_squashed).detach() + z_e_squashed # for straight through gradent 
        # get index of the quantized vector in the implicit codebook
        idx_bools = torch.eq(z.int().unsqueeze(1), self.codebook).all(dim=-1)
        idx = torch.nonzero(idx_bools, as_tuple=True)[1] # idx.shape = [b*64]
        # get usage of the implicit codebook 
        usage, unique = self.get_codebook_usage(idx)
        return z, z_e, z_q, usage, unique, idx  

    def decode(self, z): # z.shape: [b*64, d]
        z = z.view(-1, 64, z.shape[-1]) # [b*64, d] -> [b, 64, d]
        z = z.view(z.shape[0], 8, 8, z.shape[-1]) # [b, 64, d] -> [b, 8, 8, d]
        z = z.permute(0, 3, 1, 2) # [b, 8, 8, d] -> [b, d, 8, 8]
        z = F.relu(self.dec_conv5(z))
        z = self.dec_residual_stack2(z)
        z = F.relu(self.dec_conv4(z))
        z = F.relu(self.dec_conv3(z))
        z = self.dec_residual_stack1(z)
        z = F.relu(self.dec_conv2(z))
        x = self.dec_conv1(z) # no relu here to allow all pixel values in img 
        # x = torch.tanh(z) # project all pixel values to be in range [-1, 1] since training imgs are in this range - NOTE this is not necessary and dilutes the loss signal
        return x

    def forward(self, x):
        z_e = self.encode(x)
        z, z_e, z_q, usage, unique, idx = self.quantize(z_e)
        x = self.decode(z)
        return x, z_e, z_q, usage, unique   
    
# convert flat tensor to img
def to_img(x):
    x = x.clamp(-1, 1) # clamp img to be strictly in [-1, 1]
    x = 0.5 * x + 0.5 # transform img from range [-1, 1] -> [0, 1]
    x = x.permute(0,2,3,1)
    return x

# function to generate img from VAE 
def generate_img(vae, latent_size, device):
    z = torch.FloatTensor(latent_size).uniform_().to(device)
    x_sampled = vae.decode(z)
    img_sampled = to_img(x_sampled)
    return img_sampled 

# VQVAE loss 
def loss_function(recon_x, x):
    criterion = nn.MSELoss(reduction='mean')
    reconstruction_loss = criterion(recon_x, x)
    return reconstruction_loss

# function to save a test img and its reconstructed img 
def save_img_reconstructed(x, x_r, save_path):
    concat_img = torch.cat([x, x_r], dim=1)
    concat_img = concat_img.detach().cpu().numpy()
    concat_img = np.uint8( concat_img * 255 )
    # bgr to rgb 
    # concat_img = concat_img[:, :, ::-1]
    cv2.imwrite(save_path, concat_img)

# utility function to load model weights from checkpoint - loads to the device passed as 'device' argument
def load_ckpt(checkpoint_path, model, optimizer=None, scheduler=None, device=torch.device('cpu'), mode='eval'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if mode == 'eval':
        model.eval() 
        return model
    else:
        model.train()
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            return model, optimizer, scheduler
        else:
            return model, optimizer

# utility function to save a checkpoint (model_state, optimizer_state, scheduler_state) - saves on cpu (to save gpu memory)
def save_ckpt(device, checkpoint_path, model, optimizer, scheduler=None):
    # transfer model to cpu
    model = model.to('cpu')
    # prepare dicts for saving
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, checkpoint_path)
    # load model back on original device 
    model = model.to(device)


# utility function to load img and captions data 
def load_data():
    imgs_folder = '/home/vivswan/experiments/muse/dataset_coco_val2017/images/'
    captions_file_path = '/home/vivswan/experiments/muse/dataset_coco_val2017/annotations/captions_val2017.json'
    captions_file = open(captions_file_path)
    captions = json.load(captions_file)
    img_dict, img_cap_pairs = {}, []
    print('Loading Data...')
    num_iters = len(captions['images']) + len(captions['annotations'])
    pbar = tqdm(total=num_iters)

    for img in captions['images']:
        id, file_name = img['id'], img['file_name']
        img_dict[id] = file_name
    for cap in captions['annotations']:
        id, caption = cap['image_id'], cap['caption']
        # use img_name as key for img_cap_dict
        img_filename = img_dict[id]

        # load image from img path 
        img_path = imgs_folder + img_filename
        img = cv2.imread(img_path, 1)
        resize_shape = (img_size, img_size)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) / 255
        img = torch.tensor(img)
        img = img.permute(2, 0, 1) # [w,h,c] -> [c,w,h]
        transforms = torchvision.transforms.Compose([
            # NOTE: no random resizing and cropping here as it wouldn't really induce robustness (for robustness, we should do this when sampling a minibatch) 
            # torchvision.transforms.Resize( int(1.25*img_size) , antialias=True),  # image_size + 1/4 * image_size
            # torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0) , antialias=True),
            # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # zero mean, unit std
        ])
        img = transforms(img)

        # append prefix text to caption : "An example of"
        # caption = 'An example of ' + caption + ': '
        img_cap_pairs.append([img, caption])
        pbar.update(1)
    pbar.close()
    return img_cap_pairs


# utility function to process minibatch - convert img_filenames to augmented_imgs 
def process_batch(minibatch, img_size, device):
    # augmented_imgs = []
    augmented_imgs, captions = list(map(list, zip(*minibatch)))

    # # get augmented imgs
    # imgs_folder = '/home/vivswan/experiments/muse/dataset_coco_val2017/images/'
    # for img_filename in img_files:
    #     img_path = imgs_folder + img_filename
    #     img = cv2.imread(img_path, 1)
    #     resize_shape = (img_size, img_size)
    #     img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
    #     img = np.float32(img) / 255
    #     img = torch.tensor(img)
    #     img = img.permute(2, 0, 1) # [w,h,c] -> [c,w,h]
    #     transforms = torchvision.transforms.Compose([
    #         torchvision.transforms.Resize( int(1.25*img_size) , antialias=True),  # image_size + 1/4 * image_size
    #         torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0) , antialias=True),
    #         # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
    #         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # zero mean, unit std
    #     ])
    #     img = transforms(img)
    #     augmented_imgs.append(img)

    augmented_imgs = torch.stack(augmented_imgs, dim=0).to(device)
    return augmented_imgs




# main 
if __name__ == '__main__':
    num_quantized_values = [7, 5, 5, 5, 5] # L in fsq paper
    num_blocks = 2 # number of residual blocks in a residual stack
    img_size = 128 
    img_channels = 3
    img_shape = torch.tensor([img_channels, img_size, img_size])
    resize_shape = (img_size, img_size)
    max_epochs = 80000
    epochs_done = 74000
    batch_size = 256
    lr = 3e-4
    sample_img_freq = int(max_epochs/160)
    plot_freq = int(max_epochs/16)
    random_seed = 30

    checkpoint_path = './ckpts/FSQ_Conv.pt' # path to a save and load checkpoint of the trained model
    resume_training_from_ckpt = True          

    save_img_folder = './out_imgs_FSQ_Conv'
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)       

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load dataset
    dataset = load_data()
    dataset_len = len(dataset)

    # init model, loss_fn and optimizer
    model = FSQ_Conv(device, num_blocks, num_quantized_values).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) # no weight decay ?

    if resume_training_from_ckpt:
        model, optimizer = load_ckpt(checkpoint_path, model, optimizer, device=device, mode='train')

    # for plotting results
    results_train_loss = []
    results_codebook_usage = []
    results_codebook_unique = []

    max_grad_norm = -float('inf')

    # train 
    for epoch in tqdm(range(max_epochs)):
        epoch += epochs_done 

        # fetch minibatch
        idx = np.arange(dataset_len)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        minibatch = [dataset[i] for i in idx]

        # process minibatch - convert img_filenames to augmented_imgs 
        imgs = process_batch(minibatch, img_size, device) # imgs.shape:[batch_size, 3, 32, 32]

        # imgs = imgs.view(imgs.size(0), -1) # flatten all images in the batch [b, 3*32*32]
        recon_imgs, z_e, z_q, usage, unique = model(imgs) # imgs.shape: [b,c,w,h]
        loss = loss_function(recon_imgs, imgs)
        optimizer.zero_grad()
        loss.backward()
        # gradient cliping 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # # calculate max_grad_norm 
        # for p in model.parameters(): 
        #     grad_norm = p.grad.norm().item()
        #     if max_grad_norm < grad_norm:
        #         max_grad_norm = grad_norm
        #         print('max_grad_norm: ', max_grad_norm)

        results_train_loss.append(loss.item())
        results_codebook_usage.append(usage.item())
        results_codebook_unique.append(unique)

        if (epoch+1) % sample_img_freq == 0:
            x_r = to_img(recon_imgs.data)
            x = to_img(imgs.data)
            # img_generated = generate_img(model, mu.shape, device)
            save_img_reconstructed(x[0], x_r[0], save_img_folder + '/{}_reconstructed.png'.format(epoch))
            # save_image(img_generated, './cifar/out_imgs_VAE/{}_generated.png'.format(epoch))

        if (epoch+1) % plot_freq == 0:
            # save model checkpoint
            save_ckpt(device, checkpoint_path, model, optimizer)

            # plot results
            fig, ax = plt.subplots(2,2, figsize=(15,10))

            ax[0,0].plot(results_train_loss, label='train_loss')
            ax[0,0].legend()
            ax[0,0].set(xlabel='eval_iters')
            ax[1,0].plot(results_codebook_unique, label='codebook_unique')
            ax[1,0].legend()
            ax[1,0].set(xlabel='eval_iters')
            ax[0,1].plot(results_codebook_usage, label='codebook_usage')
            ax[0,1].legend()
            ax[0,1].set(xlabel='train_iters')
            ax[1,1].plot(results_codebook_usage, label='codebook_usage')
            ax[1,1].legend()
            ax[1,1].set(xlabel='train_iters')

            plt.suptitle('final_train_loss: ' + str(results_train_loss[-1]))
            plt.savefig(save_img_folder + '/plot_' + str(epoch) + '.png')
