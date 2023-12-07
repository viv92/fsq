import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import cv2 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import json 
from copy import deepcopy 
import os 

from utils_transformer import * 

# class for transformer input embedding
class Transformer_Embeddings(nn.Module):
    def __init__(self, x_dim, max_seq_len, d_model, dropout, device):
        super().__init__()
        self.x_emb = nn.Linear(x_dim, d_model, bias=False)
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, d_model)) 
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        self.device = device
    def forward(self, x): # s.shape: [batch_size, max_seq_len, x_dim]
        batch_size, input_seq_len = x.shape[0], x.shape[1]
        x_emb = self.x_emb(x) # x_emb.shape: [batch_size, max_seq_len, d_model]
        # add positional embeddings
        pos_emb = self.pos_emb
        pos_emb = pos_emb.unsqueeze(0) # pos_emb.shape: [1, max_seq_len, d_model]
        pos_emb = pos_emb.expand(batch_size, -1, -1) # pos_emb.shape: [batch_size, max_seq_len, d_model]
        final_emb = self.dropout( self.norm(x_emb + pos_emb) )
        return final_emb
    

# class implementing Transformer 
# NOTE that since our goal here is just compression, this is a non-causal encoder-only transformer
class Transformer(nn.Module):
    def __init__(self, embedder, encoder, d_model, out_dim):
        super().__init__()
        self.embedder = embedder 
        self.encoder = encoder
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, x): # x.shape: [batch_size, max_seq_len, x_dim]
        x_emb = self.embedder(x) # x_emb.shape: [batch_size, max_seq_len, d_model]
        encoder_out = self.encoder(x_emb) # encoder_out.shape: [batch_size, max_seq_len, d_model]
        final_out = self.out_proj(encoder_out) # final_out.shape: [batch_size, max_seq_len, out_dim]
        return final_out
    

# caller function to instantiate the transformer, using the defined hyperparams as input
def init_transformer(x_dim, max_seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, out_dim, device):
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout) # multi head attention block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block for each encoder / decoder block
    embedder = Transformer_Embeddings(x_dim, max_seq_len, d_model, dropout, device) # embedder block to obtain sequence of embeddings from sequence of input tokens
    encoder_layer = EncoderLayer(deepcopy(attn), deepcopy(ff), d_model, dropout) # single encoder layer
    encoder = Encoder(encoder_layer, n_layers, d_model) # encoder = stacked encoder layers
    model = Transformer(embedder, encoder, d_model, out_dim) # a non-causal encoder-only transfomer
    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
    

# class implementing finite-scalar based VQ-VAE, with transformer based encoder-decoder
class FSQ_Transformer(nn.Module):
    def __init__(self, device, num_quantized_values, encoder_transformer, decoder_transformer, max_seq_len):
        super().__init__()
        self.L = num_quantized_values
        self.latent_dim = len(num_quantized_values) # d in fsq paper
        self.num_latents = max_seq_len 
        self.codebook = self.get_implicit_codebook(num_quantized_values).to(device) 
        self.codebook_size = self.codebook.shape[0]
        self.codebook_usage = torch.zeros(self.codebook_size) # like eligibility traces to measure codebook usage
        self.encoder = encoder_transformer 
        self.decoder = decoder_transformer 
        self.device = device 

    def encode(self, x): # [b, max_seq_len, x_dim] NOTE that max_seq_len = num_latents per x
        x = self.encoder(x) # [b, max_seq_len, latent_dim]
        z_e = x.flatten(start_dim=0, end_dim=-2) # [b, max_seq_len, latent_dim] -> [b * max_seq_len, latent_dim]
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

    def quantize(self, z_e): # z_e.shape: [b * max_seq_len, latent_dim]
        L = torch.tensor(self.L).unsqueeze(0).to(device) # [1, latent_dim]
        z_e_squashed = (L/2).int() * torch.tanh(z_e) # [b * max_seq_len, latent_dim]
        z_q = torch.round(z_e_squashed)
        z = (z_q - z_e_squashed).detach() + z_e_squashed # for straight through gradent 
        # get index of the quantized vector in the implicit codebook
        idx_bools = torch.eq(z.int().unsqueeze(1), self.codebook).all(dim=-1)
        idx = torch.nonzero(idx_bools, as_tuple=True)[1] # idx.shape = [b * max_seq_len]
        # get usage of the implicit codebook 
        usage, unique = self.get_codebook_usage(idx)
        return z, z_e, z_q, usage, unique, idx  

    def decode(self, z): # z.shape: [b * max_seq_len, latent_dim]
        z = z.view(-1, self.num_latents, z.shape[-1]) # [b * max_seq_len, latent_dim] -> [b, max_seq_len, latent_dim]
        x = self.decoder(z) # [b, max_seq_len, x_dim]
        # x = torch.tanh(z) # project all pixel values to be in range [-1, 1] since training imgs are in this range - NOTE this is not necessary and dilutes the loss signal
        return x

    def forward(self, x):
        z_e = self.encode(x)
        z, z_e, z_q, usage, unique, idx = self.quantize(z_e)
        x = self.decode(z)
        return x, z_e, z_q, usage, unique   
    

# utility function to convert img to sequence of patches (used to process img data)
def img_to_patch_seq(img, patch_size, num_latents):
    b,c,h,w = img.shape
    p = patch_size 
    assert (h % p == 0) and (w % p == 0)
    seq_len = (h * w) // (p * p)
    assert num_latents == seq_len 
    n_rows, n_cols = h//p, w//p
    patch_seq = []
    for row in range(n_rows):
        for col in range(n_cols):
            i = row * p
            j = col * p
            patch = img[:, :, i:i+p, j:j+p]
            patch = torch.flatten(patch, start_dim=1, end_dim=-1) # patch.shape: [b, patch_dim]
            patch_seq.append(patch)
    patch_seq = torch.stack(patch_seq, dim=0) # patch_seq.shape: [seq_len, b, patch_dim]
    patch_seq = patch_seq.transpose(0, 1) # patch_seq.shape: [b, seq_len, patch_dim]
    # assert seq_len == patch_seq.shape[1]
    # assert patch_dim == patch_seq.shape[-1]
    return patch_seq

# utility function to convert sequence of patches to an img (used to process img data)
def patch_seq_to_img(x, patch_size, channels): # x.shape: [batch_size, max_seq_len, patch_dim]
    batch_size, seq_len, patch_dim = x.shape[0], x.shape[1], x.shape[2]
    p = patch_size 
    x = x.permute(1, 0, 2) # x.shape: [seq_len, batch_size, patch_dim]
    x = x.reshape(seq_len, batch_size, channels, p*p)
    x = x.reshape(seq_len, batch_size, channels, p, p)
    nrows = int(math.sqrt(seq_len))
    ncols = nrows
    for i in range(nrows):
        for j in range(ncols):
            if j == 0:
                row = x[i * nrows + j]
            else:
                row = torch.cat((row, x[i * nrows + j]), dim=-1)
        if i == 0:
            imgs = row # row.shape: [batch_size, 3, p, w]
        else:
            imgs = torch.cat((imgs, row), dim=-2) # imgs.shape: [batch_size, 3, h, w]
    # imgs = imgs.permute(0, 2, 3, 1) # imgs.shape: [batch_size, h, w, 3]
    return imgs 

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
    concat_img = concat_img[:, :, ::-1]
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


# fetch dataset - using data loader
def mnist_dl(img_size, batch_size):
    tf = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(img_size),  # args.image_size 
        # torchvision.transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5) # equivalent to transforming pixel values from range [0,1] to [-1,1]
    ])
    dataset = MNIST(
        "/home/vivswan/experiments/fsq/dataset_mnist",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    return dataloader




# main 
if __name__ == '__main__':
    # hyperparams
    num_quantized_values = [7, 5, 5, 5, 5] # L in fsq paper
    latent_dim = len(num_quantized_values)

    img_size = 28 # mnist
    img_channels = 1
    img_shape = torch.tensor([img_channels, img_size, img_size])
    resize_shape = (img_size, img_size)
    patch_size = 4 # 8 # so that we have 16 latents per img which should be sufficient
    assert img_size % patch_size == 0
    patch_dim = img_channels * (patch_size**2)

    seq_len = (img_size // patch_size) ** 2 # equal to num latents per item
    
    d_model = patch_dim * 4
    n_heads = 8
    assert d_model % n_heads == 0
    d_k = d_model // n_heads 
    d_v = d_k 
    n_layers = 6
    d_ff = d_model * 4
    dropout = 0.1

    max_epochs = 200
    epochs_done = 0
    batch_size = 512
    lr = 3e-4
    sample_img_freq = int(max_epochs/40)
    plot_freq = int(max_epochs/2)
    random_seed = 10

    checkpoint_path = f'./ckpts/FSQ_Transformer_mnist|patchSize={patch_size}.pt' # path to a save and load checkpoint of the trained model
    resume_training_from_ckpt = False               

    save_img_folder = f'./out_imgs_FSQ_Transformer_mnist|patchSize={patch_size}'
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)       

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load img dataset
    dataloader = mnist_dl(img_size, batch_size)

    # init transformer encoder
    encoder_transformer = init_transformer(patch_dim, seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, latent_dim, device)
    # init transformer decoder
    decoder_transformer = init_transformer(latent_dim, seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, patch_dim, device)
    # init FSQ_Transformer 
    model = FSQ_Transformer(device, num_quantized_values, encoder_transformer, decoder_transformer, seq_len).to(device)

    # init optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr) # no weight decay ?

    if resume_training_from_ckpt:
        model, optimizer = load_ckpt(checkpoint_path, model, optimizer, device=device, mode='train')

    # for plotting results
    results_train_loss = []
    results_codebook_usage = []
    results_codebook_unique = []

    max_grad_norm = -float('inf')

    # train 
    for epoch in tqdm(range(epochs_done, epochs_done + max_epochs)):

        # fetch minibatch
        for imgs, labels in dataloader:
            
            imgs = imgs.to(device) # [b,c,w,h]

            # convert img to sequence of patches
            x = img_to_patch_seq(imgs, patch_size, seq_len) # x.shape: [b, seq_len, patch_dim]

            # forward prop through FSQ 
            recon_x, z_e, z_q, usage, unique = model(x) # recon_x.shape: [b, seq_len, patch_dim]

            # train step
            loss = loss_function(recon_x, x)
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
            # convert patch sequence to img 
            recon_imgs = patch_seq_to_img(recon_x, patch_size, img_channels)

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
