import matplotlib
matplotlib.use('Agg')
import os
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


###############################################################################
####################### 1. LOAD THE TRAINING SET DATA #########################
###############################################################################
image_dir_1 = 'summer2winter_yosemite'
image_dir_2 = 'apple2orange'
image_dir_3 = 'horse2zebra'

def get_data_loader(image_type, image_size, batch_size, image_dir, num_workers=0):
    """
    Returns train and test loaders for a given image type (domain) of the given application 
    *Images will be resized to 128x128x3, converted into Tensors, and normalized.
    : param image_type: (str) "A" or "B" (which of the 2 domains)
    : param image_size: (int) image width/height (assumes square image)
    : param batch_size: (int) batch size
    : param image_dir: (str) directory name for images (list above)
    """
    transform = transforms.Compose([transforms.Resize(image_size), # resize to 128x128
                                    transforms.ToTensor()])

    image_path = image_dir
    train_path = os.path.join(image_path, 'train{}'.format(image_type))
    test_path = os.path.join(image_path, 'test{}'.format(image_type))

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

img_dir = image_dir_3
img_size = 128
batch_size = 1
dataloader_X, test_dataloader_X = get_data_loader(image_type='A', image_size = img_size, 
                                                  batch_size = batch_size, image_dir = img_dir)
dataloader_Y, test_dataloader_Y = get_data_loader(image_type='B', image_size = img_size, 
                                                  batch_size = batch_size, image_dir = img_dir)

print('# of horse images: ', len(dataloader_X))
print('# of zebra images: ', len(dataloader_Y))


###############################################################################
###################### 2. DEFINE DISCRIMINATOR NETWORK ########################
###############################################################################
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization."""
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    
    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()
        # first layer has *no* batchnorm
        self.conv1 = conv(3, conv_dim, kernel_size = 4, batch_norm=False)
        self.conv2 = conv(conv_dim, conv_dim*2, kernel_size = 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, kernel_size = 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, kernel_size = 4)
        self.conv5 = conv(conv_dim*8, 1, kernel_size = 4, stride = 1, batch_norm=False)

    def forward(self, x):
        ########################################  Input x :      (3, 128, 128)
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.2)  # (64, 64, 64)
        x = F.leaky_relu(self.conv2(x), negative_slope = 0.2)  # (128, 32, 32)
        x = F.leaky_relu(self.conv3(x), negative_slope = 0.2)  # (256, 16, 16)
        x = F.leaky_relu(self.conv4(x), negative_slope = 0.2)  # (512, 8, 8)
        x = self.conv5(x)                                      # (5, 5, 1)
        
        return x
    

###############################################################################
######################### 3. DEFINE RESIDUAL BLOCK ############################
###############################################################################
class ResidualBlock(nn.Module):
    """Defines a residual block.
       This adds an input x to a convolutional layer (applied to x) with the same size input and output.
       These blocks allow a model to learn an effective transformation from one domain to another.
    """
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        # define two convolutional layers + batch normalization that will act as our residual function, F(x)
        # layers should have the same shape input as output; I suggest a kernel_size of 3
        self.refl_padding = nn.ReflectionPad2d(1)
        self.conv1 = conv(conv_dim, conv_dim, kernel_size=3, stride=1, padding=0)
        self.conv2 = conv(conv_dim, conv_dim, kernel_size=3, stride=1, padding=0)
        
    def forward(self, x):
        # apply a ReLu activation the outputs of the first layer
        # return a summed output, x + resnet_block(x)
        out = self.refl_padding(x)   # (256, 34, 34)
        out = F.relu(self.conv1(out))  # (256, 32, 32)
        out = self.refl_padding(out)   # (256, 34, 34)
        out = self.conv2(out)          # (256, 32, 32)
        x = x + out
        
        return x    
    
###############################################################################
######################### 4. DEFINE GENERATOR NETWORK #########################
###############################################################################    
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, 
           output_padding=1, batch_norm=True):
    """Creates a transpose convolutional layer, with optional batch normalization."""
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, 
                                     padding, output_padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)


class CycleGenerator(nn.Module):
    
    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()
        # 1. Define the encoder part of the generator
        self.refl_padding = nn.ReflectionPad2d(3)
        
        self.conv1 = conv(3, conv_dim, kernel_size=7, stride=1, padding=0)
        self.conv2 = conv(conv_dim, conv_dim*2, kernel_size = 3)
        self.conv3 = conv(conv_dim*2, conv_dim*4, kernel_size = 3) 

        # 2. Define the resnet part of the generator
        res_layers = []
        for layer in range(n_res_blocks):
            res_layers.append(ResidualBlock(conv_dim*4))
        self.res_blocks = nn.Sequential(*res_layers) 

        # 3. Define the decoder part of the generator
        self.deconv1 = deconv(conv_dim*4, conv_dim*2, kernel_size = 3)   
        self.deconv2 = deconv(conv_dim*2, conv_dim, kernel_size = 3)   
        # no batch norm on last layer
        self.conv_final = conv(conv_dim, 3, kernel_size=7, stride=1, padding=0, batch_norm=False)   # (3, 128, 128)

    def forward(self, x):
        """Given an image x, returns a transformed image."""
        ##############  Input x :      (3, 128, 128)
        out = self.refl_padding(x)     # (3, 134, 134)
        
        out = F.relu(self.conv1(out))    # (64, 128, 128)
        out = F.relu(self.conv2(out))    # (128, 64, 64)
        out = F.relu(self.conv3(out))    # (256, 32, 32)

        out = self.res_blocks(out)       # (256, 32, 32)

        out = F.relu(self.deconv1(out))  # (128, 64, 64)
        out = F.relu(self.deconv2(out))  # (64, 128, 128)
        
        out = self.refl_padding(out)     # (64, 134, 134)
        # skip layer
        out = F.tanh(x + self.conv_final(out))
        #out = F.tanh(self.conv_final(out))  # (3, 128, 128)

        return out
    

###############################################################################
###################### 5. FUNCTION TO INITIALIZE WEIGHTS ######################
###############################################################################
from torch.nn import init

def init_weights(m):
    
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.normal_(m.weight.data, 0.0, 0.02)
        
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
        
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


###############################################################################
################# 6. BUILD D & G MODELS & INITIALIZE WEIGHTS ##################
###############################################################################
def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    
    G_XtoY = CycleGenerator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
    G_XtoY.apply(init_weights)
    G_YtoX = CycleGenerator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
    G_YtoX.apply(init_weights)
    
    D_X = Discriminator(conv_dim=d_conv_dim)
    D_X.apply(init_weights)
    D_Y = Discriminator(conv_dim=d_conv_dim)
    D_Y.apply(init_weights)
    

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')

    return G_XtoY, G_YtoX, D_X, D_Y

G_XtoY, G_YtoX, D_X, D_Y = create_model()
'''
D_X.load_state_dict(torch.load('D_X.pt'))
D_Y.load_state_dict(torch.load('D_Y.pt'))
G_XtoY.load_state_dict(torch.load('G_XtoY.pt'))
G_YtoX.load_state_dict(torch.load('G_YtoX.pt'))
'''

###############################################################################
#################### 7. FUNCTION TO SCALE THE PIXEL VALUES ####################
###############################################################################
def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-255.'''
    # scale from 0-1 to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    
    return x


###############################################################################
####################### 8. DEFINE LOSSES & OPTIMIZERS #########################
###############################################################################
def real_mse_loss(D_out):
    return torch.mean((D_out-1)**2)

def fake_mse_loss(D_out):
    return torch.mean(D_out**2)

def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
    # calculate reconstruction loss as absolute (l1) value difference between the real & reconstructed images
    reconstr_loss = torch.mean(torch.abs(real_im - reconstructed_im))
        
    return lambda_weight * reconstr_loss    


import torch.optim as optim
beta1=0.5
beta2=0.999

def lr_decay(params, epoch):
    if epoch <= 100:
         lr = 0.0002
    else:
         lr = -(epoch-100)*0.0002/100 + 0.0002
    
    return optim.Adam(params, lr, [beta1, beta2])


###############################################################################
###### 9. FUNCTION TO MAKE A BUFFER OF 50 FAKE IMAGES FOR DISCRIMINATOR #######
###############################################################################
# History of 50 images to use when updating the discriminator (pool_size = 50)
import random

def pick_from_pool(pool_size, fakes):
    
    fakes = [each.data for each in fakes]
    if len(fakes) <= pool_size:
        random_index = random.randint(0, len(fakes)-1)
        return fakes[random_index]
    
    else:
        fakes_50 = fakes[-50:]
        random_index = random.choice(np.arange(50))
        return fakes_50[random_index]


###############################################################################
######## 10. FUNCTION TO CONVERT A TORCH TENSOR IMAGE INTO NUMPY ARRAY ########
###############################################################################
# for saving & plotting processed images
def to_numpy(img):
    
    if torch.cuda.is_available:
        img = img.cpu()
    img = img.data.numpy()
    img = (((img+1)*255)/2).astype(np.uint8)
    img = img.squeeze(0).transpose(1, 2, 0)
    
    return img


###############################################################################
############################ 11. TRAIN THE NETWORK ############################
###############################################################################
import tqdm

def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, n_epochs):
    
    print_every=1
    losses = []
    
    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = test_iter_X.next()[0]
    fixed_Y = test_iter_Y.next()[0]
    fixed_X = scale(fixed_X) # make sure to scale to a range -1 to 1
    fixed_Y = scale(fixed_Y)

    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)
    #batches_per_epoch = min(len(iter_X), len(iter_Y))
    smaller = min(len(dataloader_X), len(dataloader_Y))
    batches_per_epoch = int(np.floor(smaller / batch_size)) # how many batches are in an epoch
    print('Batches per epoch: ', batches_per_epoch)
    
    for epoch in range(1, n_epochs+1):
        
        g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters
        g_optimizer = lr_decay(g_params, epoch)
        d_x_optimizer = lr_decay(D_X.parameters(), epoch)
        d_y_optimizer = lr_decay(D_Y.parameters(), epoch)
        
        fakes_X, fakes_Y = [], []
        #for i in tqdm.tqdm(range(batches_per_epoch)):
        for i in tqdm.tqdm(range(900)):

            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)
            
            images_X, _ = iter_X.next()
            images_X = scale(images_X) # make sure to scale to a range -1 to 1

            images_Y, _ = iter_Y.next()
            images_Y = scale(images_Y)
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            images_X = images_X.to(device)
            images_Y = images_Y.to(device)


            # ============================================
            #            TRAIN THE DISCRIMINATORS
            # ============================================
            ##   1. D_X   ##
            d_x_optimizer.zero_grad()
            D_X_real_loss = real_mse_loss(D_X(images_X))

            fake_X = G_YtoX(images_Y)
            ###
            if i < 50:
                fakes_X.append(fake_X)
                fake_random = pick_from_pool(50, fakes_X)
            else:
                fakes_X.pop(0)
                fakes_X.append(fake_X)
                fake_random = pick_from_pool(50, fakes_X)
            ###
            D_X_fake_loss = fake_mse_loss(D_X(fake_random))

            d_x_loss = (D_X_real_loss + D_X_fake_loss) * 0.5
            d_x_loss.backward()
            d_x_optimizer.step()


            ##   2. D_Y   ##
            d_y_optimizer.zero_grad()
            D_Y_real_loss = real_mse_loss(D_Y(images_Y))

            fake_Y = G_XtoY(images_X)
            ###
            if i < 50:
                fakes_Y.append(fake_Y)
                fake_random = pick_from_pool(50, fakes_Y)
            else:
                fakes_Y.pop(0)
                fakes_Y.append(fake_Y)
                fake_random = pick_from_pool(50, fakes_Y)
            ###
            D_Y_fake_loss = fake_mse_loss(D_Y(fake_Y))

            d_y_loss = (D_Y_real_loss + D_Y_fake_loss) * 0.5
            d_y_loss.backward()
            d_y_optimizer.step()


            # =========================================
            #            TRAIN THE GENERATORS
            # =========================================
            g_optimizer.zero_grad()
            # 1. YtoX
            fake_X = G_YtoX(images_Y)
            g_YtoX_loss = real_mse_loss(D_X(fake_X))

            reconstructed_Y = G_XtoY(fake_X)
            reconstructed_y_loss = cycle_consistency_loss(images_Y, reconstructed_Y, lambda_weight=10)

            # 2. XtoY
            fake_Y = G_XtoY(images_X)
            g_XtoY_loss = real_mse_loss(D_Y(fake_Y))

            reconstructed_X = G_YtoX(fake_Y)
            reconstructed_x_loss = cycle_consistency_loss(images_X, reconstructed_X, lambda_weight=10)

            # 3. Add up all G losses
            g_total_loss = g_YtoX_loss + g_XtoY_loss + reconstructed_y_loss + reconstructed_x_loss
            g_total_loss.backward()
            g_optimizer.step()

        
        if epoch % print_every == 0:
            # append real and fake discriminator losses and the generator loss
            losses.append((d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
            print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                    epoch, n_epochs, d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))


        sample_every=5
        # Save the generated samples
        if epoch % sample_every == 0:
            # Save model parameters
            torch.save(D_X.state_dict(), 'D_X.pt')
            torch.save(D_Y.state_dict(), 'D_Y.pt')
            torch.save(G_XtoY.state_dict(), 'G_XtoY.pt')
            torch.save(G_YtoX.state_dict(), 'G_YtoX.pt')
        
            G_YtoX.eval() # set generators to eval mode for sample generation
            G_XtoY.eval()
            
            fake_sample_Y = G_XtoY(fixed_X.cuda())
            recon_sample_X = G_YtoX(fake_sample_Y)
            
            fake_sample_X = G_YtoX(fixed_Y.cuda())
            recon_sample_Y = G_XtoY(fake_sample_X)
            
            X, fake_X, recon_Y = to_numpy(fixed_X), to_numpy(fake_sample_X), to_numpy(recon_sample_Y)
            Y, fake_Y, recon_X = to_numpy(fixed_Y), to_numpy(fake_sample_Y), to_numpy(recon_sample_X)
            
            if(epoch == sample_every):
                plt.imsave('Input_X.png', X)
                plt.imsave('Input_Y.png', Y)
            
            plt.imsave('fake_sample_Y_'+str(epoch)+'.png', fake_Y)
            plt.imsave('recon_sample_X_'+str(epoch)+'.png', recon_X)
            
            plt.imsave('fake_sample_X_'+str(epoch)+'.png', fake_X)
            plt.imsave('recon_sample_Y_'+str(epoch)+'.png', recon_Y)
            print('Saved samples.')
            
            G_YtoX.train()
            G_XtoY.train()

    return losses

n_epochs = 55
losses = training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, n_epochs=n_epochs)
