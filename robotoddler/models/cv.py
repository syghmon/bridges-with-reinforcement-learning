import torch
from torch import nn
from torch.nn.functional import relu

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

    def forward(self, inputs):
        return self.layers(inputs)
    

class MLP(nn.Module):
    def __init__(self, in_d, out_d, hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = [64]

        cur_dim = in_d
        layers = []
        for next_dim in hidden_dim:
            layers.append(nn.Linear(cur_dim, next_dim))
            layers.append(nn.ReLU())
            cur_dim = next_dim
        layers.append(nn.Linear(cur_dim, out_d))

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.layers(inputs)
    

class ConvNet(nn.Module):
    """
    A plain conv net for predicting q-values and binary successor features
    """
    def __init__(self, in_channels=4, img_size=(512, 512), num_features=6):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(in_channels, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

        # compute bottleneck size
        with torch.no_grad():
            x = self.layers(torch.zeros(1, in_channels, *img_size))
            self.bottleneck_size = x.shape[1] * x.shape[2] * x.shape[3]

        self.mlp = MLP(self.bottleneck_size + num_features, 2 * num_features + 1)
        
    def forward(self, block_features, action_features):
        x = torch.cat([block_features, action_features], dim=1)
        x = self.layers(x)
        x = self.mlp(x.view(-1, self.bottleneck_size))
        return x
        
    #def forward(self, block_features, binary_features, action_features, reward_features, obstacle_features):
    #    x = torch.cat([block_features, action_features, reward_features, obstacle_features], dim=1)
    #    x = self.layers(x)
    #    x = self.mlp(torch.cat([x.view(-1, self.bottleneck_size), binary_features], dim=1))
    #    q_values, succ_binary_features = x[:,0], x[:,1:]
    #    succ_binary_features = succ_binary_features.view(-1, 2, binary_features.shape[1])
    #    return q_values, None, succ_binary_features
    

class SuccessorMLP(nn.Module):
    """
    A MLP with bottle net that predicts q_values, successor images and features.
    """
    def __init__(self, in_channels=4, img_size=(512, 512), num_features=6, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 128]
        self.img_size = img_size
        # calculate input and output dimensions
        dim_in = in_channels * img_size[0] * img_size[1] + num_features
        dim_out = 2 * img_size[0] * img_size[1] + 2 * num_features
        self.mlp = MLP(dim_in, dim_out, hidden_dims)

    def forward(self, block_features, binary_features, action_features, reward_features, obstacle_features):
        # concatenate all inputs
        x = torch.cat([block_features, action_features, reward_features, obstacle_features], dim=1).view(block_features.shape[0], -1)
        x = torch.cat([x, binary_features], dim=1)

        # mlp
        x = self.mlp(x)
        
        # split output into succ image and binary features
        img_dim = 2 * self.img_size[0] * self.img_size[1]
        succ_block_features = x[:, :img_dim].view(-1, 2, *self.img_size)
        succ_binary_features = x[:, img_dim:].view(-1, 2, binary_features.shape[1])

        # compute q values
        q_values = torch.sum(succ_block_features.softmax(dim=1)[:, 1] * reward_features.squeeze(1), dim=(-1, -2))
        return q_values, succ_block_features, succ_binary_features



class UNet_old(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_old, self).__init__()
        
        # Contracting path
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Expanding path
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
    def forward(self, block_features, action_features, reward_features, obstacle_features):
        x = torch.cat([block_features, action_features, reward_features, obstacle_features], dim=1)
        # Contracting path
        x = self.encoder(x)
        
        # Expanding path
        x = self.decoder(x)
        
        return x

# Code adapted from https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3    
class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        self.n_class = n_class
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 64x64x4
        self.e11 = nn.Conv2d(4, 16, kernel_size=3, padding=1) 
        self.e12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e21 = nn.Conv2d(16, 32, kernel_size=3, padding=1) 
        self.e22 = nn.Conv2d(32, 32, kernel_size=3, padding=1) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e31 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.e32 = nn.Conv2d(64, 64, kernel_size=3, padding=1) 
        
        if False:
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

            self.e41 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
            self.e42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.e51 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.e52 = nn.Conv2d(256, 256, kernel_size=3, padding=1)


            # Decoder
            self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.d11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
            self.d12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

            self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.d21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
            self.d22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(16, n_class, kernel_size=1)
        #self.stability_layer = nn.Linear(16*64*64, 1) # At the end of the UNet
        #self.stability_layer = nn.Linear(64*16*16, 1) # In the middle of the UNet
        
    def forward(self, block_features, binary_features, action_features, reward_features, obstacle_features):
        x = torch.cat([block_features, action_features, reward_features, obstacle_features], dim=1)
        
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        
        #stability = torch.sigmoid(self.stability_layer(xe32.flatten(1,3)))
        
        if False:
            xp3 = self.pool3(xe32)

            xe41 = relu(self.e41(xp3))
            xe42 = relu(self.e42(xe41))
            xp4 = self.pool4(xe42)

            xe51 = relu(self.e51(xp4))
            xe52 = relu(self.e52(xe51))
            
            # Decoder
            xu1 = self.upconv1(xe52)
            xu11 = torch.cat([xu1, xe42], dim=1)
            xd11 = relu(self.d11(xu11))
            xd12 = relu(self.d12(xd11))

            xu2 = self.upconv2(xd12)
            xu22 = torch.cat([xu2, xe32], dim=1)
            xd21 = relu(self.d21(xu22))
            xd22 = relu(self.d22(xd21))

        #xu3 = self.upconv3(xd22)
        xu3 = self.upconv3(xe32)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        succ_block_features = self.outconv(xd42)
        if self.n_class == 2:
            succ_block_features = succ_block_features.softmax(dim=1)[:, 1]
        
        #q_values = torch.sum(succ_block_features[:, 0] * (reward_features.squeeze(1) - obstacle_features.squeeze(1) - block_features.squeeze(1)), dim=(-1, -2)) + (stability.squeeze() - 1)
        #q_values = torch.sum(succ_block_features[:, 0] * reward_features.squeeze(1), dim=(-1, -2)) - torch.exp(-10 * stability.squeeze())
        
        return succ_block_features
        #return q_values, succ_block_features, torch.hstack([stability, binary_features[:,1:]])
    
    
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.SFImage = UNet(1)
        self.SFStability = ConvNet(in_channels=2, img_size=(64, 64), num_features=0) # Independent of pi !!! Only pass block and action features
        
    def forward(self, block_features, binary_features, action_features, reward_features, obstacle_features):
        succ_block_features = self.SFImage(block_features, binary_features, action_features, reward_features, obstacle_features)
        stability = torch.sigmoid(self.SFStability(block_features, action_features))
        
        # Assumes only non colliding actions are considered, otherwise we can add the collision term in the q values
        q_values = torch.sum(succ_block_features[:, 0] * reward_features.squeeze(1), dim=(-1, -2)) * (1 - torch.exp(-10 * stability.squeeze())) - torch.exp(-10 * stability.squeeze())
        
        return q_values, succ_block_features, stability