import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from manotorch.manolayer import ManoLayer, MANOOutput
from torch.nn import functional as F
from model_diffusion import *
import torch.nn.init as init

class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=256):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout

class HandShapeEncoder(nn.Module):
    def __init__(self, channel_num, latent_dim, num_points):
        super(HandShapeEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(channel_num, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(num_points, 1024),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )
        self.fcc = nn.Linear(256, 1)
        self.enc_mu = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Softsign()
        )
        self.enc_var = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Softsign()
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.conv2(x)
        x = self.fc2(x)
        x = self.conv3(x)
        x = self.fc3(x)
        x = self.fcc(x)
        x = torch.squeeze(x)
        return self.enc_mu(x), self.enc_var(x)


class HandShapeDecoder(nn.Module):
    def __init__(self, channel_num, latent_dim, num_points):
        super(HandShapeDecoder, self).__init__()
        self.dec_bn1 = nn.BatchNorm1d(latent_dim)
        self.dec_rb1 = ResBlock(latent_dim, latent_dim*2)
        self.dec_rb2 = ResBlock(latent_dim*2, latent_dim*4)
        self.dec_pose = nn.Sequential(
            nn.Linear(latent_dim*4, 45),
            nn.Linear(45, 45),
            nn.Sigmoid()
        )
        self.dec_shape = nn.Sequential(
            nn.Linear(latent_dim*4, 10),
            nn.Linear(10, 10),
            nn.Sigmoid()
        )
        self.rot_metrix = nn.Sequential(
            nn.Linear(latent_dim*4, 9),
            nn.Linear(9, 9),
            nn.Sigmoid()
        )
        self.dec_contact = nn.Linear(latent_dim * 4, 778)
        self.num_points = num_points
        self.channel_num = channel_num
        self.mano_layer = ManoLayer(center_idx=0, mano_assets_root='mano')

    def forward(self, x):
        x = self.dec_bn1(x)
        x = self.dec_rb1(x)
        x = self.dec_rb2(x)
        shape = self.dec_shape(x)
        pose  = self.dec_pose(x)
        rot_metrix = self.rot_metrix(x)

        zero_pose = torch.zeros(x.size(dim=0), 3, dtype=torch.float,
                           device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        pose = torch.cat((zero_pose, pose), dim=1)

        mano_output: MANOOutput = self.mano_layer(pose, shape)
        origins = mano_output.joints[:, 0, :].unsqueeze(1)

        return mano_output.verts - origins, mano_output.joints - origins, rot_metrix


class HandShapeAutoencoder(nn.Module):
    def __init__(self, channel_num, latent_dim, num_points):
        super(HandShapeAutoencoder, self).__init__()
        self.encoder = HandShapeEncoder(channel_num, latent_dim, num_points)
        self.decoder = HandShapeDecoder(channel_num, latent_dim, num_points)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        mean, logvar = self.encoder(x)
        hand_verts, hand_joints, rot_metrix = self.decoder(mean)
        return hand_verts, hand_joints, rot_metrix, mean, logvar


class glove_to_keypoints_20230922(nn.Module):
    def __init__(self):
        super(glove_to_keypoints_20230922, self).__init__()
        self.fc1 = nn.Linear(in_features=18, out_features=128, bias=False)
        self.fc2 = nn.Linear(in_features=128, out_features=512, bias=False)
        self.fc3 = nn.Linear(in_features=512, out_features=1024, bias=False)
        self.fc4 = nn.Linear(in_features=1024, out_features=1024, bias=False)
        self.fc5 = nn.Linear(in_features=1024, out_features=21*3, bias=False)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        output = self.fc5(x)

        return output

class PVCNNEncoder(nn.Module):
    def __init__(self, channel_num, latent_dim, num_points):
        super(PVCNNEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(channel_num, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(num_points, 1024),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )
        self.fcc = nn.Linear(256, 1)
        self.enc_mu = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Softsign()
        )
        self.enc_var = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Softsign()
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.conv2(x)
        x = self.fc2(x)
        x = self.conv3(x)
        x = self.fc3(x)
        x = self.fcc(x)
        x = torch.squeeze(x)

        return self.enc_mu(x), self.enc_var(x)

class Hand_decoder(nn.Module):
    def __init__(self, channel_num, latent_dim, num_points):
        super(Hand_decoder, self).__init__()
        self.dec_bn1 = nn.BatchNorm1d(latent_dim)
        self.dec_rb1 = ResBlock(latent_dim, latent_dim*2)
        self.dec_rb2 = ResBlock(latent_dim*2, latent_dim*4)
        self.dec_pose = nn.Sequential(
            nn.Linear(latent_dim*4, 48),

        )
        self.dec_shape = nn.Sequential(
            nn.Linear(latent_dim*4, 10),

        )
        self.dec_contact = nn.Linear(latent_dim * 4, 778)
        self.num_points = num_points
        self.channel_num = channel_num
        self.mano_layer = ManoLayer(center_idx=0, mano_assets_root='mano')

    def forward(self, x):
        x = self.dec_bn1(x)
        x = self.dec_rb1(x)
        x = self.dec_rb2(x)
        pose = self.dec_pose(x)
        shape = self.dec_shape(x)
        contact = self.dec_contact(x)

        mano_output: MANOOutput = self.mano_layer(pose, shape)


        return mano_output.verts, contact

class HandAutoencoder(nn.Module):
    def __init__(self, channel_num, latent_dim, num_points):
        super(HandAutoencoder, self).__init__()
        self.encoder = PVCNNEncoder(channel_num, latent_dim, num_points)
        self.decoder = Hand_decoder(channel_num, latent_dim, num_points)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        mean, logvar = self.encoder(x)
        hand_verts, contact = self.decoder(mean)
        contact = torch.unsqueeze(contact, dim=2)
        x_recon = torch.cat([hand_verts, contact], dim=2)
        return x_recon, mean, logvar

class PVCNNEncoder(nn.Module):
    def __init__(self, channel_num, latent_dim, num_points):
        super(PVCNNEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(channel_num, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(num_points, 1024),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )

        self.fcc = nn.Linear(256, 1)

        self.enc_mu = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Softsign()
        )
        self.enc_var = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Softsign()
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.conv2(x)
        x = self.fc2(x)
        x = self.conv3(x)
        x = self.fc3(x)
        #x = self.bn(x)
        x = self.fcc(x)
        #x = self.ll(x)
        x = torch.squeeze(x)


        #return torch.distributions.normal.Normal(self.enc_mu(x), F.softplus(self.enc_var(x)))

        return self.enc_mu(x), self.enc_var(x)

class PVCNNDecoder(nn.Module):
    def __init__(self, channel_num, latent_dim, num_points):
        super(PVCNNDecoder, self).__init__()
        self.dec_bn1 = nn.BatchNorm1d(latent_dim)
        self.dec_rb1 = ResBlock(latent_dim, latent_dim*2)
        self.dec_rb2 = ResBlock(latent_dim*2, latent_dim*4)
        self.dec_pose = nn.Linear(latent_dim*4, num_points*channel_num)
        self.num_points = num_points
        self.channel_num = channel_num
    def forward(self, x):
        x = self.dec_bn1(x)
        x = self.dec_rb1(x)
        x = self.dec_rb2(x)
        x = self.dec_pose(x)
        x = x.view(-1, self.channel_num, self.num_points)
        return x

class PVCNNAutoencoder(nn.Module):
    def __init__(self, channel_num, latent_dim, num_points):
        super(PVCNNAutoencoder, self).__init__()
        self.encoder = PVCNNEncoder(channel_num, latent_dim, num_points)
        self.decoder = PVCNNDecoder(channel_num, latent_dim, num_points)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        mean, logvar = self.encoder(x)

        x_recon = self.decoder(mean)
        x_recon = torch.transpose(x_recon, 1, 2)

        return x_recon, mean, logvar



class Latent_diffusion_fx(nn.Module):
    def __init__(self, latent_dim):
        super(Latent_diffusion_fx, self).__init__()

        self.diffusion_model = DiffusionModel_fx(latent_dim)

    def forward(self, latent, cond_feature):

        loss, x, target, model_out, unreduced_loss = self.diffusion_model(latent, cond_feature)

        return model_out, loss

