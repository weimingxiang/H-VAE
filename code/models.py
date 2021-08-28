import os
import math
from pytorch_lightning.core.hooks import CheckpointHooks
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from ray import tune
import numpy as np

from sklearn.model_selection import StratifiedKFold
import torch.distributions.multivariate_normal as mn
from torch.utils.data import DataLoader, WeightedRandomSampler

import mutils
# from utils import load_class, GetKS, TrainDataset, TestDataset

isA = False
data_type = 3
if isA == False:
    ldir = '~/result/ray_results_' + str(data_type)
else:
    ldir = '~/result/ray_results_' + str(data_type) + "A"


if data_type == 0:
    checkpoint = "/home/xwm/crisper/gan_code/checkpoints0/epoch=12-step=59409.ckpt"
elif data_type == 1:
    checkpoint = "/home/xwm/crisper/gan_code/checkpoints1/epoch=16-step=7802.ckpt"
elif data_type == 2:
    checkpoint = "/home/xwm/crisper/gan_code/checkpoints2/epoch=14-step=8504.ckpt"
elif data_type == 3:
    checkpoint = "/home/xwm/crisper/gan_code/checkpoints3/epoch=13-step=7461.ckpt"


class Decoder(nn.Module):
    def __init__(self, z_dim, c_dim, gf_dim):
        super(Decoder, self).__init__()

        if data_type == 0:
            self.convTrans0 = nn.ConvTranspose2d(
                z_dim, gf_dim*8, (12, 8), 1, 0, bias=False)
        else:
            self.convTrans0 = nn.ConvTranspose2d(
                z_dim, gf_dim*8, (11, 8), 1, 0, bias=False)
        self.bn0 = nn.BatchNorm2d(gf_dim*8)
        self.relu0 = nn.ReLU(inplace=True)

        self.convTrans1 = nn.ConvTranspose2d(
            gf_dim*8, gf_dim*4, 4, (1, 2), (0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(gf_dim*4)
        self.relu1 = nn.ReLU(inplace=True)

        self.convTrans2 = nn.ConvTranspose2d(
            gf_dim*4, gf_dim*2, 4, (1, 2), (0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(gf_dim*2)
        self.relu2 = nn.ReLU(inplace=True)

        self.convTrans3 = nn.ConvTranspose2d(
            gf_dim*2, gf_dim, 4, (1, 2), (0, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(gf_dim)
        self.relu3 = nn.ReLU(inplace=True)

        self.convTrans4 = nn.ConvTranspose2d(
            gf_dim, c_dim, 4, (1, 2), (0, 1), bias=False)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, z):
        h0 = self.relu0(self.bn0(self.convTrans0(z)))
        h1 = self.relu1(self.bn1(self.convTrans1(h0)))
        h2 = self.relu2(self.bn2(self.convTrans2(h1)))
        h3 = self.relu3(self.bn3(self.convTrans3(h2)))
        h4 = self.convTrans4(h3)
        output = self.tanh(h4)
        return output  # (c_dim, 64, 64)

# Discriminator


class Encoder(nn.Module):
    def __init__(self, z_dim, c_dim, df_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.df_dim = df_dim

        self.embedding = nn.Embedding(25, hidden_dim)
        self.conv0 = nn.Conv2d(c_dim, df_dim, 4, (1, 2), (0, 1), bias=False)
        self.relu0 = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(df_dim, df_dim*2, 4, (1, 2), (0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(df_dim*2)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(df_dim*2, df_dim*4, 4,
                               (1, 2), (0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(df_dim*4)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(df_dim*4, df_dim*8, 4,
                               (1, 2), (0, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(df_dim*8)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        if data_type == 0:
            self.fc_z1 = nn.Linear(df_dim*8*12*8, z_dim)
            self.fc_z2 = nn.Linear(df_dim*8*12*8, z_dim)
        else:
            self.fc_z1 = nn.Linear(df_dim*8*11*8, z_dim)
            self.fc_z2 = nn.Linear(df_dim*8*11*8, z_dim)

        # self.conv4 = nn.Conv2d(df_dim*8, 1, 4, 1, 0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        input_e = self.embedding(input).unsqueeze(1)  # b, 1, 25, 128
        h0 = self.relu0(self.conv0(input_e))  # b, df_dim, 22, 64
        h1 = self.relu1(self.bn1(self.conv1(h0)))
        h2 = self.relu2(self.bn2(self.conv2(h1)))
        h3 = self.relu3(self.bn3(self.conv3(h2)))
        if data_type == 0:
            mu = self.fc_z1(h3.view(-1, self.df_dim*8*12*8))  # (1, 128*8*4*4)
            sigma = self.fc_z2(h3.view(-1, self.df_dim*8*12*8))
        else:
            mu = self.fc_z1(h3.view(-1, self.df_dim*8*11*8))  # (1, 128*8*4*4)
            sigma = self.fc_z2(h3.view(-1, self.df_dim*8*11*8))
        return mu, sigma, input_e  # by squeeze, get just float not float Tenosor


class VAE(pl.LightningModule):

    def __init__(self, config, data_dir=None):
        super(VAE, self).__init__()

        self.data_dir = data_dir
        self.lr = config['lr']
        self.hidden_dim = config['hidden_dim']
        self.batch_size = config['batch_size']
        self.class_num = config['class_num']
        self.beta1 = config['beta1']
        self.z_dim = config['z_dim']
        self.c_dim = config['c_dim']
        self.df_dim = config['df_dim']

        self.encoder = Encoder(self.z_dim, self.c_dim,
                               self.df_dim, self.hidden_dim)
        self.decoder = Decoder(self.z_dim, self.c_dim, self.df_dim)

        self.criterion = nn.MSELoss(size_average=False)

        self.ini_z()

    def ini_z(self):
        self.Z = []
        for i in range(self.class_num):
            # Z : [class_num, z_dim]
            self.Z.append(torch.zeros((1, self.z_dim), dtype=torch.float))

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        ids1, ids2, ids3 = x
        mu, log_sigmoid, ie = self.encoder(ids1)

        # reparameterization
        std = torch.exp(log_sigmoid / 2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = z.view(-1, self.z_dim, 1, 1)

        # reconstruct image
        x_reconstruct = self.decoder(z)
        # reconstruct_loss + KL_divergence
        reconstruct_loss = self.criterion(x_reconstruct, ie)

        kl_div = -0.5 * torch.sum(1+log_sigmoid-mu.pow(2)-log_sigmoid.exp())
        loss = reconstruct_loss + kl_div
        # loss = self.criterion(x_reconstruct, ie)

        # Manually back step
        opt_e, opt_d = self.optimizers()
        self.manual_backward(loss)
        opt_e.step()
        opt_d.step()

        self.log("ae/train_loss", loss, on_step=True,
                 on_epoch=True, logger=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, label = batch
        ids1, ids2, ids3 = x
        mu, log_sigmoid, ie = self.encoder(ids1)
        std = torch.exp(log_sigmoid/2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = z.view(-1, 1, self.z_dim)
        self.Z = mutils.batch2one(self.Z, label, z, self.class_num)

    def test_epoch_end(self, output):
        N = []
        for i in range(self.class_num):
            label_mean = torch.mean(self.Z[i][1:], dim=0).float()
            label_cov = torch.from_numpy(
                np.cov(self.Z[i][1:].numpy(), rowvar=False)).float()
            m = mn.MultivariateNormal(label_mean, label_cov)
            N.append(m)
        torch.save({'distribution': N}, os.path.join(
            self.data_dir, 'gan_code/class_distribution') + str(data_type) + '.dt')

    def prepare_data(self):

        if data_type == 1:
            X_ids, X_items, labels = mutils.load_class(self.data_dir)

            train_items, test_items = mutils.Dataset_split_senerio1(
                X_items, 0.2, 2021)
            train_labels, test_labels = mutils.Dataset_split_senerio1(
                labels, 0.2, 2021)

        elif data_type == 2:

            X_ids, X_items, labels = mutils.load_class(self.data_dir)

            train_items, train_labels, test_items, test_labels = mutils.Dataset_split_senerio2(
                X_ids, X_items, labels, 4)

        elif data_type == 3:

            X_ids, X_items, labels = mutils.load_class(self.data_dir)

            train_items, train_labels, test_items, test_labels = mutils.Dataset_split_senerio3(
                X_ids, X_items, labels, 2021)

        elif data_type == 0:
            X_ids, train_items, train_labels, sgrna_nums = mutils.load_datasetI1(
                self.data_dir)
            test_items, test_labels = mutils.load_datasetI2(self.data_dir)

        if data_type != 0:
            self.train_dataset = mutils.TrainDataset(train_items, train_labels)
            self.test_dataset = mutils.TestDataset(test_items, test_labels)
        else:
            self.train_dataset = mutils.TrainDataset_indels(
                train_items, train_labels)
            self.test_dataset = mutils.TestDataset_indels(
                test_items, test_labels)

        # WeightSample
        weights = np.array([len(self.train_dataset.neg_index),
                            len(self.train_dataset.pos_index)])
        weights = 1.0 / weights
        sample_weights = []
        for label in train_labels:
            sample_weights.append(weights[int(label)])
        self.wsampler = WeightedRandomSampler(
            sample_weights, len(sample_weights))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=2, sampler=self.wsampler)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=2)

    def test_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=2)

    @property
    def automatic_optimization(self):
        return False

    def configure_optimizers(self):
        opt_e = torch.optim.Adam(
            self.encoder.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        opt_d = torch.optim.Adam(
            self.decoder.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        return [opt_e, opt_d]


class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, gf_dim):
        super(Generator, self).__init__()
        if data_type == 0:
            self.convTrans0 = nn.ConvTranspose2d(
                z_dim, gf_dim*8, (12, 8), 1, 0, bias=False)
        else:
            self.convTrans0 = nn.ConvTranspose2d(
                z_dim, gf_dim*8, (11, 8), 1, 0, bias=False)
        self.bn0 = nn.BatchNorm2d(gf_dim*8)
        self.relu0 = nn.ReLU(inplace=True)

        self.convTrans1 = nn.ConvTranspose2d(
            gf_dim*8, gf_dim*4, 4, (1, 2), (0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(gf_dim*4)
        self.relu1 = nn.ReLU(inplace=True)

        self.convTrans2 = nn.ConvTranspose2d(
            gf_dim*4, gf_dim*2, 4, (1, 2), (0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(gf_dim*2)
        self.relu2 = nn.ReLU(inplace=True)

        self.convTrans3 = nn.ConvTranspose2d(
            gf_dim*2, gf_dim, 4, (1, 2), (0, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(gf_dim)
        self.relu3 = nn.ReLU(inplace=True)

        self.convTrans4 = nn.ConvTranspose2d(
            gf_dim, c_dim, 4, (1, 2), (0, 1), bias=False)
        self.tanh = nn.Tanh()

    def forward(self, z):
        h0 = self.relu0(self.bn0(self.convTrans0(z)))
        h1 = self.relu1(self.bn1(self.convTrans1(h0)))
        h2 = self.relu2(self.bn2(self.convTrans2(h1)))
        h3 = self.relu3(self.bn3(self.convTrans3(h2)))
        h4 = self.convTrans4(h3)
        output = self.tanh(h4)
        return output


class Discriminator(nn.Module):
    def __init__(self, z_dim, c_dim, df_dim, class_num, hidden_dim):
        super(Discriminator, self).__init__()
        self.df_dim = df_dim

        self.embedding = nn.Embedding(25, hidden_dim)
        self.conv0 = nn.Conv2d(c_dim, df_dim, 4, (1, 2), (0, 1), bias=False)
        self.relu0 = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(df_dim, df_dim*2, 4, (1, 2), (0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(df_dim*2)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(df_dim*2, df_dim*4, 4,
                               (1, 2), (0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(df_dim*4)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(df_dim*4, df_dim*8, 4,
                               (1, 2), (0, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(df_dim*8)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        if data_type == 0:
            self.fc_aux = nn.Linear(df_dim*8*12*8, class_num)
            self.fc_dis = nn.Linear(df_dim*8*12*8, 1)
        else:
            self.fc_aux = nn.Linear(df_dim*8*11*8, class_num)
            self.fc_dis = nn.Linear(df_dim*8*11*8, 1)
        self.softmax = nn.LogSoftmax()
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, use_e):
        input_e = None
        if use_e:
            input = self.embedding(input).unsqueeze(1)
        h0 = self.relu0(self.conv0(input))
        h1 = self.relu1(self.bn1(self.conv1(h0)))
        h2 = self.relu2(self.bn2(self.conv2(h1)))
        h3 = self.relu3(self.bn3(self.conv3(h2)))
        # cl = self.class_logistics(h3.view(-1, self.df_dim*8*4*4))
        # gl = self.gan_logistics(cl)
        if data_type == 0:
            output = self.fc_aux(h3.view(-1, self.df_dim*8*12*8))
            realfake = self.sigmoid(self.fc_dis(
                h3.view(-1, self.df_dim*8*12*8))).view(-1, 1).squeeze(1)
        else:
            output = self.fc_aux(h3.view(-1, self.df_dim*8*11*8))
            realfake = self.sigmoid(self.fc_dis(
                h3.view(-1, self.df_dim*8*11*8))).view(-1, 1).squeeze(1)
        return h3, output, realfake


class BGAN(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super(BGAN, self).__init__()

        self.data_dir = data_dir
        self.lr_ratio = config['lr_ratio']
        self.lambda1 = config['lambda1']
        self.lambda2 = config['lambda2']
        self.batch_size = config['batch_size']
        self.hidden_dim = config['hidden_dim']
        self.class_num = config['class_num']
        self.z_dim = config['z_dim']
        self.c_dim = config['c_dim']
        self.df_dim = config['df_dim']

        self.lr = config['lr']
        self.beta1 = config['beta1']

        self.generator = Generator(self.z_dim, self.c_dim, self.df_dim)
        self.discriminator = Discriminator(
            self.z_dim, self.c_dim, self.df_dim, self.class_num, self.hidden_dim)

        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = mutils.FocalLoss(gamma=2)
        self.dis_criterion = nn.BCELoss()
        self.init_weight()

    def init_weight(self):
        weights = torch.load(
            checkpoint)['state_dict']
        state_e = {e_w.replace(
            'encoder.', ''): weights[e_w] for e_w in weights.keys() if 'encoder' in e_w}
        state_d = {d_w.replace(
            'decoder.', ''): weights[d_w] for d_w in weights.keys() if 'decoder' in d_w}

        self.generator.load_state_dict(state_d)
        del state_e['fc_z1.weight']
        del state_e['fc_z1.bias']
        del state_e['fc_z2.weight']
        del state_e['fc_z2.bias']
        state_e.update(
            {'fc_aux.weight': self.discriminator.state_dict()['fc_aux.weight']})
        state_e.update(
            {'fc_aux.bias': self.discriminator.state_dict()['fc_aux.bias']})
        state_e.update(
            {'fc_dis.weight': self.discriminator.state_dict()['fc_dis.weight']})
        state_e.update(
            {'fc_dis.bias': self.discriminator.state_dict()['fc_dis.bias']})

        # self.discriminator.load_state_dict(state_e)

        self.distribution = torch.load(
            "/home/xwm/crisper/gan_code/class_distribution" + str(data_type) + ".dt")['distribution']

    def conditional_latent_generator(self, batch, target_distribution="uniform"):
        distribution1 = self.class_uratio
        if target_distribution == "d":
            distribution1 = self.class_dratio
        elif target_distribution == "g":
            distribution1 = self.class_gratio

        sampled_labels = []
        sampled_labels_p = np.random.uniform(0, 1, batch)

        samples = []
        for c in list(range(self.class_num)):
            mask = np.logical_and((sampled_labels_p > 0),
                                  (sampled_labels_p <= distribution1[c]))
            sampled_labels_p = sampled_labels_p - distribution1[c]

            samples.append(self.distribution[c].sample(
                (np.sum(mask == True),)))
            sampled_labels.extend(np.full(np.sum(mask == True), c))

        fake_z = torch.cat(samples, dim=0)

        return fake_z.cuda(), torch.tensor(sampled_labels, dtype=torch.long)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        ids1, ids2, ids3 = x
        batch_size = ids1.size(0)
        fake_num = math.ceil(batch_size / self.lambda2)

        opt_g, opt_d = self.optimizers()

        conditional_z, z_label = self.conditional_latent_generator(
            fake_num, target_distribution='d')

        real_label = torch.LongTensor(batch_size).cuda()
        aux_label = torch.FloatTensor(batch_size).cuda()
        fake_label = torch.LongTensor(fake_num).cuda()

        label = y.long().squeeze()

        sample_features, D_real, D_rf = self.discriminator(ids1, 1)
        real_label.resize_(batch_size).copy_(label)
        aux_label.data.resize_(batch_size).fill_(1)

        D_loss_real = self.criterion(D_real, real_label)
        D_loss_rf = self.dis_criterion(D_rf, aux_label)
        if isA == False:
            errD_real = D_loss_real  # + D_loss_rf
        else:
            errD_real = D_loss_real
        self.manual_backward(errD_real)
        if isA == False:
            aux_label = torch.FloatTensor(fake_num).cuda()
            noise = conditional_z[0:fake_num].view(-1, self.z_dim, 1, 1)
            fake_label.resize_(noise.shape[0]).copy_(z_label)
            aux_label.data.resize_(fake_num).fill_(0)
            fake = self.generator(noise)
            _, D_fake, D_f_rf = self.discriminator(fake.detach(), 0)
            D_loss_fake = self.criterion(D_fake, fake_label)
            D_loss_frf = self.dis_criterion(D_f_rf, aux_label)
            # errD_fake = D_loss_fake + D_loss_frf
            errD_fake = D_loss_fake
            self.manual_backward(errD_fake)
            D_loss = errD_real + errD_fake
        opt_d.step()
        opt_d.zero_grad()

        # update G
        if isA == False:
            conditional_z, z_label = self.conditional_latent_generator(
                fake_num, target_distribution='d')
            noise = conditional_z[0:fake_num].view(-1, self.z_dim, 1, 1)
            fake_label.resize_(noise.shape[0]).copy_(z_label)
            fake = self.generator(noise)
            # z_label = torch.LongTensor(z_label).cuda()
            # noise = conditional_z.view(-1, self.z_dim, 1, 1)
            # fake = self.generator(noise)
            # _, D_fake, _ = self.discriminator(fake, 0)
            # G_loss = self.criterion(D_fake, z_label)
            # self.manual_backward(G_loss)
            # opt_g.step()
            aux_label.data.resize_(fake_num).fill_(1)
            _, G_fake, G_f_rf = self.discriminator(fake, 0)
            G_loss_fake = self.criterion(G_fake, fake_label)
            G_loss_frf = self.dis_criterion(G_f_rf, aux_label)
            errG_fake = G_loss_fake
            # errG_fake = G_loss_fake + G_loss_frf
            self.manual_backward(errG_fake)
            opt_g.step()
            opt_g.zero_grad()

        self.log('train/d_loss_real', errD_real,
                 on_epoch=True, logger=True, prog_bar=True)
        if isA == False:
            self.log('train/d_loss_fake', errD_fake,
                     on_epoch=True, logger=True, prog_bar=True)
            self.log('train/g_loss', errG_fake, on_epoch=True,
                     logger=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        ids1, ids2, ids3 = x
        _, logits, _ = self.discriminator(ids1, 1)
        # logits = output[:,1]

        return {"logits": logits, 'y': y.long()}

    def validation_epoch_end(self, outputs):
        logits = torch.cat([torch.softmax(x["logits"], dim=-1)
                            for x in outputs])
        logits = logits[:, 1]
        ys = torch.cat([x["y"] for x in outputs])
        roc_value, prc_value, ks = mutils.GetKS(
            ys.data.cpu().numpy(), logits.data.cpu().numpy())
        self.log("ptl_roc_value", roc_value)
        self.log("ptl_prc_value", prc_value)
        print(roc_value)
        print(prc_value)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(
        ), lr=self.lr*self.lr_ratio, weight_decay=0.01)
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, weight_decay=0.01)

        return [opt_g, opt_d]

    def prepare_data(self):
        if data_type == 1:
            X_ids, X_items, labels = mutils.load_class(self.data_dir)

            train_items, test_items = mutils.Dataset_split_senerio1(
                X_items, 0.2, 2021)
            train_labels, test_labels = mutils.Dataset_split_senerio1(
                labels, 0.2, 2021)

        elif data_type == 2:
            X_ids, X_items, labels = mutils.load_class(self.data_dir)

            train_items, train_labels, test_items, test_labels = mutils.Dataset_split_senerio2(
                X_ids, X_items, labels, 3)

        elif data_type == 3:
            X_ids, X_items, labels = mutils.load_class(self.data_dir)

            train_items, train_labels, test_items, test_labels = mutils.Dataset_split_senerio3(
                X_ids, X_items, labels, 2021)

        elif data_type == 0:
            X_ids, train_items, train_labels, sgrna_nums = mutils.load_datasetI1(
                self.data_dir)
            test_items, test_labels = mutils.load_datasetI2(self.data_dir)

        if data_type != 0:
            self.train_dataset = mutils.TrainDataset(train_items, train_labels)
            self.test_dataset = mutils.TestDataset(test_items, test_labels)
        else:
            self.train_dataset = mutils.TrainDataset_indels(
                train_items, train_labels)
            self.test_dataset = mutils.TestDataset_indels(
                test_items, test_labels)

        # dataset statistic
        self.per_class_count = [
            len(self.train_dataset.neg_index), len(self.train_dataset.pos_index)]
        self.class_aratio = [
            class_count/sum(self.per_class_count) for class_count in self.per_class_count]

        self._set_class_ratios()

        # WeightSample
        weights = np.array([len(self.train_dataset.neg_index), len(
            self.train_dataset.pos_index) * self.lambda1])
        weights = 1.0 / weights
        sample_weights = []
        for label in train_labels:
            sample_weights.append(weights[int(label)])
        self.wsampler = WeightedRandomSampler(
            sample_weights, len(sample_weights))

    def _set_class_ratios(self):
        target = 1 / self.class_num
        self.class_uratio = np.full(self.class_num, target)

        # set gratio
        self.class_gratio = np.full(self.class_num, 0.0)
        for c in range(self.class_num):
            self.class_gratio[c] = 2 * target - self.class_aratio[c]

        # set dratio
        self.class_dratio = np.full(self.class_num, 0.0)
        for c in range(self.class_num):
            self.class_dratio[c] = 2 * target - self.class_aratio[c]

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=2, sampler=self.wsampler)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=2)

    @property
    def automatic_optimization(self):
        return False


class MdCrispr(pl.LightningModule):

    def __init__(self, config, data_dir=None):
        super(MdCrispr, self).__init__()

        self.data_dir = data_dir

        self.lr = config["lr"]
        self.hidden_dim = config['hidden_dim']
        self.kernel_size = config['kernel_size']
        self.dilation_size = config['dilation_size']
        self.out_channels = config['out_channels']
        self.dropout_size = config['dropout_size']
        self.batch_size = config['batch_size']
        # self.data_index = config[tune.suggest.repeater.TRIAL_INDEX]

        self.embedding = nn.Embedding(16, self.hidden_dim)
        self.embedding1 = nn.Embedding(4, self.hidden_dim)
        self.embedding2 = nn.Embedding(4, self.hidden_dim)

        self.conv1 = nn.Conv2d(3, self.out_channels, kernel_size=(
            self.kernel_size, 23), dilation=(self.dilation_size, 1))
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout(self.dropout_size)
        self.non_linear = nn.ReLU()

        self.linear1 = nn.Linear((self.hidden_dim - self.dilation_size *
                                  (self.kernel_size - 1)) * self.out_channels, 2, bias=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        ids1, ids2, ids3 = x

        output = self.embedding(ids1)
        emb_q = self.embedding1(ids2)
        emb_a = self.embedding2(ids3)

        conv_input = torch.stack([output, emb_q, emb_a], dim=1)
        conv_input = conv_input.transpose(2, 3)
        out_conv = self.conv1(conv_input)
        out_conv = self.bn1(out_conv)
        out_conv = self.non_linear(out_conv)
        out_conv = out_conv.view(-1, (self.hidden_dim - (self.kernel_size - 1)
                                      * self.dilation_size) * self.out_channels)
        input_fc = self.dropout(out_conv)
        score = self.linear1(input_fc)

        return score

    def cross_entropy_loss(self, logits, labels):
        return self.criterion(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y.long())

        self.log("ptl/train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y.long())

        return {"val_loss": loss, "logits": logits, 'y': y.long()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logits = torch.cat([x["logits"][:, 1] for x in outputs])
        ys = torch.cat([x["y"] for x in outputs])
        roc_value, prc_value, ks = mutils.GetKS(
            ys.data.cpu().numpy(), logits.data.cpu().numpy())
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/roc_value", roc_value)
        self.log("ptl/prc_value", prc_value)
        print(roc_value)
        print(prc_value)

    def prepare_data(self):
        X_ids, X_items, labels = mutils.load_class(self.data_dir)

        train_items, test_items = mutils.Dataset_split_senerio1(
            X_items, 0.2, 2021)
        train_labels, test_labels = mutils.Dataset_split_senerio1(
            labels, 0.2, 2021)
        # train_items, train_labels, test_items, test_labels = mutils.Dataset_split_senerio2(X_ids, X_items, labels, 3)

        self.train_dataset = mutils.TrainDataset(train_items, train_labels)
        self.test_dataset = mutils.TestDataset(test_items, test_labels)

        # WeightSample
        weights = np.array([len(self.train_dataset.neg_index),
                            len(self.train_dataset.pos_index)])
        weights = 1.0 / weights
        sample_weights = []
        for label in train_labels:
            sample_weights.append(weights[int(label)])
        self.wsampler = WeightedRandomSampler(
            sample_weights, len(sample_weights))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=self.wsampler, pin_memory=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class CnnCrispr(pl.LightningModule):

    def __init__(self, config, data_dir=None):
        super(CnnCrispr, self).__init__()

        self.data_dir = data_dir

        self.lr = config["lr"]
        self.batch_size = config['batch_size']
        # self.data_index = config[tune.suggest.repeater.TRIAL_INDEX]

        self.embedding = nn.Embedding(16, 100)
        self.embedding.weight.requires_grad = True
        self.bilstm = nn.LSTM(100, 40, bidirectional=True)

        self.conv1 = nn.Sequential(nn.Conv1d(80, 10, 5),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(10))
        self.conv2 = nn.Sequential(nn.Conv1d(10, 20, 5),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(20))
        self.conv3 = nn.Sequential(nn.Conv1d(20, 40, 5),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(40))
        self.conv4 = nn.Sequential(nn.Conv1d(40, 80, 5),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(80))
        self.conv5 = nn.Sequential(nn.Conv1d(80, 100, 5),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(100),
                                   nn.Flatten())

        self.linear1 = nn.Linear(300, 20)
        self.linear2 = nn.Linear(20, 2)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        ids1, ids2, ids3 = x

        output = self.embedding(ids1)

        output, (_, _) = self.bilstm(output)
        output = nn.functional.relu(output)
        output = output.permute(0, 2, 1)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)

        output = nn.functional.dropout(output, 0.3)
        output = nn.functional.relu(self.linear1(output))
        output = self.linear2(output)

        return output

    def cross_entropy_loss(self, logits, labels):
        return self.criterion(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y.long())

        self.log("ptl/train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y.long())

        return {"val_loss": loss, "logits": logits, 'y': y.long()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logits = torch.cat([x["logits"][:, 1] for x in outputs])
        ys = torch.cat([x["y"] for x in outputs])
        roc_value, prc_value, ks = mutils.GetKS(
            ys.data.cpu().numpy(), logits.data.cpu().numpy())
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/roc_value", roc_value)
        self.log("ptl/prc_value", prc_value)
        print(roc_value)
        print(prc_value)

    def prepare_data(self):
        X_ids, X_items, labels = mutils.load_class(self.data_dir)

        # train_items, test_items = mutils.Dataset_split_senerio1(X_items, 0.2, 2021)
        # train_labels, test_labels = mutils.Dataset_split_senerio1(labels, 0.2, 2021)
        train_items, train_labels, test_items, test_labels = mutils.Dataset_split_senerio2(
            X_ids, X_items, labels, 3)

        self.train_dataset = mutils.TrainDataset(train_items, train_labels)
        self.test_dataset = mutils.TestDataset(test_items, test_labels)

        # WeightSample
        weights = np.array([len(self.train_dataset.neg_index),
                            len(self.train_dataset.pos_index)])
        weights = 1.0 / weights
        sample_weights = []
        for label in train_labels:
            sample_weights.append(weights[int(label)])
        self.wsampler = WeightedRandomSampler(
            sample_weights, len(sample_weights))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=self.wsampler, pin_memory=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class MCrispr(pl.LightningModule):

    def __init__(self, config, data_dir=None):
        super(MCrispr, self).__init__()

        self.data_dir = data_dir

        self.lr = config["lr"]
        self.hidden_dim = config['hidden_dim']
        self.kernel_size = config['kernel_size']
        self.dilation_size = config['dilation_size']
        self.out_channels = config['out_channels']
        self.dropout_size = config['dropout_size']
        self.batch_size = config['batch_size']
        # self.data_index = config[tune.suggest.repeater.TRIAL_INDEX]

        self.embedding = nn.Embedding(16, self.hidden_dim)
        # self.embedding1 = nn.Embedding(4, self.hidden_dim)
        # self.embedding2 = nn.Embedding(4, self.hidden_dim)

        self.conv1 = nn.Conv2d(1, self.out_channels, kernel_size=(
            self.kernel_size, 23), dilation=(self.dilation_size, 1))
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout(self.dropout_size)
        self.non_linear = nn.ReLU()

        self.linear1 = nn.Linear((self.hidden_dim - self.dilation_size *
                                  (self.kernel_size - 1)) * self.out_channels, 2, bias=False)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = mutils.FocalLoss(gamma=config['gamma'], alpha=config['focal_rate'])

    def forward(self, x):
        ids1, ids2, ids3 = x

        output = self.embedding(ids1)
        # emb_q = self.embedding1(ids2)
        # emb_a = self.embedding1(ids3)

        conv_input = torch.stack([output], dim=1)
        conv_input = conv_input.transpose(2, 3)
        out_conv = self.conv1(conv_input)
        out_conv = self.bn1(out_conv)
        out_conv = self.non_linear(out_conv)
        out_conv = out_conv.view(-1, (self.hidden_dim - (self.kernel_size - 1)
                                      * self.dilation_size) * self.out_channels)
        input_fc = self.dropout(out_conv)
        score = self.linear1(input_fc)
        # score = F.softmax(score, dim=-1)

        return score

    def cross_entropy_loss(self, logits, labels):
        return self.criterion(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y.long())
        # print(loss.item())
        self.log("ptl/train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y.long())

        return {"val_loss": loss, "logits": logits, 'y': y.long()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logits = torch.cat([x["logits"][:, 1] for x in outputs])
        ys = torch.cat([x["y"] for x in outputs])
        roc_value, prc_value, ks = mutils.GetKS(
            ys.data.cpu().numpy(), logits.data.cpu().numpy())
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/roc_value", roc_value)
        self.log("ptl/prc_value", prc_value)
        # print(logits)
        print(roc_value)
        print(prc_value)

    def prepare_data(self):
        X_ids, X_items, labels = mutils.load_class(self.data_dir)

        # five_folds = StratifiedKFold(n_splits=5, shuffle=False)
        # for i, (train_idx1, test_idx1) in enumerate(five_folds.split(X_items, labels)):
        #     if i == self.data_index:
        #         train_idx, test_idx = train_idx1, test_idx1
        train_items, test_items = mutils.Dataset_split_senerio1(
            X_items, 0.2, 2021)
        train_labels, test_labels = mutils.Dataset_split_senerio1(
            labels, 0.2, 2021)
        # train_items, train_labels, test_items, test_labels = mutils.Dataset_split_senerio2(X_ids, X_items, labels, 2)
        # train_ids, train_items, train_labels = X_ids[train_idx], X_items[train_idx], labels[train_idx]
        # test_ids, test_items, test_labels = X_ids[test_idx], X_items[test_idx], labels[test_idx]

        self.train_dataset = mutils.TrainDataset(train_items, train_labels)
        self.test_dataset = mutils.TestDataset(test_items, test_labels)

        # WeightSample
        weights = np.array([len(self.train_dataset.neg_index),
                            len(self.train_dataset.pos_index)])
        weights = 1.0 / weights
        sample_weights = []
        for label in train_labels:
            sample_weights.append(weights[int(label)])
        self.wsampler = WeightedRandomSampler(
            sample_weights, len(sample_weights))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=self.wsampler, num_workers=1)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, num_workers=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class MTCrispr(pl.LightningModule):

    def __init__(self, config, data_dir=None):
        super(MTCrispr, self).__init__()

        self.data_dir = data_dir

        self.lr = config["lr"]
        self.hidden_dim = config['hidden_dim']
        self.kernel_size = config['kernel_size']
        self.dilation_size = config['dilation_size']
        self.out_channels = config['out_channels']
        self.dropout_size = config['dropout_size']
        self.batch_size = config['batch_size']
        self.att_head = config['attention_head']

        self.embedding = TransEmbeddings(emb_size=self.hidden_dim)

        self.conv1 = nn.Conv2d(1, self.out_channels, kernel_size=(
            self.kernel_size, 23), dilation=(self.dilation_size, 1))
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout(self.dropout_size)
        self.non_linear = nn.ReLU()

        self.linear1 = nn.Linear((self.hidden_dim - self.dilation_size *
                                  (self.kernel_size - 1)) * self.out_channels, 2, bias=False)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = mutils.FocalLoss(gamma=config['gamma'], alpha=config['focal_rate'])

    def forward(self, x):
        ids1, ids2, ids3 = x

        output = self.embedding(ids1)
        conv_input = torch.stack([output], dim=1)
        conv_input = conv_input.transpose(2, 3)
        out_conv = self.conv1(conv_input)
        out_conv = self.bn1(out_conv)
        out_conv = self.non_linear(out_conv)
        out_conv = out_conv.view(-1, (self.hidden_dim - (self.kernel_size - 1)
                                      * self.dilation_size) * self.out_channels)
        input_fc = self.dropout(out_conv)
        score = self.linear1(input_fc)
        # score = F.softmax(score, dim=-1)

        return score

    def cross_entropy_loss(self, logits, labels):
        return self.criterion(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y.long())

        self.log("ptl/train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y.long())

        return {"val_loss": loss, "logits": logits, 'y': y.long()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logits = torch.cat([x["logits"][:, 1] for x in outputs])
        ys = torch.cat([x["y"] for x in outputs])
        roc_value, prc_value, ks = mutils.GetKS(
            ys.data.cpu().numpy(), logits.data.cpu().numpy())
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/roc_value", roc_value)
        self.log("ptl/prc_value", prc_value)
        print(roc_value)
        print(prc_value)

    def prepare_data(self):
        X_ids, X_items, labels = mutils.load_class(self.data_dir)

        # train_items, test_items = mutils.Dataset_split_senerio1(X_items, 0.2, 2021)
        # train_labels, test_labels = mutils.Dataset_split_senerio1(labels, 0.2, 2021)
        train_items, train_labels, test_items, test_labels = mutils.Dataset_split_senerio2(
            X_ids, X_items, labels, 3)

        self.train_dataset = mutils.TrainDataset(train_items, train_labels)
        self.test_dataset = mutils.TestDataset(test_items, test_labels)

        # WeightSample
        weights = np.array([len(self.train_dataset.neg_index),
                            len(self.train_dataset.pos_index)])
        weights = 1.0 / weights
        sample_weights = []
        for label in train_labels:
            sample_weights.append(weights[int(label)])
        self.wsampler = WeightedRandomSampler(
            sample_weights, len(sample_weights))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=self.wsampler, pin_memory=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class TransLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(TransLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class TransEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, emb_size=256, vocab_size=16, seq_len=23):
        super(TransEmbeddings, self).__init__()
        vocab_size = vocab_size
        hidden_size = emb_size
        max_position_embeddings = seq_len
        hidden_dropout_prob = 0.2
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_size)
        self.word_embeddings.weight.requires_grad = True
        self.position_embeddings.weight.requires_grad = True

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = TransLayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # embeddings = words_embeddings
        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
