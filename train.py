from model import *
from dataset import *

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from statistics import mean


class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G
        self.lr_D = args.lr_D

        self.wgt_gan = args.wgt_gan
        self.wgt_disc = args.wgt_disc

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type
        self.norm = args.norm

        self.gpu_ids = args.gpu_ids

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.name_data = args.name_data

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

    def save(self, dir_chck, netG, netD, optimG, optimD, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                    'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    def load(self, dir_chck, netG, netD=[], optimG=[], optimD=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            netG.load_state_dict(dict_net['netG'])
            netD.load_state_dict(dict_net['netD'])
            optimG.load_state_dict(dict_net['optimG'])
            optimD.load_state_dict(dict_net['optimD'])

            return netG, netD, optimG, optimD, epoch

        elif mode == 'test':
            netG.load_state_dict(dict_net['netG'])

            return netG, epoch

    def preprocess(self, data):
        rescale = Rescale((self.ny_load, self.nx_load))
        randomcrop = RandomCrop((self.ny_out, self.nx_out))
        normalize = Normalize()
        randomflip = RandomFlip()
        totensor = ToTensor()
        # return totensor(randomcrop(rescale(randomflip(nomalize(data)))))
        return totensor(normalize(rescale(data)))

    def deprocess(self, data):
        tonumpy = ToNumpy()
        denomalize = Denomalize()
        return denomalize(tonumpy(data))


    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G
        lr_D = self.lr_D

        wgt_gan = self.wgt_gan
        wgt_disc = self.wgt_disc

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ny_in = self.ny_in
        nx_in = self.nx_in

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_data_train = os.path.join(self.dir_data, name_data, 'train')
        dir_log = os.path.join(self.dir_log, self.scope, name_data)

        transform_train = transforms.Compose([Normalize(), Rescale((self.ny_load, self.nx_load)), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        dataset_train = Dataset(dir_data_train, data_type=self.data_type, nch=self.nch_out, transform=transform_train)

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

        num_train = len(dataset_train)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

        ## setup network
        netG = DCGAN(nch_in, nch_out, nch_ker, norm)
        netD = Discriminator(nch_out, nch_ker, norm)

        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netD, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        fn_GAN = nn.BCELoss().to(device)
        fn_GP = GradientPaneltyLoss().to(device)

        paramsG = netG.parameters()
        paramsD = netD.parameters()

        optimG = torch.optim.Adam(paramsG, lr=lr_G, betas=(self.beta1, 0.999))
        optimD = torch.optim.Adam(paramsD, lr=lr_D, betas=(self.beta1, 0.999))

        # schedG = get_scheduler(optimG, self.opts)
        # schedD = get_scheduler(optimD, self.opts)

        # schedG = torch.optim.lr_scheduler.ExponentialLR(optimG, gamma=0.9)
        # schedD = torch.optim.lr_scheduler.ExponentialLR(optimD, gamma=0.9)

        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            netG, netD, optimG, optimD, st_epoch = self.load(dir_chck, netG, netD, optimG, optimD, mode=mode)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG.train()
            netD.train()

            loss_G_train = []
            loss_D_real_train = []
            loss_D_fake_train = []

            for i, data in enumerate(loader_train, 1):
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)

                input = torch.randn(batch_size, nch_in, ny_in, nx_in).to(device)
                label = data.to(device)

                # forward netG
                output = netG(input)

                # backward netD
                set_requires_grad(netD, True)
                optimD.zero_grad()

                pred_real = netD(label)
                pred_fake = netD(output.detach())

                alpha = torch.rand(label.size(0), 1, 1, 1).to(self.device)
                output_ = (alpha * label + (1 - alpha) * output.detach()).requires_grad_(True)
                src_out_ = netD(output_)

                # loss_D_real = fn_GAN(pred_real, torch.ones_like(pred_real))
                # loss_D_fake = fn_GAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D_real = -torch.mean(pred_real)
                loss_D_fake = torch.mean(pred_fake)

                loss_D_gp = fn_GP(src_out_, output_)

                loss_D = 0.5 * (loss_D_real + loss_D_fake) + loss_D_gp

                loss_D.backward()
                optimD.step()

                # backward netG
                set_requires_grad(netD, False)
                optimG.zero_grad()

                pred_fake = netD(output)

                # loss_G = fn_GAN(pred_fake, torch.ones_like(pred_fake))
                loss_G = -torch.mean(pred_fake)

                loss_G.backward()
                optimG.step()

                # get losses
                loss_G_train += [loss_G.item()]
                loss_D_real_train += [loss_D_real.item()]
                loss_D_fake_train += [loss_D_fake.item()]

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: '
                      'GEN GAN: %.4f DISC FAKE: %.4f DISC REAL: %.4f' %
                      (epoch, i, num_batch_train,
                       mean(loss_G_train), mean(loss_D_fake_train), mean(loss_D_real_train)))

                if should(num_freq_disp):
                    ## show output
                    output = transform_inv(output)
                    label = transform_inv(label)

                    writer_train.add_images('output', output, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('label', label, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

            writer_train.add_scalar('loss_G', mean(loss_G_train), epoch)
            writer_train.add_scalar('loss_D_fake', mean(loss_D_fake_train), epoch)
            writer_train.add_scalar('loss_D_real', mean(loss_D_real_train), epoch)

            # update schduler
            # schedG.step()
            # schedD.step()

            ## save
            if (epoch % num_freq_save) == 0:
                self.save(dir_chck, netG, netD, optimG, optimD, epoch)

        writer_train.close()

    def test(self):
        mode = self.mode

        batch_size = self.batch_size
        device = self.device
        gpu_ids = self.gpu_ids

        ny_in = self.ny_in
        nx_in = self.nx_in

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm

        name_data = self.name_data

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_result = os.path.join(self.dir_result, self.scope, name_data)
        dir_result_save = os.path.join(dir_result, 'images')
        if not os.path.exists(dir_result_save):
            os.makedirs(dir_result_save)

        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        ## setup network
        netG = DCGAN(nch_in, nch_out, nch_ker, norm)
        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## load from checkpoints
        st_epoch = 0

        netG, st_epoch = self.load(dir_chck, netG, mode=mode)

        ## test phase
        with torch.no_grad():
            netG.eval()
            # netG.train()

            input = torch.randn(batch_size, nch_in, ny_in, nx_in).to(device)

            output = netG(input)

            output = transform_inv(output)

            for j in range(output.shape[0]):
                name = j
                fileset = {'name': name,
                            'output': "%04d-output.png" % name}

                if nch_out == 3:
                    plt.imsave(os.path.join(dir_result_save, fileset['output']), output[j, :, :, :].squeeze())
                elif nch_out == 1:
                    plt.imsave(os.path.join(dir_result_save, fileset['output']), output[j, :, :, :].squeeze(), cmap=cm.gray)

                append_index(dir_result, fileset)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>output</th></tr>")

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    # for kind in ["input", "output", "label"]:
    for kind in ["output"]:
        index.write("<td><img src='images/%s'></td>" % fileset[kind])

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)
