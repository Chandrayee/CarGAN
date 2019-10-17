import argparse
import os
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.ndimage import rotate, zoom
from attrdict import AttrDict
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.image import BboxImage, AxesImage
from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path
import matplotlib.patches as patches
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='models_wgan',type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--dataset_name', default='smalltown01', type=str)

def set_image(obj, data, scale, x=[0., 0., 0.]):
    ox = x[0]
    oy = x[1]
    angle = x[2]
    img = rotate(data, np.rad2deg(angle))
    h, w = img.shape[0], img.shape[1]
    obj.set_data(img)
    obj.set_extent([ox-scale*w*0.5, ox+scale*w*0.5, oy-scale*h*0.5, oy+scale*h*0.5])

def get_generator(args, checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator

def get_discriminator(args,checkpoint):
    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type)
    discriminator.load_state_dict(checkpoint['d_state'])
    discriminator.cuda()
    discriminator.train()
    return discriminator

def visualize(traj_real, traj_fake, seq_start_end):
    fig, ax = plt.subplots()
    CAR = {
        color: zoom(plt.imread('img/car-{}.png'.format(color)), [0.3, 0.3, 1.])
        for color in ['gray', 'orange', 'purple', 'red', 'white', 'yellow']
    }
    rect = patches.Rectangle((-50,-35),100,70,linewidth=1,edgecolor='none',facecolor='gray')
    ax.add_patch(rect)
    color = ['gray', 'orange', 'purple', 'red', 'yellow','white']
    scale = 5./max(CAR['gray'].shape[:2])
    scene_num = st.sidebar.selectbox('scene #',range(0,len(seq_start_end)),1)
    start, end = seq_start_end[scene_num]
    print(start,end)
    num_agents = end - start
    traj_real = traj_real[:,start:end,:].cpu().detach().numpy() #trajectories of all the agents in the scene
    traj_fake = traj_fake[:,start:end,:].cpu().detach().numpy()
    color=iter(cm.rainbow(np.linspace(0,1,num_agents)))
    for i in range(num_agents):
        c=next(color)
        ax.plot(traj_fake[:,i,0], traj_fake[:,i,1], linestyle = '--', color = c, label = "agent#" + str(i))
        ax.plot(traj_real[:,i,0], traj_real[:,i,1], color = c)
    ax.set_xlim(-50,50)
    ax.set_ylim(-50,50)
    plt.xlabel('Meters')
    plt.ylabel('Meters')
    ax.legend()
    st.write(fig)
    final_scene_num = scene_num
    final_agent_num = st.sidebar.selectbox('Agent #',range(0,end-start),1)
    return final_scene_num, final_agent_num

#test single generator sample
def generate_data(batch_num, loader, generator):
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch
            if count == batch_num:
                break
        pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
        traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
        final_scene_num, final_agent_num = visualize(traj_real, traj_fake, seq_start_end)
    return obs_traj, traj_real, traj_real_rel, traj_fake, traj_fake_rel, final_agent_num, final_scene_num,seq_start_end

    #create noise
def add_noise(obs_traj, traj_real, traj_real_rel, discriminator, agent_num, scene_num, seq_start_end):
    past = obs_traj.size(0)
    future = traj_real_rel.size(0) - past
    start, end = seq_start_end[scene_num]
    num = start+agent_num
    t_rel = traj_real_rel[:,num,:]
    t_real = traj_real[:,num,:]
    n_past = torch.zeros(past + 1, obs_traj.size(2))
    r = 0.8 #perturbation sampled from normal distribution
    n_future = ((r - (-r))*torch.rand(future - 1,obs_traj.size(2)) + (-r))
    n = (torch.cat([n_past,n_future],dim=0)).cuda()
    traj_p_real = t_real.add(n)
    traj_p_rel = torch.zeros_like(traj_p_real)
    traj_p_rel[1:,:] = traj_p_real[1:,:] - traj_p_real[:-1,:]
    var = traj_p_rel.unsqueeze(dim=0).permute(1,0,2)
    t_rel = t_rel.unsqueeze(dim=0).permute(1,0,2)
    t_real = t_real.unsqueeze(dim=0).permute(1,0,2)
    return var, t_rel, t_real, start, end, num

def plot_allpaths(adv_abs,start, end, num, traj_real):
    fig, ax = plt.subplots()
    for i in range(end-start):
        ax.plot(traj_real[:,i,0].cpu().detach().numpy(), traj_real[:,i,1].cpu().detach().numpy())
    ax.plot(adv_abs[:,:,0].cpu().detach().numpy(), adv_abs[:,:,1].cpu().detach().numpy())
    ax.set_xlim(-50,50)
    ax.set_ylim(-50,50)
    plt.xlabel('Meters')
    plt.ylabel('Meters')
    st.write(fig)

def compute_adversarial(var, t_rel, t_real, traj_real, start, end, num, discriminator,seq_start_end):
    lr = 0.001
    losses = []
    t = t_rel
    print(var.size())
    adversarial = Variable(var, requires_grad=True)
    adv_abs = relative_to_abs(adversarial.data[10:,:,:],t_real[10,:,:])
    st.write("Showing the starting configuration ....")
    plot_allpaths(adv_abs,start, end, num, traj_real)
    st.write('Select the number of iterations for optimization')
    iter = st.sidebar.selectbox('# of iterations',range(0,1000),100)
    if st.sidebar.checkbox('Find adversarial'):
        for idx in range(iter):
            discriminator.zero_grad()
            out = discriminator(t_real,adversarial,seq_start_end[0])
            loss= -0.5*torch.sum((t - adversarial)**2) - out
            loss.backward()
            adversarial.data[10:,:,:] = adversarial.data[10:,:,:] - lr * adversarial.grad.data[10:,:,:]
            losses.append(loss)
            if idx % 20 == 0:
                print('Loss: {}'.format(loss))
                adv_abs = relative_to_abs(adversarial.data,t_real[0,:,:])
                st.write("Showing configurations every 20 iterations ....")
                plot_allpaths(adv_abs,start, end, num, traj_real)
            adversarial.grad.zero_()
    return adversarial.data

def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]
    print(paths)
    checkpoint_gen = torch.load(paths[1])
    checkpoint_dis = torch.load(paths[0])
    _args = dict([('clipping_threshold_d',0), ('obs_len',10), ('batch_norm',False), ('timing',0),
             ('checkpoint_name','gan_test'), ('num_samples_check',5000), ('mlp_dim',64), ('use_gpu',1), ('encoder_h_dim_d',16),
             ('num_epochs',900), ('restore_from_checkpoint',1), ('g_learning_rate',0.0005), ('pred_len',20), ('neighborhood_size',2.0),
             ('delim','tab'), ('d_learning_rate',0.0002), ('d_steps',2), ('pool_every_timestep', False), ('checkpoint_start_from', None),
             ('embedding_dim',16), ('d_type','local'), ('grid_size',8), ('dropout',0.0), ('batch_size',4), ('l2_loss_weight',1.0),
             ('encoder_h_dim_g',16), ('print_every',10), ('best_k',10), ('num_layers',1), ('skip',1), ('bottleneck_dim',32), ('noise_type','gaussian'),
             ('clipping_threshold_g',1.5), ('decoder_h_dim_g',32), ('gpu_num','0'), ('loader_num_workers',4), ('pooling_type','pool_net'),
             ('noise_dim',(20,)),('g_steps',1), ('checkpoint_every',50), ('noise_mix_type','global'), ('num_iterations',80000)])
    _args = AttrDict(_args)
    generator = get_generator(_args, checkpoint_gen)
    discriminator = get_discriminator(_args, checkpoint_dis)
    data_path = get_dset_path(args.dataset_name, args.dset_type)
    _, loader = data_loader(_args, data_path)
    obs_traj, traj_real, traj_real_rel, traj_fake, traj_fake_rel, final_agent_num, final_scene_num,seq_start_end = generate_data(10,loader, generator)
    var, t_rel, t_real, start, end, num = add_noise(obs_traj, traj_real, traj_real_rel, discriminator, final_agent_num, final_scene_num, seq_start_end)
    result = compute_adversarial(var, t_rel, t_real, traj_real, start, end, num, discriminator, seq_start_end)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
