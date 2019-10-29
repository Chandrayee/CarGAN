import argparse
import time
import math
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
    st.button('show animation')
    animation_image = st.empty()
    fig, ax = plt.subplots()
    ax.set_xlim(-50,50)
    ax.set_ylim(-30,30)
    plt.xlabel("Meters")
    plt.ylabel("Meters")
    CAR = {
        color: zoom(plt.imread('/home/ubuntu/CarGAN/img/car-{}.png'.format(color)), [0.3, 0.3, 1.])
        for color in ['gray', 'orange', 'purple', 'red', 'white', 'yellow']
    }
    #rect = patches.Rectangle((-50,-35),100,70,linewidth=1,edgecolor='none',facecolor='gray')
    #ax.add_patch(rect)
    scale = 5./max(CAR['gray'].shape[:2])
    st.sidebar.subheader('Select a scene to visualize')
    scene_num = st.sidebar.selectbox('Scene #',range(0,len(seq_start_end)),1)
    start, end = seq_start_end[scene_num]
    num_agents = end - start
    traj_real = traj_real[:,start:end,:].cpu().detach().numpy() #trajectories of all the agents in the scene
    traj_fake = traj_fake[:,start:end,:].cpu().detach().numpy()
    color = ['gray', 'orange', 'purple', 'red', 'yellow','white']
    angle = np.zeros((traj_fake.shape[0],traj_fake.shape[1]))
    c = 0
    for i in range(num_agents):
        x_pos = traj_fake[:,i,0]
        y_pos = traj_fake[:,i,1]
        den = x_pos[1:] - x_pos[:-1]
        num = y_pos[1:] - y_pos[:-1]
        angle[1:,i] = np.arctan2(num,den)
        ax.plot(traj_fake[:,i,0], traj_fake[:,i,1], lw = 0.5, linestyle = '--', color = color[c], label = "agent#" + str(i))
        plt.text(traj_fake[0,i,0]+2., traj_fake[0,i,1]+2.,str(i))
        c += 1
    angle = angle.reshape((angle.shape[0],angle.shape[1],1))
    x_vec = np.concatenate((traj_fake,angle),axis=2)
    cars = [AxesImage(ax, interpolation='bicubic', zorder=100) for _ in np.arange(x_vec.shape[1])]
    for num in range(x_vec.shape[0]):
        c = 0
        for car in cars:
            ax.add_artist(car)
            set_image(car,CAR[color[c]],scale,x_vec[num,c,:])
            c += 1
        animation_image.pyplot(fig)
    final_scene_num = scene_num
    st.sidebar.subheader('Select an agent to make adversarial')
    final_agent_num = st.sidebar.selectbox('Agent #',range(0,end-start),1)
    return final_scene_num, final_agent_num

#test single generator sample
# @st.cache
def generate_data(batch_num, loader, generator):
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch
            if count == batch_num:
                break
        st.write(len(seq_start_end))
        pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
        traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    return obs_traj, traj_real, traj_real_rel, traj_fake, traj_fake_rel,seq_start_end


def showscene(traj_real, traj_fake, seq_start_end):
    final_scene_num, final_agent_num = visualize(traj_real, traj_fake, seq_start_end)
    return final_agent_num, final_scene_num

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
    color = ['gray', 'orange', 'purple', 'red', 'yellow','white']
    c = 0
    for i in range(end-start):
        ax.plot(traj_real[:,i,0].cpu().detach().numpy(), traj_real[:,i,1].cpu().detach().numpy(), color = color[c])
        c += 1
    ax.plot(adv_abs[:,:,0].cpu().detach().numpy(), adv_abs[:,:,1].cpu().detach().numpy(), linestyle = '--',color=color[num-start])
    ax.set_xlim(-50,50)
    ax.set_ylim(-30,30)
    plt.xlabel('Meters')
    plt.ylabel('Meters')
    return fig

def compute_adversarial(var, t_rel, t_real, traj_real, start, end, num, discriminator,seq_start_end):
    lr = 0.001
    losses = []
    t = t_rel
    adversarial = Variable(var, requires_grad=True)
    st.sidebar.subheader('How many iterations of the optimization?')
    iter = st.sidebar.selectbox('# of iterations',range(0,5000),100)
    adv_plot = st.empty()
    adv_abs = relative_to_abs(adversarial.data[10:,:,:],t_real[10,:,:])
    fig = plot_allpaths(adv_abs,start, end, num, traj_real)
    adv_plot.pyplot(fig)
    if st.sidebar.button('Find adversarial'):
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
                fig = plot_allpaths(adv_abs,start, end, num, traj_real,)
                adv_plot.pyplot(fig)
                #time.sleep(1)
            adversarial.grad.zero_()
    return adversarial.data

def main(args):
    st.header("SynAV")
    st.write("SynAV is sequence-sequence model with adversarial loss used to synthesize future paths of cars. The input to SynAV are multiple sequences of car trajectories, each sequence is 10 steps long and the output are the 20 steps predictions of future paths of these cars. You can visualize the future trajectories of all the cars in a given traffic scene below. The dataset contains 4 scenes. Select any scene from the dropdown on the left and visualize the paths of the cars in the scene.")
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [os.path.join(args.model_path, file_) for file_ in filenames]
    else:
        if os.path.isdir('/home/ubuntu/CarGAN/models_wgan'):
            filenames = os.listdir('/home/ubuntu/CarGAN/models_wgan')
        filenames.sort()
        paths = [
                        os.path.join('/home/ubuntu/CarGAN/models_wgan', file_) for file_ in filenames
                    ]
        #paths = [args.model_path]

    print(paths)
    @st.cache(ignore_hash=True)
    def get_models():
        fast_load = st.cache(torch.load, ignore_hash=True)
        checkpoint_gen = fast_load(paths[1])
        checkpoint_dis = fast_load(paths[0])
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
        return _args, generator, discriminator, data_path, loader
    _args, generator, discriminator, data_path, loader = get_models()

    generated_data = st.cache(generate_data, ignore_hash=True, show_spinner=False)(30,loader, generator)
    obs_traj, traj_real, traj_real_rel, traj_fake, traj_fake_rel, seq_start_end = generated_data
    final_agent_num, final_scene_num = showscene(traj_real, traj_fake, seq_start_end)
    var, t_rel, t_real, start, end, num = add_noise(obs_traj, traj_real, traj_real_rel, discriminator, final_agent_num, final_scene_num, seq_start_end)
    st.write("SynAV uses the discriminator to synthesize adversarial behavior. Select an agent (car) from the dropdown on the left whose behavior you want to change. SynAV initializes the adversarial path by random noise to the generator output. It then adjusts the path of the agent by maximizing L2 loss from the correct generator output and taking gradient of the discriminator's loss function with respect to the input path similar to Fast Gradient Sign Attack. Click on Find adversarial to compute the new path. Existing paths are shown in bold line and evolving path is shown in dotted line.")
    result = compute_adversarial(var, t_rel, t_real, traj_real, start, end, num, discriminator, seq_start_end)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
