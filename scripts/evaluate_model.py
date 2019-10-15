import argparse
import os
import torch
from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from attrdict import AttrDict
import matplotlib
import matplotlib.pyplot as plt
from sgan.utils import relative_to_abs, get_dset_path
from sgan.losses import displacement_error, final_displacement_error


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--dataset_name', default='smalltown01', type=str)

def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_

def get_generator(args,checkpoint):
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

def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = batch
            ade, fde, d_real, d_fake = [], [], [], []
            total_traj += pred_traj_gt.size(1)
            for _ in range(num_samples):
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
                fde.append(final_displacement_error(
                            pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))
                traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
                traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
                traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
                traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)
            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * 20)
        fde = sum(fde_outer) / (total_traj)
    print(ade)
    print(fde)
    return ade, fde

def main(args):
    if os.path.isdir(args.model_path):
        file_ = os.listdir(args.model_path)
        print(file_)
        path = os.path.join(args.model_path, file_[0])
    else:
        path = args.model_path
    checkpoint = torch.load(path)
    _args = dict([('clipping_threshold_d',0), ('obs_len',10), ('batch_norm',False), ('timing',0),
             ('checkpoint_name','gan_test'), ('num_samples_check',5000), ('mlp_dim',64), ('use_gpu',1), ('encoder_h_dim_d',16),
             ('num_epochs',900), ('restore_from_checkpoint',1), ('g_learning_rate',0.0005), ('pred_len',20), ('neighborhood_size',2.0),
             ('delim','tab'), ('d_learning_rate',0.0002), ('d_steps',2), ('pool_every_timestep', False), ('checkpoint_start_from', None),
             ('embedding_dim',16), ('d_type','local'), ('grid_size',8), ('dropout',0.0), ('batch_size',4), ('l2_loss_weight',1.0),
             ('encoder_h_dim_g',16), ('print_every',10), ('best_k',10), ('num_layers',1), ('skip',1), ('bottleneck_dim',32), ('noise_type','gaussian'),
             ('clipping_threshold_g',1.5), ('decoder_h_dim_g',32), ('gpu_num','0'), ('loader_num_workers',4), ('pooling_type','pool_net'),
             ('noise_dim',(20,)),('g_steps',1), ('checkpoint_every',50), ('noise_mix_type','global'), ('num_iterations',80000)])
    _args = AttrDict(_args)
    generator = get_generator(_args,checkpoint)
    data_path = get_dset_path(args.dataset_name, args.dset_type)
    _, loader = data_loader(_args, data_path)
    ade, fde = evaluate(_args, loader, generator, args.num_samples)
    print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            args.dataset_name, _args.pred_len, ade, fde))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
