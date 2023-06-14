import argparse
import os
import torch
from pathlib import Path
import jax
import numpy as np
from torch.utils.data import random_split
from data_loader import LCSFEM_Bias_Dataset, NumpyLoader
from model import Model, MLP
from train import train_one_epoch
from eval import eval

from jax.lib import xla_bridge

import jax.numpy as jnp
from flax.training import train_state, orbax_utils
from flax.metrics.tensorboard import SummaryWriter
import orbax.checkpoint
from flax import linen as nn
import optax

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = '\"platform\"'


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    parser.add_argument('--test_ratio', default=0.2, type=float,
                        help='Test ratio for splitting the dataset')
    parser.add_argument(
        '--batch_size', default=64, type=int,
        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument(
        '--accum_iter', default=1, type=int,
        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # model parameters
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--n_input_vars', default=3, type=int,
                        help='images input size')

    # optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument(
        '--blr', type=float, default=1e-3, metavar='LR',
        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # dataset parameters
    parser.add_argument(
        '--purple_air_dir',
        default='../../NASA_Citizen_Science/data/PurpleAir/SF', type=str,
        help='pa dataset path')

    parser.add_argument(
        '--air_now_dir',
        default='../../NASA_Citizen_Science/data/airNow/stations', type=str,
        help='AirNow dataset path')

    parser.add_argument(
        '--pair_file',
        default='../../NASA_Citizen_Science/metrics/min_great_circle_df.csv',
        type=str, help='Distance file path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem', action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


@jax.vmap
def mse_loss(preds, targets):
    return jnp.mean((preds.flatten() - targets)**2)


def main(args):
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    jax.random.PRNGKey(seed)
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)

    # init the dataset
    dataset = LCSFEM_Bias_Dataset(
        args.pair_file, args.purple_air_dir, args.air_now_dir, ln_scale=True)
    len_test = int(len(dataset) * args.test_ratio)
    train_set, test_set = random_split(
        dataset, [(len(dataset) - len_test), len_test])
    print(len(train_set), len(test_set))
    print(len(dataset))

    training_generator = NumpyLoader(
        train_set, batch_size=args.batch_size, num_workers=0)
    testing_generator = NumpyLoader(
        test_set, batch_size=args.batch_size, num_workers=0)

    # init the model
    model = MLP([args.n_input_vars, 256, 128, 64, 1])

    # init train state
    init_data = jnp.ones((args.batch_size, args.n_input_vars), jnp.float32)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(rng, init_data)['params'],
        tx=optax.adam(args.lr),
    )

    # init eval state
    rng, z_key, eval_rng = jax.random.split(rng, args.n_input_vars)
    z = jax.random.normal(z_key, (64, 256))

    # init checkpoint dir
    ckpt_dir = 'ckpts'

    # checkpoint
    config = {'batch_size': args.batch_size, 'accum_iter': args.accum_iter}
    ckpt = {'model': state, 'config': config, 'data': training_generator}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=5, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        'ckpts/deter', orbax_checkpointer, options)
    
    # tensorboard
    summary_writer = SummaryWriter('logs/deter')

    # training the model
    for epoch in range(args.epochs):
        # training loop
        state, train_loss = train_one_epoch(state, training_generator)

        
        # save checkpoint
        ckpt = {'model': state, 'config': config, 'data': training_generator}
        checkpoint_manager.save(epoch, ckpt, save_kwargs={
                                'save_args': save_args})

        # evaluation loop
        eval_loss = eval(state, testing_generator)

        print(
            f'Epoch: {epoch:d}, train_loss: {train_loss:.4f}, eval_loss: {eval_loss:.4f}')
        summary_writer.scalar('train_loss', train_loss, epoch)
        summary_writer.scalar('eval_loss', eval_loss, epoch)

    # predict the testing set using the trained model
    y_true, y_pred = predict(testing_generator, checkpoint_manager)
    # export the prediction
    y_true = jnp.concatenate(y_true)
    y_pred = jnp.concatenate(y_pred)
    output_dir = Path(args.output_dir)
    jnp.savetxt(output_dir / "test_y_true.csv", y_true, delimiter=",")
    jnp.savetxt(output_dir / "test_y_pred.csv", y_pred, delimiter=",")
    
    
def predict(testing_generator, checkpoint_manager):
    y_true = []
    y_pred = []
    # restor the latest state
    step = checkpoint_manager.latest_step()
    state = checkpoint_manager.restore(step)
    for i, (x, y) in enumerate(testing_generator):
        sub_y_pred = state.apply_fn({'params': state.params}, x)
        y_pred.append(sub_y_pred)
        y_true.append(y)
    return y_true, y_pred
        

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
