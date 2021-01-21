import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from nets.crn import CRNModel
import os
from utils.logger_utils import get_logger
from recipe_dataset import MyDataset
from collec_function import basic_collection_function
from torch.utils.data import DataLoader
from utils.plot_utils import add_spect_image
import tqdm


def set_seed(manual_seed):
    import random
    from torch.backends import cudnn
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    cudnn.benchmark = True
    # cudnn.deterministic = True


def static_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(pt_path, model, optimizer, global_step, current_epoch, train_loss, valid_loss):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "global_epoch": current_epoch,
        "train_loss": train_loss,
        "valid_loss": valid_loss}, pt_path)


def load_checkpoint(pt_path, model, optimizer):
    checkpoint = torch.load(pt_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint['global_step']
    epoch = checkpoint['global_epoch']
    train_loss = checkpoint['train_loss']
    valid_loss = checkpoint['valid_loss']
    return epoch, step, train_loss, valid_loss


def make_nonzero_masks(xs, lens):
    masks = torch.zeros_like(xs)
    for i in range(lens.size(0)):
        masks[i, :lens[i], :] = 1.
    return masks


def calc_loss(outs, targets, masks, use_split_loss, loss_func):
    error = loss_func(outs, targets, reduction='none') * masks
    if use_split_loss:
        pitch_loss = error[:, :, -3:].sum() / masks[:, :, -3:].sum()
        fbank_loss = error[:, :, :-3].sum() / masks[:, :, :-3].sum()
        return {"pitch_loss": pitch_loss, "fbank_loss": fbank_loss, "loss": pitch_loss+fbank_loss}
    else:
        loss = error.sum() / masks.sum()
        return {"loss": loss}


def train(model, opt, my_dataset):
    pbar = tqdm.tqdm(total=my_dataset.__len__(), ascii=True, unit="batch")
    model.train()
    train_loss = 0
    global global_step
    for i, np_batch in enumerate(my_dataset):
        with torch.no_grad():
            mini_batch = {}
            for key, value in np_batch.items():
                if value.dtype == np.float32 or value.dtype == np.float64:
                    mini_batch[key] = torch.Tensor(value).to(device)
                elif value.dtype == np.int32:
                    mini_batch[key] = torch.Tensor(value).long().to(device)
                else:
                    mini_batch[key] = value

        outs = model.forward(mini_batch["noisy"], mini_batch["length"].squeeze())
        masks = make_nonzero_masks(outs, mini_batch["length"].squeeze())
        loss_dict = calc_loss(outs, mini_batch["speech"], masks, args.use_split_loss, loss_func)
        loss = loss_dict["loss"]

        if np.isnan(loss.item()) or np.isinf(loss.item()):
            logger.warn("The loss is Nan or Inf, skip it.")
            continue

        model.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        train_loss += loss.item()

        if global_step % 1000 == 0:
            for key, value in loss_dict.items():
                summary_writer.add_scalar(f"train/{key}", value.item(), global_step)
            add_spect_image(summary_writer, mini_batch["noisy"][:16, :, :], f"train/noisy", 4, global_step)
            add_spect_image(summary_writer, mini_batch["speech"][:16, :, :], f"train/speech", 4, global_step)
            add_spect_image(summary_writer, outs[:16, :, :], f"train/enhanced", 4, global_step)
            logger.info(f"Epoch {current_epoch}, step {global_step}: train_loss: {loss.item()}")
        global_step += 1
        pbar.update(1)
        # summary_writer.flush()
    return train_loss / my_dataset.__len__()


def valid(model, opt, my_dataset):
    pbar = tqdm.tqdm(total=my_dataset.__len__(), ascii=True, unit="batch")
    model.eval()
    valid_loss = 0
    for i, np_batch in enumerate(my_dataset):
        with torch.no_grad():
            mini_batch = {}
            for key, value in np_batch.items():
                if value.dtype == np.float32 or value.dtype == np.float64:
                    mini_batch[key] = torch.Tensor(value).to(device)
                elif value.dtype == np.int32:
                    mini_batch[key] = torch.Tensor(value).long().to(device)
                else:
                    mini_batch[key] = value

            outs = model.forward(mini_batch["noisy"], mini_batch["length"].squeeze())
            masks = make_nonzero_masks(outs, mini_batch["length"].squeeze())
            loss_dict = calc_loss(outs, mini_batch["speech"], masks, args.use_split_loss, loss_func)
            loss = loss_dict["loss"]

            one_valid_loss = loss.item()
            valid_loss += one_valid_loss
            assert not (np.isnan(valid_loss) or np.isinf(valid_loss)), logger.critical("Nan or Inf loss detected!!")
            if i == 0:
                for key, value in loss_dict.items():
                    summary_writer.add_scalar(f"valid/{key}", value.item(), current_epoch)
                add_spect_image(summary_writer, mini_batch["noisy"][:16, :, :], f"valid/noisy", 4, current_epoch)
                add_spect_image(summary_writer, mini_batch["speech"][:16, :, :], f"valid/speech", 4, current_epoch)
                add_spect_image(summary_writer, outs[:16, :, :], f"valid/speech", 4, current_epoch)
            pbar.update(1)
    return valid_loss / my_dataset.__len__()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tag", type=str, default="crn", help="Name of this train")
    parser.add_argument("--train_recipe", type=str, default="tr/feat.scp",
                        help="Script file of training set including noisy and clean features")
    parser.add_argument("--valid_recipe", type=str, default="vc/feat.scp",
                        help="Script file of valid set including noisy and clean features")
    parser.add_argument("--max_num_saved_checkpoint", type=int, default=10,
                        help="The maximum number of saved checkpoints, early checkpoints will be removed to save disk")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="The patience of early stopping")
    parser.add_argument("--grad_clip", type=float, default=0, help="Clip the gradient to a given number")
    parser.add_argument("--total_epoch", type=int, default=200, help="The total number of training epoch")
    parser.add_argument("--max_lens", type=int, default=800, help="The maximum length of features")
    parser.add_argument("--work_num", type=int, default=4, help="The work number of dataloader")
    parser.add_argument("--cp_path", type=str, default="checkpoints", help="Dictionary of checkpoints")
    parser.add_argument("--log_path", type=str, default="log", help="Dictionary of log files")
    parser.add_argument("--tb_path", type=str, default="tensorboard_dir", help="Dictionary of tensorboard summary")
    parser.add_argument("--batch_size", '-b', type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint path to load")
    parser.add_argument("--train_first", type=bool, default=False, help="Whether to train the model before eval")
    parser.add_argument("--loss_type", type=str, default="l1", choices=['l1', 'l2'], help="The loss function")
    parser.add_argument("--use_split_loss", type=bool, default=False,
                        help="Whether to calculate loss for pitch and fbank features, separately")
    parser.add_argument("--cmvn_path", type=str, default=None, help="The cmvn ark file to apply...")
    args = parser.parse_args()

    set_seed(0)
    device = torch.device("cuda")
    batch_size = args.batch_size
    checkpoint_dir = os.path.join(args.cp_path, args.run_tag)
    tensorboard_dir = os.path.join(args.tb_path, args.run_tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    logger = get_logger(os.path.join(args.log_path, args.run_tag))
    summary_writer = SummaryWriter(os.path.join(args.tb_path, args.run_tag), flush_secs=10)
    if args.max_num_saved_checkpoint > args.early_stopping_patience:
        args.max_num_saved_checkpoint = args.early_stopping_patience + 1

    logger.info("Initializing the enhancement method, model and optimizer")
    crn = CRNModel(dim=83, causal=False, units=256, conv_channels=8, use_batch_norm=True, pitch_dims=3)
    ngpu = torch.cuda.device_count()
    if ngpu > 1:
        logger.info(f"Let's use {ngpu} GPUs !")
        crn = nn.DataParallel(crn, device_ids=list(range(ngpu)))
        logger.info(f"Automatically increase the batch size from {args.batch_size} -> {args.batch_size * ngpu}")
        args.batch_size *= ngpu
        args.work_num *= ngpu
    crn.to(device)
    optimizer = torch.optim.Adam(crn.parameters(), args.learning_rate, amsgrad=True)
    logger.info(f"The model has {static_parameters(crn) / 2 ** 20:.2f} M trainable parameters.")
    loss_func = F.l1_loss if args.loss_type == 'l1' else F.mse_loss
    logger.info(f"Use {args.loss_type} loss, use_split_loss: {args.use_split_loss}")

    current_epoch = 0
    global_step = 0
    if args.load_checkpoint is not None:
        current_epoch, global_step, tr_loss, vc_loss = load_checkpoint(args.load_checkpoint, crn, optimizer)
        logger.info(f"Load checkpoint from {args.load_checkpoint}, epoch {current_epoch}, step {global_step}")

    train_set = MyDataset(args.train_recipe, args.max_lens, cmvn_path=args.cmvn_path, reverse_cmvn=False)
    train_dataset = DataLoader(train_set, args.batch_size, True, num_workers=args.work_num,
                               collate_fn=basic_collection_function, drop_last=True, pin_memory=True)
    valid_set = MyDataset(args.valid_recipe, args.max_lens, cmvn_path=args.cmvn_path, reverse_cmvn=False)
    valid_dataset = DataLoader(valid_set, args.batch_size, True, num_workers=args.work_num,
                               collate_fn=basic_collection_function, drop_last=True, pin_memory=True)

    lowest_valid_loss = None
    early_stopping_patience = args.early_stopping_patience
    saved_checkpoint_list = []

    if not args.train_first:
        with torch.no_grad():
            valid_loss = valid(crn, optimizer, valid_dataset)
        logger.info(f"Initial valid loss: {valid_loss:.6f}")
        summary_writer.add_scalar("valid/loss", valid_loss, current_epoch)

    for _ in range(args.total_epoch):
        train_loss = train(crn, optimizer, train_dataset)
        assert not (np.isnan(train_loss) or np.isinf(train_loss)), f"Invalid training loss {train_loss}."
        current_epoch += 1
        with torch.no_grad():
            valid_loss = valid(crn, optimizer, valid_dataset)
            assert not (np.isnan(train_loss) or np.isinf(train_loss)), f"Invalid validation loss {valid_loss}."
        logger.info(f"Epoch {current_epoch:03d}, step {global_step:06d}: "
                    f"Train loss {train_loss:.6f}, valid loss {valid_loss:.6f}")
        summary_writer.add_scalar("valid/loss", valid_loss, current_epoch)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{global_step:09d}.pt")
        save_checkpoint(checkpoint_path, crn, optimizer, global_step, current_epoch, train_loss, valid_loss)
        logger.info(f"Save the model to {checkpoint_path}")

        # remove too old checkpoint
        saved_checkpoint_list.append(checkpoint_path)
        if len(saved_checkpoint_list) > args.max_num_saved_checkpoint:
            to_remove_checkpoint = saved_checkpoint_list[-args.max_num_saved_checkpoint - 1]
            os.remove(to_remove_checkpoint)
            logger.info(f"Remove {to_remove_checkpoint}")

        # save best model as best_model.pt
        if lowest_valid_loss is None or valid_loss <= lowest_valid_loss:
            lowest_valid_loss = valid_loss
            save_checkpoint(os.path.join(checkpoint_dir, "best_model.pt"), crn, optimizer, global_step,
                            current_epoch, train_loss, valid_loss)
            early_stopping_patience = args.early_stopping_patience
            logger.info(f"Save the best model!")

        # early stopping
        if valid_loss > lowest_valid_loss:
            early_stopping_patience -= 1
            if early_stopping_patience <= 0:
                logger.critical("Early stopped!")
                break
    logger.critical("Train completed!")
