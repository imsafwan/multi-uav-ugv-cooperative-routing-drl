import os
import time
from tqdm import tqdm
import torch
import math
import csv
import csv
import os
import csv
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from termcolor import colored, cprint
from nets.attention_model_modified_v7 import set_decode_type


from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    if isinstance(model, DataParallel):
        return model.module
    else:
        return model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
           
            cost, _, _ = model(move_to(bat, opts.device))
            
        return cost.data.cpu()
    
    total_samples = len(dataset)
    batches = math.ceil(total_samples / opts.eval_batch_size)
    print(f'There are {batches} batches.')
    
    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size, num_workers=0), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    cprint("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name), 'red', attrs = ['bold'])
    step = epoch * (opts.epoch_size // opts.batch_size)
    
    
    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        uav_graph_size = opts.uav_graph_size, ugv_graph_size = opts.ugv_graph_size,  num_samples = opts.epoch_size))
    
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)
    
   
    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    
    #torch.autograd.set_detect_anomaly(True)
    start_time = time.time()
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        
        eps_start_time = time.time()
        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1
        print('\n Episode time ----->',time.time() - eps_start_time)

    epoch_duration = time.time() - start_time
    print('\n Epoch time --------->', epoch_duration)
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()



def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    # Unpack batch and move to device
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    

    # Perform forward pass
    cost, log_likelihood, unfinished_batches = model(x)

    # Identify finished batch indices
    finished_batches = torch.arange(cost.size(0), device=cost.device)  # All batch indices
    finished_batches = finished_batches[~torch.isin(finished_batches, unfinished_batches)]  # Exclude unfinished batches

    # Gather costs of finished batches
    finished_costs = cost[finished_batches]

    # Calculate the average cost of finished batches
    avg_finished_cost = finished_costs.mean()


    cprint(' Batch instance score in an episode ------> {} '.format(cost[9]), 'green', attrs=['bold'])
    cprint(' Avg batch score in an episode ------> {} '.format(torch.mean(cost)), 'blue', attrs=['bold'])

    # Evaluate baseline, get baseline loss
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate raw advantage
    advantage = cost - bl_val
    

    # Compute REINFORCE loss
    reinforce_loss = (advantage * log_likelihood).mean()
    

    # Calculate total loss
    loss = reinforce_loss + bl_loss
    

    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print("NaN or Inf detected in loss before backward")
        raise RuntimeError("NaN or Inf detected in loss")

    # Perform backward pass
    optimizer.zero_grad()
    loss.backward()


    # Clip gradient norms
    clipped_grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    

    # TensorBoard logging
    if not opts.no_tensorboard:
        tb_logger.log_value('loss', loss.item(), step)
        #tb_logger.log_value('entropy', entropy.item(), step)
        tb_logger.log_value('bl_val', bl_val.mean().item(), step)
        tb_logger.log_value('bl_loss', bl_loss, step)
        tb_logger.log_value('unfinished batches', unfinished_batches.numel(), step)
        tb_logger.log_value('finished avg cost', avg_finished_cost.item(), step)

    # Debugging gradients for the batch
    if step % int(opts.log_step) == 0:
        log_values(cost, clipped_grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)





