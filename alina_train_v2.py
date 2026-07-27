# alina_train.py
#### import packages ####

import hydra
from omegaconf import DictConfig, OmegaConf
import json
import time
import numpy as np
import random
import pickle
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
import tqdm 
import naskit as nsk
# special import for torch
sys.path.append('/usr/local/lib/python3.12/dist-packages') #path to global torch
import torch
import alina
from alina import AlinaDataset
import mlflow
#### support functions ####
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_checkpoint(path, device): #!!!!
    state = torch.load(path, map_location=device, weights_only=False)
    model = alina.AliNA(
         model_parameters = state["model_params"],
         dimer_embeddings = state["dimer_embeddings"],
         center_pad = state["center_pad"] )
    model.load_state_dict(state["model_state_dict"])
    optim = torch.optim.AdamW(model.parameters(), lr=1)
    optim.load_state_dict(state["optim_state_dict"])

    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    return (model, optim)

class Checkpointer:
    def __init__(self, dir_path, model, model_params, optim, maximize_metrics):
        # maximize_metrics:
        #           True  if a model maximizes metrics (Fscore),
        #           False if it minimizes metrics
        self.maximize_metrics = maximize_metrics
        self.dir_path = Path(dir_path)
        self.model = model
        self.model_params = model_params
        self.optim = optim
        self.best_value = float("-inf") if maximize_metrics else float("inf")
        self.best_model_name = None
    
    def __call__(self, name):
        if not self.dir_path.is_dir():
            os.mkdir(self.dir_path)

        state = self.model.state
        state["optim_state_dict"] = self.optim.state_dict()
        torch.save(state, self.dir_path/f"{name}.pth")
        
    def save_by_metric(self, name, value):
        save_model = False
        # flag which indicates whether a model should be saved or not
        if self.maximize_metrics:
            if value >= self.best_value:
                save_model = True
        else: #if a model minimizes metrics
            if value <= self.best_value:
                save_model = True
            
        if not save_model:
            return
            
        best_name = f"best_val={value:.4f}_{name}"
        if self.best_model_name is not None:
            os.remove(self.dir_path/f"{self.best_model_name}.pth")
        self.best_value = value
        self.best_model_name = best_name
        self(self.best_model_name)

def get_cexplr_scheduler(warmup=200, peak=5e-4, c=4e-4, 
                         min_lr=5.e-5, max_lr=1.e-3,
                         rules=[]
                        ):
    ### create scheduler ###
    def lr_scheduler(n):
        for sth, v in rules[::-1]:
            if n>=sth:
                return v

        if n<warmup:
            lr = ((peak)/warmup)*n
        else:
            ampl = peak - min_lr
            step = (n-warmup)
            step = step * (0.25 + 0.75*np.exp(-step))
            lr = (ampl*0.5)*np.cos(c*step) + (ampl*0.5 + min_lr)
        
        if lr>max_lr: lr=max_lr
        if n>warmup and (step*c >= np.pi): lr=min_lr

        return lr
    return lr_scheduler

def wBCELoss(p, y):
    # Arguments:
    # p: model's prediction
    # y: ground truth
    a = - y*torch.log(p + 1e-7) # a ~ 1
    b = - (1.0-y)*torch.log(1.0 - p + 1e-7) 

    n_bonds = y.sum()  
    batch, seq, _ = p.shape
    n_free = batch*seq*seq - n_bonds 
    loss = a.sum()/n_bonds + b.sum()/n_free
    return loss
    
def FLoss(p, y):
    tp = torch.sum(p*y, dim=(1,2)) # batch, seq, seq -> batch
    psum = p.sum(dim=(1,2))
    
    recall = torch.mean(tp / y.sum(dim=(1,2)))
    precision = torch.mean(tp / (psum + 1e-7))
    loss = 2*precision*recall / (precision + recall)
    return loss

def AlinaMetrics(p, y):
    TH, eps = 0.5, 1e-7
    p = (p>=TH).float() # probs to labels, not differenc. --> loss is not calculated
    
    tp = torch.sum(p*y, dim=(1,2)) # batch, seq, seq -> batch
    tp_fp = p.sum(dim=(1,2))
    tp_fn = y.sum(dim=(1,2))
    
    # psum[(psum==0.)] = 1.
    
    recall    = tp / (tp_fn + eps)
    precision = tp / (tp_fp + eps)
    fscore = 2*precision*recall / (precision + recall + eps)
    
    fscore, recall, precision = float(fscore.mean()), float(recall.mean()), float(precision.mean())

    return recall, precision, fscore

def validate(model, loader, loss_fn, device): #!!!
    val_pred, val_y = [], []
    with torch.no_grad():
        for x, y, _, _ in tqdm.tqdm(loader):
            pred = model(x.to(device)).cpu()
            val_pred.append(pred)
            val_y.append(y)

    val_pred, val_y = torch.cat(val_pred, dim=0), torch.cat(val_y, dim=0)
    #loss = Loss(pred, y) #??? why do we calculate loss only with the last pair (pred,y)
    loss = loss_fn(val_pred, val_y) #!!!
    recall, precision, Fscore = AlinaMetrics(val_pred, val_y)
    return loss, recall, precision, Fscore

def train(model, train_loader, valid_loader, device, optim, loss_fn,
          lr_scheduler, scaler, checkpointer, log_fn,
          LOG_EVERY, VALID_EVERY, max_train_steps,
          grad_acum, clip_grad):
    #model, train_loader, valid_loader, device, optim, loss_fn, max_train_steps 
    #lr_scheduler, scaler, checkpointer, grad_acum, LOG_EVERY, VALID_EVERY
    global_step = 0
    train_step = 0
    ep = 0
    
    step_start_time = time.time()
    iter_start_time = time.time()
    pred_list, true_list = [], []
    
    try:
        while (train_step<max_train_steps):
            ep+=1
            model.train()
            for i, (x, y, _, _) in enumerate(train_loader):
                # step == 1 batch
                global_step += 1
                true_list.append(y)
                x, y = x.to(device), y.to(device)
    
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # 32 --> 16 (model, loss)
                    pred = model(x) #model prediction
                    loss = loss_fn(pred, y) #calculate loss
                    pred_list.append(pred.detach().cpu()) #without grads --> to cpu
                        
                scaler.scale(loss).backward()
                
                if (global_step + 1)%grad_acum==0:
                    scaler.unscale_(optim) #optim == Adam
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad() 
                    lr_scheduler.step() #update lr
    
                    stepps = 1/(time.time() - step_start_time)
                    step_start_time = time.time()
                    
                    train_step += 1
                    if train_step%LOG_EVERY==0: #logging
                        y = true_list[0] if len(true_list)==1 else torch.cat(true_list, dim=0)
                        pred = pred_list[0] if len(pred_list)==1 else torch.cat(pred_list, dim=0)
        
                        loss = loss_fn(pred, y)
                        recall, precision, Fscore = AlinaMetrics(pred, y)
    
                        log_fn('Loss', float(loss), train_step)
                        log_fn('recall', recall, train_step)
                        log_fn('precision', precision, train_step)
                        log_fn('Fscore', Fscore, train_step)
                        log_fn('Lr', lr_scheduler.get_last_lr()[0], train_step)
                        log_fn('step/s', stepps, train_step)
    
                    pred_list, true_list = [], []
    
                    if train_step%VALID_EVERY==0:
                        model.eval()
                        print("\n--- Validation")
                        loss, recall, precision, Fscore = validate(model, valid_loader, loss_fn, device)
                        model.train()
                        log_fn('valid_Loss', float(loss), train_step)
                        log_fn('valid_recall', recall, train_step)
                        log_fn('valid_precision', precision, train_step)
                        log_fn('valid_Fscore', Fscore, train_step)
                        checkpointer.save_by_metric(f"step={train_step}", Fscore)
    
                    # if train_step%SAVE_EVERY==0:
                    #     checkpointer(f"step={train_step}")
    
                itps = 1/(time.time() - iter_start_time)
                iter_start_time = time.time()
                print(f"\rEp: {ep} | {i+1} / {len(train_loader)} | {itps:.2f} it/s | "
                    f"Loss: {round(float(loss), 6)}; ", end='')
                    
                if train_step>=max_train_steps:
                    break
        print()
    
    except KeyboardInterrupt:
        mlflow.end_run()
        print("\n---   Stopped   ---")
    
    mlflow.end_run()
    checkpointer(f"step={train_step}")

@hydra.main(version_base=None, config_path="configs", config_name="alina")
def main(cfg: DictConfig) -> None:

    #### set random seed for reproducibility ####
    set_seeds(cfg.random_seed) 
    #### set up ml-flow experiment ####
    
    mlflow.set_tracking_uri(uri="http://127.0.0.1:31420") 
    mlflow.set_experiment(cfg.experiment_name)

    #### set up loss function ####
    if (cfg.special_params.LOSS=="Floss"):
        loss_fn = FLoss
        maximize = True
    elif (cfg.special_params.LOSS=="BCE"):
        loss_fn = wBCELoss
        maximize = False
    else:
        pass #throw an error
    
    #### unpack parameters ####
    params = OmegaConf.to_container(cfg, resolve=True)

    model_parameters = params['hparams']
    lr_params        = params['lr_params']
    
    dimer_embeddings = cfg.special_params.DIMER_EMBED
    center_pad       = cfg.special_params.CENTER_PAD
    max_len          = cfg.special_params.MAX_LEN

    max_train_steps  = cfg.const.MAX_TRAIN_STEPS
    weight_decay     = cfg.const.WEIGHT_DECAY #default: 0.1 
    valid_every      = cfg.const.VALID_EVERY
    batch_size       = cfg.const.BATCH_SIZE
    clip_grad        = cfg.const.CLIP_GRAD
    log_every        = cfg.const.LOG_EVERY
    grad_acum        = cfg.const.GRAD_ACUM

    output_dir       = Path(cfg.work_dir)
    maximize_metrics = True

    device = torch.device(f"cuda:{cfg.const.DEVICE_IDX}" if torch.cuda.is_available() else "cpu")
    run_name = f"{cfg.run_name_prefix}_{cfg.run_name_template.template}"

    #### lr scheduler ####
    lr_func = get_cexplr_scheduler(**lr_params)
    print(f"lr parameters: {lr_params}\n")
    print(f"model parameters: {model_parameters}\n")
    
    #### set up model and optim ####
    if (cfg.checkpoint_path is None):
        ### create new model and optimizer
        print("\nCreate model and optimizer...")
        model = alina.AliNA(model_parameters,
                            dimer_embeddings = dimer_embeddings,
                            center_pad = center_pad)
        optim = torch.optim.AdamW(model.parameters(), lr=1,
                                  weight_decay=weight_decay, maximize=maximize)
        print("\tdone")
    else:
        ### load model and optimizer
        print("\nLoad model and optimizer...")
        model, optim = load_checkpoint(cfg.checkpoint_path, device)
        # update optim parameters
        for param_group in optim.param_groups:
            param_group['weight_decay'] = weight_decay 
            param_group['maximize'] = maximize
        print("\tdone")
        
          
    model = model.to(device)
    model = torch.compile(model)

    #### set up scheduler, scaler, checkpointer ####
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)
    #lambdaLR --> base_lr*custom_lr --> поэтому ставим base_lr=1 (lr=1), чтобы остался только custom_lr
    scaler = torch.amp.GradScaler("cuda") 
    checkpointer = Checkpointer(output_dir/run_name, model, model_parameters, optim, maximize_metrics)

    #### load train/valid datasets --> create DataLoaders ####
    train_dataset = AlinaDataset.load(cfg.data.train_data_path)
    valid_dataset = AlinaDataset.load(cfg.data.valid_data_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, drop_last=True,
        collate_fn=alina.make_collate(max_len, center_pad=center_pad)
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size,
        shuffle=False, drop_last=False,
        collate_fn=alina.make_collate(max_len, center_pad=center_pad)
    )

    #### start mlflow experiment ####
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(params)
    log_fn = mlflow.log_metric
    
    train(model, train_loader, valid_loader, device,
          optim, loss_fn, lr_scheduler, scaler, checkpointer, log_fn,
          log_every, valid_every, max_train_steps,
          grad_acum, clip_grad)
    print("\n---   Finished   ---")


if __name__ == "__main__":
    main()