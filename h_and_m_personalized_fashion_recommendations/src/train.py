import os
import sys
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import get_dataloaders
from models.base import HMModel
from models.metric import BCEDiceLoss
from utils import mean_average_precision, mk_empty_dir, seed_everything


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    max_epochs: int,
    critrion: nn.Module,
    evaluation: Callable,
    lr: float = 3e-4,
    val_dataloader: DataLoader = None,
    patience: int = None,
    use_cuda: bool = False,
    save_dir: str = None,
    save_filepath: str = None,
):
    if use_cuda:
        model = model.cuda()

    optimizer = get_optimizer(model, lr)
    scheduler = get_scheduler(optimizer, type="cosine")

    best_score = 0
    best_epoch = 0
    num_patience = 0
    metric = 0.0
    for epoch in range(max_epochs):
        # Train
        model.train()
        tbar = tqdm(train_dataloader, file=sys.stdout)
        loss_history = []
        for batch in tbar:
            inputs, target = batch

            if use_cuda:
                inputs = {key: val.cuda() for key, val in inputs.items()}
                target = target.cuda()

            z = model(inputs)
            loss = critrion(z, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

            loss_history.append(loss.detach().cpu().item())

            tbar.set_description(
                f"Epoch-{epoch} | train_loss={np.mean(loss_history):.6f} | lr={lr:.1e}"
            )

        # Validation
        targets = []
        preds = []
        model.eval()
        for batch in val_dataloader:
            inputs, target = batch

            if use_cuda:
                inputs = {key: val.cuda() for key, val in inputs.items()}
                target = target.cuda()

            pred = model(inputs)
            _, pred = torch.topk(z, k=12, dim=1)
            pred = pred.detach().cpu().numpy()
            preds.extend(pred)

            target = [t.nonzero().view(-1).detach().cpu().tolist() for t in target]
            targets.extend(target)

        targets = np.array(targets, dtype=object)
        preds = np.array(preds)

        metric = evaluation(targets, preds, k=12)
        log = f"map@12 of validation is {metric:.6f}"

        if best_score < metric:
            log += " [updated]"
            best_score = metric
            best_epoch = epoch
            torch.save(
                model.state_dict(), os.path.join(save_dir, save_filepath + ".pth")
            )

            num_patience += 1
            if patience is not None and num_patience > patience:
                print(log)
                break
        print(log)

    print(f"Best Score is {best_score:.6f} of Epoch-{best_epoch}")
    # NOTE: dump best metric for calculate averaging cv score.
    with open(
        os.path.join(save_dir, save_filepath + f"-metric-{best_score:.6f}"), "w"
    ) as f:
        f.write("")


def get_optimizer(net, lr):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer


def get_scheduler(optimizer, type: str = "cosine"):
    if type == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=500, eta_min=1e-5
        )
    elif type == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=500, gamma=0.8
        )
    return lr_scheduler


def main():
    config = Config()
    exp_num = config.exp_num

    seed_everything(seed=config.seed)
    save_dir = f"../data/model/{exp_num}/"
    mk_empty_dir(save_dir)

    num_splits = 5
    for num_fold in range(num_splits):
        train_dataloader, val_dataloader = get_dataloaders(
            num_fold=num_fold, batch_size=256, max_seq_len=config.max_seq_len
        )
        model = HMModel(
            article_embedding_size=config.article_embedding_size,
            max_seq_len=config.max_seq_len,
        )

        # scheduler = get_scheduler(optimizer, type="step")

        print(f"Start Training of {num_fold}-fold\n")
        train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            lr=5e-4,
            max_epochs=10,
            patience=5,
            critrion=BCEDiceLoss(),
            evaluation=mean_average_precision,
            use_cuda=torch.cuda.is_available(),
            save_dir=save_dir,
            save_filepath=f"{num_fold}-fold",
        )
        print("\nEnd Training\n\n")


if __name__ == "__main__":
    main()
