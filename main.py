import logging
import os
import time
import torch

from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
import tqdm

def validate(model, val_dataloader,args):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            for k,v in batch.items():
                batch[k]=v.to(args.device)
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = MultiModal(args)
    optimizer, scheduler = build_optimizer(args, model)
    # if args.device == 'cuda':
    #     # model = torch.nn.parallel.DataParallel(model.to(args.device))
    #     model.to(args.device)
    model.to(args.device)

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            for k,v in batch.items():
                batch[k]=v.to(args.device)
            loss, accuracy, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch+1} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

        # 4. validation
        loss, results = validate(model, val_dataloader,args)
        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"Epoch {epoch+1} step {step}: loss {loss:.3f}, {results}")

        # 5. save checkpoint
        mean_f1 = results['mean_f1']
        if mean_f1 > best_score:
            best_score = mean_f1
            state_dict = model.state_dict()
            torch.save({'epoch': epoch+1, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                       f'{args.savedmodel_path}/model_epoch_{epoch+1}_mean_f1_{mean_f1}.bin')


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
