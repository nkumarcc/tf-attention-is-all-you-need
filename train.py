import torch
import pickle
import secrets
import os
import argparse
from datetime import date

import logging

from transformer import LabelSmoothingLoss, Transformer, get_optimizer_and_scheduler
from dataloader import get_iters, get_vocab_transform, get_dataloader
from dataloader.constants import SRC_LANGUAGE, TGT_LANGUAGE, PAD, PAD_IDX


def generate_hash(length):
    return secrets.token_hex(length)[:length]

def get_accuracy(output, target, pad_idx):
    output_flatten = output.view(-1, output.shape[-1])
    target_flatten = target[:, 1:].reshape(-1)
    non_padding_mask = target_flatten.ne(pad_idx)
    correct = output_flatten.argmax(1).eq(target_flatten).masked_select(non_padding_mask).sum().item()
    total = non_padding_mask.sum().item()
    return correct / total

def run_training(
    path_to_data,
    num_epochs=30,
    batch_size=16,
    gradient_accumulation_steps=4,
    is_small=False
):
    model_id = generate_hash(6)
    today = date.today()
    filepath = f'./weights_and_metrics/{today}_{model_id}'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'./logs/{today}_{model_id}.log'),
            logging.StreamHandler()
        ],
    )

    # Get the iterators
    train_iter, val_iter, _ = get_iters(path_to_data)

    # Get the tokenizers and vocab
    token_transform, vocab_transform = get_vocab_transform(train_iter)

    # Get the dataloaders
    train_dataloader, small_train_dataloader = get_dataloader(batch_size, train_iter, token_transform, vocab_transform)
    val_dataloader, small_val_dataloader = get_dataloader(batch_size, val_iter, token_transform, vocab_transform)

    if is_small:
        train_dataloader = small_train_dataloader
        val_dataloader = small_val_dataloader

    # Initialize the model
    d_model = 512
    src_vocab_size = len(vocab_transform[SRC_LANGUAGE])
    tgt_vocab_size = len(vocab_transform[TGT_LANGUAGE])
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model=d_model)    
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    logging.info(f'Model initialized. Moving to {device}.')

    # Define loss function and optimizer
    criterion = LabelSmoothingLoss(2, vocab_transform[TGT_LANGUAGE][PAD], 0.1)
    optimizer, scheduler = get_optimizer_and_scheduler(model, model_dim=d_model, lr_mul=0.5)

    # Lists to store loss, accuracy, and BLEU values for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    grad_acc_counter = 0

    for epoch in range(num_epochs):
        model.train()  # set model to training mode
        total_train_loss = 0
        total_train_acc = 0

        # Iterate over the training data
        for i, (src, tgt) in enumerate(train_dataloader):
            # Tokenize and numericalize the source and target sentences
            src = src.to(device)
            tgt = tgt.to(device)

            # Forward pass
            output = model(src, tgt[:, :-1])

            # Compute loss
            loss = criterion(output.view(-1, tgt_vocab_size), tgt[:, 1:].reshape(-1))

            # Compute accuracy
            train_acc = get_accuracy(output, tgt, PAD_IDX)

            # Backward pass
            loss.backward()

            grad_acc_counter += 1
            if grad_acc_counter % gradient_accumulation_steps == 0:
                # Update weights
                optimizer.step()
                scheduler.step()
                # Zero the gradients
                optimizer.zero_grad()

            total_train_loss += loss.item()
            total_train_acc += train_acc

        # Compute the average loss and accuracy for this epoch
        average_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(average_train_loss)
        average_train_acc = total_train_acc / len(train_dataloader)
        train_accuracies.append(average_train_acc)

        logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {average_train_loss}, Train Accuracy: {average_train_acc}')
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'{filepath}/transformer_epoch_{epoch}.pth')

        # Validation phase
        model.eval()  # set model to evaluation mode
        total_val_loss = 0
        total_val_acc = 0

        with torch.no_grad():
            for i, (src, tgt) in enumerate(val_dataloader):
                # Tokenize and numericalize the source and target sentences
                src = src.to(device)
                tgt = tgt.to(device)

                # Forward pass
                output = model(src, tgt[:, :-1])

                # Compute loss
                loss = criterion(output.view(-1, tgt_vocab_size), tgt[:, 1:].reshape(-1))

                # Compute accuracy
                val_acc = get_accuracy(output, tgt, PAD_IDX)

                total_val_loss += loss.item()
                total_val_acc += val_acc

        # Compute the average loss and accuracy for this epoch
        average_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(average_val_loss)
        average_val_acc = total_val_acc / len(val_dataloader)
        val_accuracies.append(average_val_acc)

        logging.info(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {average_val_loss}, Val Accuracy: {average_val_acc}')

        if epoch % 5 == 0:
            with open(f'{filepath}/losses_accuracies_{epoch}.pkl', 'wb') as f:
                pickle.dump({
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies,
                }, f)
    

    torch.save(model.state_dict(), f'{filepath}/transformer_final.pth')
    with open(f'{filepath}/losses_accuracies_final.pkl', 'wb') as f:
        pickle.dump({
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
        }, f)

    logging.info(f'Completed training!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with the specified parameters.")
    
    # Add arguments
    parser.add_argument('--path_to_data', type=str, required=True, 
                        help='Path to the data for training.')
    parser.add_argument('--num_epochs', type=int, default=30, 
                        help='Number of epochs for training. Default is 30.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training. Default is 64.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of gradient accumulation steps. Default is 4.')
    parser.add_argument('--is_small', action='store_true', 
                        help='Flag indicating if the data is small. No value needed.')
    
    args = parser.parse_args()

    if args.batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError('Batch size must be divisible by gradient accumulation steps.')
    
    run_training(
        args.path_to_data,
        args.num_epochs,
        int(args.batch_size / args.gradient_accumulation_steps),
        args.gradient_accumulation_steps,
        args.is_small
    )