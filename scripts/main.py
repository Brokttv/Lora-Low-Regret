import argparse
import torch
from transformers import DistilBertForSequenceClassification
import json

from lora import setup_model
from data import get_dataloaders
from train import train, val, test
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning")
    parser.add_argument('--lrs', type=float, nargs='+', default=[1e-4, 5e-4, 1e-3], 
                        help='Learning rates to sweep (default: 1e-4 5e-4 1e-3)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--rank', type=int, default=128, help='LoRA rank')
    parser.add_argument('--alpha', type=int, default=256, help='LoRA alpha')
    parser.add_argument('--batch_size', type=int, default=32, help= 'batch size')
    parser.add_argument('--full-finetune', action='store_true', help='Use full fine-tuning instead of LoRA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_data, val_data, test_data = get_dataloaders(batch_size = args.batch_size)
    
    results = {}
    
    for lr in args.lrs:
        print(f"\n{'='*60}")
        print(f"Training with LR: {lr}")
        print(f"{'='*60}")
        
        set_seed(args.seed)
        
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)
        model.to(device)
        
        use_lora = not args.full_finetune
        trainable_params = setup_model(model, use_lora, args.rank, args.alpha, device)
        
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.001, betas=(0.9, 0.999))
        
        train_losses = []
        val_losses = []
        
        for epoch in range(args.epochs):
            train_loss = train(model, optimizer, train_data, device)
            train_losses.append(train_loss)
            
            val_loss = val(model, val_data, device)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Test accuracy after training
        test_acc = test(model, test_data, device)
        print(f"Test Accuracy: {test_acc:.4f}")
        
        results[str(lr)] = {
            "best_val_loss": min(val_losses),
            "test_acc": test_acc,
            "train_losses": train_losses,
            "val_losses": val_losses
        }
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for lr, result in results.items():
        print(f"LR {lr}: Best Val Loss = {result['best_val_loss']:.4f}, Test Acc = {result['test_acc']:.4f}")
    
    print(f"\nResults saved to results.json")


if __name__ == '__main__':
    main()
