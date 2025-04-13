import argparse, os, shutil, torch, wandb, tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from model_mbert_wqrm import MBertWQRM, create_dataloader

def main():
    parser = argparse.ArgumentParser(description='Train ModernBERT classifier')
    parser.add_argument('--model', type=str, default="answerdotai/ModernBERT-large")
    parser.add_argument('--train_fn', type=str, default="data/lamp_PR_train.json")
    parser.add_argument('--val_fn', type=str, default="data/lamp_PR_val.json")
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_every', type=int, default=20)
    
    args = parser.parse_args()

    args.optim_every = 1
    if args.batch_size > 8:
        args.optim_every = args.batch_size // 8
        args.batch_size = (args.batch_size // args.optim_every)
        print(f'Batch size set to {args.batch_size} and optim_every set to {args.optim_every}')

    model_suffix = "PR" if "_PR_" in args.train_fn else "P" if "_P_" in args.train_fn else "R"

    wandb.init(
        project="writing-rewards-nlu",
        config={"train_fn": args.train_fn, "val_fn": args.val_fn, "learning_rate": args.learning_rate, "max_grad_norm": args.max_grad_norm, "epochs": args.epochs, "batch_size": args.batch_size, "eval_every": args.eval_every, "model": args.model, "optim_every": args.optim_every}
        )
    wandb.run.name = f'{model_suffix}-lr{args.learning_rate:.1e}-bs{args.batch_size}-oe{args.optim_every}'

    model = MBertWQRM(args.model)
    train_loader = create_dataloader(args.train_fn, args.batch_size)
    val_loader = create_dataloader(args.val_fn, 100)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=len(train_loader) // 10,  # Add warmup steps (10% of first epoch)
        num_training_steps=len(train_loader) * args.epochs
    )

    best_loss = 100.0 # nothing above 0.65 needs to be saved
    for epoch in range(args.epochs):
        for batch_idx, (paragraphs1, paragraphs2, rationales, labels_cls, labels_reg) in enumerate(tqdm.tqdm(train_loader)):
            outputs = model(paragraphs1, paragraphs2, rationales)
            
            cls_logits, reg_logits = outputs
            labels_cls = labels_cls.to(cls_logits.device)
            labels_reg = labels_reg.to(reg_logits.device)

            n_reg, n_cls = 0, 0
            loss_cls, loss_reg = 0, 0
            for label_cls, label_reg, cls_logit, reg_logit in zip(labels_cls, labels_reg, cls_logits, reg_logits):
                if label_cls != -1: # it's classification
                    n_cls += 1
                    loss_item = -torch.log(cls_logit) if label_cls == 0 else -torch.log(1 - cls_logit)
                    loss_cls += loss_item
                else:
                    n_reg += 1
                    loss_item = (reg_logit - label_reg).pow(2)
                    loss_reg += loss_item

            loss_cls = loss_cls / (n_cls + 1e-8)
            loss_reg = (loss_reg / (n_reg + 1e-8)) / 4.0 # just to scale it to the cls loss

            # Combined loss
            if n_reg > 0 and n_cls > 0:
                loss = loss_cls + loss_reg
            elif n_reg > 0:
                loss = loss_reg
            else:
                loss = loss_cls

            # Scale loss for gradient accumulation
            loss = loss / args.optim_every
            
            train_log = {'train/loss': loss.item(), 'train/n_reg': n_reg, 'train/n_cls': n_cls}
            if n_reg > 0:
                train_log["train/loss_reg"] = loss_reg.item()
            if n_cls > 0:
                train_log["train/loss_cls"] = loss_cls.item()
            wandb.log(train_log)
            
            loss.backward()

            # Only optimize every optim_every steps
            if (batch_idx + 1) % args.optim_every == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if epoch >= 1 and (batch_idx + 1) % args.eval_every == 0:
                val_metrics = model.evaluate(val_loader)
                wandb.log({'val/loss': val_metrics["loss_total"], 'val/loss_cls': val_metrics["loss_cls"], 'val/loss_reg': val_metrics["loss_reg"], 'val/accuracy': val_metrics["acc"], 'val/mse': val_metrics["mse"], 'val/mae': val_metrics["mae"], 'val/corr': val_metrics["val_corr"], 'val/N_cls': val_metrics["N_cls"], 'val/N_reg': val_metrics["N_reg"], "val/best_loss": best_loss})

                print(f'Epoch {epoch+1}/{args.epochs}, Val Loss: {val_metrics["loss_total"]:.4f} (cls: {val_metrics["loss_cls"]:.4f} + reg: {val_metrics["loss_reg"]:.4f}) acc: {val_metrics["acc"]:.4f}; mse: {val_metrics["mse"]:.4f}; mae: {val_metrics["mae"]:.4f}; corr: {val_metrics["val_corr"]:.4f})')
                if val_metrics["loss_total"] < min(best_loss, 0.60):
                    # Delete previous best model folder if it exists
                    if hasattr(model, 'best_model_dir') and os.path.exists(model.best_model_dir):
                        shutil.rmtree(model.best_model_dir)
                    
                    best_loss = val_metrics["loss_total"]
                    save_dir = f'models/mbert2-large-{model_suffix}-loss{best_loss:.3f}'
                    print(f'\033[94mSaving model to {save_dir}\033[0m')
                    model.best_model_dir = save_dir
                    model.save_model(save_dir)  # Use the new save_model method
                model.train()

    wandb.finish()

if __name__ == '__main__':
    main()
