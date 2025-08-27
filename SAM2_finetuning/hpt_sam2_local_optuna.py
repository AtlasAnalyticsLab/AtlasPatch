import torch
import yaml
import optuna
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scripts.dataset import SegmentationDataset
from scripts.utils import SaveModel, DiceBCELoss, EarlyStopping

# --------------------- Training Function ---------------------
def train(predictor, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, num_epochs, device, log_dir):
    writer = SummaryWriter(log_dir=log_dir, flush_secs=30)
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(num_epochs):
        predictor.model.train()
        train_loss = 0

        for image_batch, mask_batch, bbox_batch in train_dataloader:
            optimizer.zero_grad()

            images = [img.permute(1, 2, 0).cpu().numpy() for img in image_batch]
            masks = mask_batch.to(device)
            bboxes = [bbox if len(bbox.shape) > 1 else bbox[None, None, :] for bbox in bbox_batch]

            if masks.sum() == 0:
                continue

            predictor.set_image_batch(images)
            _, _, _, unnorm_box = predictor._prep_prompts(None, None, torch.stack(bboxes).to(device), None, False)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(None, unnorm_box.to(device), None)
            high_res_features = [feat_level[-1].unsqueeze(0).to(device) for feat_level in predictor._features["high_res_feats"]]

            low_res_masks, _, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"],
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
            loss = loss_fn(prd_masks, masks)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        predictor.model.eval()
        val_loss = 0
        with torch.inference_mode():
            for image_batch, mask_batch, bbox_batch in val_dataloader:
                images = [img.permute(1, 2, 0).cpu().numpy() for img in image_batch]
                masks = mask_batch.to(device)
                bboxes = [bbox if len(bbox.shape) > 1 else bbox[None, None, :] for bbox in bbox_batch]

                if masks.sum() == 0:
                    continue

                predictor.set_image_batch(images)
                _, _, _, unnorm_box = predictor._prep_prompts(None, None, torch.stack(bboxes).to(device), None, False)
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(None, unnorm_box.to(device), None)
                high_res_features = [feat_level[-1].unsqueeze(0).to(device) for feat_level in predictor._features["high_res_feats"]]
                low_res_masks, _, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"],
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )

                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                loss = loss_fn(prd_masks, masks)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)

        writer.add_scalars('Loss', {'train': avg_train_loss, 'validation': avg_val_loss}, epoch + 1)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        early_stopping(avg_val_loss)
        if early_stopping.should_stop:
            print("Early stopping triggered.")
            break

    writer.close()

# --------------------- Evaluation Function ---------------------
def evaluate_f1(predictor, val_dataloader, device):
    predictor.model.eval()
    all_preds, all_targets = [], []

    with torch.inference_mode():
        for image_batch, mask_batch, bbox_batch in val_dataloader:
            images = [img.permute(1, 2, 0).cpu().numpy() for img in image_batch]
            masks = mask_batch.to(device)
            bboxes = [bbox if len(bbox.shape) > 1 else bbox[None, None, :] for bbox in bbox_batch]

            if masks.sum() == 0:
                continue

            predictor.set_image_batch(images)
            _, _, _, unnorm_box = predictor._prep_prompts(None, None, torch.stack(bboxes).to(device), None, False)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(None, unnorm_box.to(device), None)
            high_res_features = [feat_level[-1].unsqueeze(0).to(device) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, _, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"],
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
            preds = (prd_masks > 0.5).int().cpu().numpy().flatten()
            targets = masks.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_targets.extend(targets)

    return f1_score(all_targets, all_preds, zero_division=1)

# --------------------- Main Optuna Objective ---------------------
def objective(trial):
    lambda_value = trial.suggest_float("lambda", 0.7, 1.0)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])

    loss_fn = DiceBCELoss(gamma=lambda_value)
    optimizer = torch.optim.AdamW(predictor.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    log_dir = f"logs/optuna_trial_{trial.number}/"
    start = timer()
    train(predictor, train_loader, val_loader, optimizer, scheduler, loss_fn, NUM_EPOCHS, DEVICE, log_dir)
    elapsed = timer() - start

    val_f1 = evaluate_f1(predictor, val_loader, DEVICE)
    trial.set_user_attr("training_time_min", elapsed / 60)
    return val_f1

# --------------------- Execution Entry ---------------------
if __name__ == '__main__':
    with open("config/hpt_local_config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)

    train_data_path = config['train_data_path']
    validation_data_path = config['validation_data_path']
    NUM_EPOCHS = config['num_epochs']
    WEIGHT_DECAY = config['weight_decay']
    sam2_checkpoint_path = config['sam2_checkpoint_path']
    model_cfg = config['model_cfg']

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    images_transform = v2.Compose([v2.Resize((256, 256)), v2.ToTensor()])
    masks_transform = v2.Compose([v2.Resize((256, 256)), v2.ToTensor()])

    train_dataset = SegmentationDataset(train_data_path, images_transform, masks_transform)
    validation_dataset = SegmentationDataset(validation_data_path, images_transform, masks_transform)

    predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint_path))
    predictor.model.to(DEVICE)

    for name, param in predictor.model.named_parameters():
        param.requires_grad = "norm" in name

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)

    print("\nBest Trial:")
    trial = study.best_trial
    print(f"  F1-score: {trial.value:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
