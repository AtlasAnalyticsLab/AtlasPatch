import yaml
import torch
from tqdm import tqdm
from datetime import datetime as dt
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from scripts.dataset import SegmentationDataset
from torch.utils.tensorboard import SummaryWriter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scripts.utils import SaveModel, DiceBCELoss, EarlyStopping


if __name__ == '__main__':

    with open("/mnt/SDA/SegmentationProject/SlideProcessor/SAM2_finetuning/config/local_config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Assign parameters from config
    train_data_path = config['train_data_path']
    validation_data_path = config['validation_data_path']
    lambda_value = config['lambda_value']
    BATCH_SIZE = config['batch_size']
    NUM_EPOCHS = config['num_epochs']
    LEARNING_RATE = config['learning_rate']
    WEIGHT_DECAY = config['weight_decay']
    NUM_EXP = config['NUM_EXP']
    sam2_checkpoint_path = config['sam2_checkpoint_path']
    model_cfg = config['model_cfg']

    custom_log_dir = f"logs/experiment_sam2_layernorm{NUM_EXP}/" + dt.now().strftime("%Y%m%d-%H%M%S")
    saved_model_name = f"trained_sam2_layernorm{NUM_EXP}"

    # Select device
    if torch.backends.mps.is_available(): 
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print(f"Device: {DEVICE}")

    print(f"Using {torch.cuda.device_count()} GPUs!")

    # Dataset & DataLoader setup
    images_transform = v2.Compose([
        v2.Resize(size=(256, 256)),
        v2.ToTensor(), #sam2 normalizes images internally
    ])
    masks_transform = v2.Compose([
        v2.Resize(size=(256, 256)),
        v2.ToTensor(),
    ])

    train_dataset = SegmentationDataset(data_root=train_data_path,
                                            image_transform=images_transform, mask_transform=masks_transform)
    validation_dataset = SegmentationDataset(data_root=validation_data_path,
                                                image_transform=images_transform, mask_transform=masks_transform)

    train_dataloader = DataLoader(train_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=2,
                                    drop_last=True,
                                    pin_memory=True)
    val_dataloader = DataLoader(validation_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=2,
                                drop_last=True,
                                pin_memory=True)

    predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint_path))

    # if torch.cuda.device_count() > 1:
    #     predictor.model = torch.nn.DataParallel(predictor.model)
    
    predictor.model.to(DEVICE)

    
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)
    predictor.model.image_encoder.train(True)

    total_params = sum(p.numel() for p in predictor.model.parameters())
    trainable_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    for name, param in predictor.model.named_parameters():
        # Check if the parameter belongs to a layer normalization layer
        if "norm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Confirm the trainable parameters
    print("Updated model parameters:")
    print("Number of model's trainable parameters: ",
        sum(p.numel() for p in predictor.model.parameters() if p.requires_grad))  # Trainable parameters

    
    writer = SummaryWriter(log_dir=custom_log_dir, flush_secs=30)
    early_stopping = EarlyStopping(patience=5, verbose=True)

    loss_fn = DiceBCELoss(gamma=lambda_value)
    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    predictor.model.to(DEVICE)

    epoch_iterator = tqdm(range(NUM_EPOCHS), desc="Epochs", unit="epoch")
    for epoch in epoch_iterator:
        predictor.model.train()
        train_loss = 0

        train_iterator = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]",
            leave=False,
            unit="batch",
        )

        for image_batch, mask_batch, bbox_batch in train_iterator:
            optimizer.zero_grad()

            images = [img.permute(1, 2, 0).cpu().numpy() for img in image_batch]
            masks = mask_batch.to(DEVICE)
            bboxes = [bbox if len(bbox.shape) > 1 else bbox[None, None, :] for bbox in bbox_batch]

            if masks.sum() == 0:  # Skip empty masks
                continue

            # image encoding
            predictor.set_image_batch(images)

            num_images = len(predictor._features["image_embed"])
           
            _, _, _, unnorm_box = predictor._prep_prompts(
            point_coords=None, point_labels=None, box=torch.stack(bboxes).to(DEVICE) if bboxes else None, mask_logits=None, normalize_coords=False
                )
            
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=None, boxes=unnorm_box.to(DEVICE), masks=None
                )
            
            high_res_features = [feat_level[-1].unsqueeze(0).to(DEVICE) for feat_level in predictor._features["high_res_feats"]]

            low_res_masks, iou_predictions, _, _ = predictor.model.sam_mask_decoder(
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

            train_iterator.set_postfix(batch_loss=loss.item())


        scheduler.step()
        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        predictor.model.eval()
        val_loss = 0
        with torch.inference_mode():
            val_iterator = tqdm(
                val_dataloader,
                desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]",
                leave=False,
                unit="batch",
            )

            for image_batch, mask_batch, bbox_batch in val_iterator:
                images = [img.permute(1, 2, 0).cpu().numpy() for img in image_batch]
                masks = mask_batch.to(DEVICE)
                bboxes = [bbox if len(bbox.shape) > 1 else bbox[None, None, :] for bbox in bbox_batch]

                if masks.sum() == 0:
                    continue

                predictor.set_image_batch(images)

                num_images = len(predictor._features["image_embed"])
           
                _, _, _, unnorm_box = predictor._prep_prompts(
                point_coords=None, point_labels=None, box=torch.stack(bboxes).to(DEVICE) if bboxes else None, mask_logits=None, normalize_coords=False
                    )
                
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=None, boxes=unnorm_box.to(DEVICE), masks=None
                    )
                
                high_res_features = [feat_level[-1].unsqueeze(0).to(DEVICE) for feat_level in predictor._features["high_res_feats"]]

                low_res_masks, iou_predictions, _, _ = predictor.model.sam_mask_decoder(
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

                val_iterator.set_postfix(batch_loss=loss.item())

            avg_val_loss = val_loss / len(val_dataloader)

        writer.add_scalars('Loss', {'train': avg_train_loss, 'validation': avg_val_loss}, epoch + 1)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        epoch_iterator.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss)

        early_stopping(avg_val_loss)
        if early_stopping.should_stop:
            print("Early stopping triggered. Stopping training...")
            break

    writer.close()


    SaveModel(predictor.model, optimizer, NUM_EPOCHS, saved_model_name)
