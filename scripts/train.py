"""Main training loop for multimodal sentiment model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.data.collate import collate_fn
from src.data.dataset import MSP_Podcast_Dataset
from src.models.fusion_net import CCMTModel
from src.utils.helpers import save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CCMT model")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_dataset = MSP_Podcast_Dataset(
        audio_root=Path(cfg['data']['dataset_root']) / 'Audios' / 'Audios',
        labels_csv=Path(cfg['data']['dataset_root']) / 'Labels' / 'labels_consensus.csv',
        transcripts_en_file="data/transcripts/transcripts_en.json",
        transcripts_fr_file="data/cache/translations_en_ro.json",
        partition=cfg['data']['train_partition'],
        target_sample_rate=cfg['data']['sample_rate'],
        apply_telephony_aug=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = CCMTModel(
        audio_backbone=cfg['model']['audio_backbone'],
        text_en_backbone=cfg['model']['text_en_backbone'],
        text_fr_backbone=cfg['model'].get('text_fr_backbone', 'camembert-base'),
        num_classes=cfg['model']['num_classes'],
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'],
    )

    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(cfg['training']['num_epochs']):
        for batch in train_loader:
            audio = batch['audio'].to(args.device)
            labels = batch['labels'].to(args.device)
            text_en = batch['text_en']
            text_fr = batch['text_fr']

            logits = model(audio, text_en, text_fr)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
            optimizer.step()

        if (epoch + 1) % cfg['checkpoint']['save_frequency'] == 0:
            save_path = Path(cfg['checkpoint']['save_dir']) / f"ccmt_epoch_{epoch+1}.pt"
            save_checkpoint({'epoch': epoch + 1, 'model_state': model.state_dict()}, str(save_path))


if __name__ == "__main__":
    main()
