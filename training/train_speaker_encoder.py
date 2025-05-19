import torch
import sys
from pathlib import Path
import logging
import os
import argparse
import soundfile as sf
from models import SpeakerEncoder, GE2ELoss
from data.datasets import SpeakerVerificationDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from utils.audio import AudioProcessor

sys.path.append(str(Path(__file__).parent.parent))
from __init__ import logger

def train_speaker_encoder(args):
    # logger = setup_logger()

    # Configuration
    config = {
        'data_root': args.data_root,
        'checkpoint_dir': args.checkpoint_dir,
        'num_utterances': args.num_utterances,
        'num_speakers': args.num_speakers,
        'batch_size': args.num_speakers * args.num_utterances,
        'lr': args.lr,
        'epochs': args.epochs,
        'save_interval': 5,
        'audio': {
            'sample_rate': 16000,
            'num_mels': 40,
            'n_fft': 1024,
            'hop_length': 256,
            'win_length': 1024
        }
    }

    # Ensure checkpoint directory exists
    try:
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        logger.info(f"Using checkpoint directory: {config['checkpoint_dir']}")
    except Exception as e:
        logger.error(f"Could not create checkpoint directory: {e}")
        return

    # Initialize components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    try:
        ap = AudioProcessor(**config['audio'])
        model = SpeakerEncoder().to(device)
        criterion = GE2ELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        return

    # Load dataset
    try:
        logger.info(f"Loading dataset from {config['data_root']}")
        if not os.path.exists(config['data_root']):
            raise FileNotFoundError(f"Data root {config['data_root']} does not exist.")
        
        dataset = SpeakerVerificationDataset(
            data_root=config['data_root'],
            audio_processor=ap,
            num_speakers=config['num_speakers'],
            num_utterances=config['num_utterances']
        )

        
        logger.debug(f"N: {dataset.num_speakers} | M: {len(dataset)}")
        
        # batch_sampler = BatchSampler(
        #     range(len(dataset)),
        #     batch_size=config['batch_size'],
        #     drop_last=True
        # )
        
        loader = DataLoader(
            dataset,
            batch_size=config['num_speakers'],  # 1 batch = N speakers
            shuffle=True,
            collate_fn=dataset.collate_fn,
            drop_last=True
        )
        
        logger.debug(f"====Dataset size: {len(loader.dataset)} | Num batches: {len(loader)}")
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        return

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            try:
                mels = batch['mel'].to(device)
                N, M = mels.size(0), mels.size(1)
                if N < 2:
                    logger.warning(f"Number of Speakers {N} is less than 2, skipping batch.")
                    # raise ValueError("Speakers size is less than 2")
                    exit(0)
                
                if M < 3:
                    logger.warning(f"Number of utterances {M} is less than 3, skipping batch.")
                    # raise ValueError("Number of utterances is less than 3")
                    exit(0)
                
                mels = mels.view(N * M, -1, config['audio']['num_mels'])

                embeddings = model(mels).view(N, M, -1)
                
                # test
                logger.debug(f"Batch size: {embeddings.size()}")
                logger.debug(f"Embeddings shape: {embeddings.shape}")
                
                loss = criterion(embeddings)
                
                # test
                logger.debug(f"Loss: {loss.item()}")
                logger.debug(f"Loss shape: {loss.shape}")
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()

                total_loss += loss.item()
                
                # test
                logger.info(f"Epoch {epoch+1}/{config['epochs']} - Batch {num_batches+1} - Loss: {loss.item():.4f} - Total loss: {total_loss:.4f}")
                
                num_batches += 1
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                # continue
                exit(0)

        scheduler.step()
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{config['epochs']} - Loss: {avg_loss:.4f}")
        else:
            logger.warning(f"Epoch {epoch+1}: no successful batches")

        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            ckpt_path = os.path.join(config['checkpoint_dir'], f"se_{epoch+1}.pt")
            try:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'config': config
                }, ckpt_path)
                logger.info(f"Checkpoint saved to {ckpt_path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--num_speakers', type=int, default=64)
    parser.add_argument('--num_utterances', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    try:
        train_speaker_encoder(args)
    except KeyboardInterrupt:
        print("Training interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)