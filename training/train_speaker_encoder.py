import torch
import argparse
from models import SpeakerEncoder, GE2ELoss
from data.datasets import SpeakerVerificationDataset
from torch.utils.data import DataLoader
from utils.audio import AudioProcessor

def train_speaker_encoder(args):
    # Configuration
    config = {
        'data_root': args.data_root,
        'checkpoint_dir': args.checkpoint_dir,
        'batch_size': args.batch_size,
        'num_utterances': args.num_utterances,
        'num_speakers': args.batch_size,
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

    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ap = AudioProcessor(**config['audio'])
    model = SpeakerEncoder().to(device)
    criterion = GE2ELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Dataset and loader
    dataset = SpeakerVerificationDataset(
        config['data_root'],
        ap,
        num_utterances=config['num_utterances']
    )
    loader = DataLoader(dataset, batch_size=config['num_speakers'], shuffle=True)

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        
        for batch in loader:
            # Prepare batch: [num_speakers, num_utterances, mel_len, n_mels]
            mels = batch['mel'].to(device)
            N, M = mels.size(0), mels.size(1)
            mels = mels.view(N * M, -1, config['audio']['num_mels'])
            
            # Forward pass
            embeddings = model(mels)
            embeddings = embeddings.view(N, M, -1)
            
            # Compute loss
            loss = criterion(embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # Logging
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': config
            }, f"{config['checkpoint_dir']}/se_{epoch+1}.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_utterances', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    train_speaker_encoder(args)