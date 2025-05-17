import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models import SpeakerEncoder, GE2ELoss
from data.datasets.speaker_verification import SpeakerVerificationDataset
from utils.audio import AudioProcessor

def train_speaker_encoder(config):
    # Initialize components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeakerEncoder().to(device)
    criterion = GE2ELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=0.5)
    
    # Data loading
    ap = AudioProcessor(**config['audio'])
    dataset = SpeakerVerificationDataset(config['data_path'], ap, config['num_utterances'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    
    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        
        for batch in loader:
            # Get batch of utterances (N speakers, M utterances each)
            mels = batch['mel'].to(device)  # (N*M, T, D)
            N = config['num_speakers']
            M = config['num_utterances']
            mels = mels.view(N, M, -1, config['audio']['num_mels'])
            
            # Forward pass
            embeddings = []
            for i in range(N):
                speaker_embeddings = []
                for j in range(M):
                    emb = model(mels[i, j])
                    speaker_embeddings.append(emb)
                speaker_embeddings = torch.stack(speaker_embeddings, dim=0)
                embeddings.append(speaker_embeddings)
            embeddings = torch.stack(embeddings, dim=0)  # (N, M, D)
            
            # Compute loss
            loss = criterion(embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        # Logging
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, f"{config['checkpoint_dir']}/se_{epoch+1}.pt")