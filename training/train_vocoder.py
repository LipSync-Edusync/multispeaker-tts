import torch
from models.vocoder import WaveNet
from data.datasets import TTSDataset
from torch.utils.data import DataLoader
from utils.audio import AudioProcessor

def train_vocoder():
    # Configuration
    config = {
        'data_root': 'path/to/tts_dataset',
        'checkpoint_dir': 'checkpoints/vocoder',
        'batch_size': 16,
        'lr': 1e-4,
        'epochs': 100,
        'save_interval': 5,
        'audio': {
            'sample_rate': 22050,
            'num_mels': 80,
            'n_fft': 1024,
            'hop_length': 256,
            'win_length': 1024
        }
    }

    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ap = AudioProcessor(**config['audio'])
    
    # Initialize vocoder
    model = WaveNet(
        n_mels=config['audio']['num_mels'],
        residual_channels=512,
        gate_channels=512,
        skip_channels=256,
        kernel_size=3,
        n_layers=30,
        dropout=0.05
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    # Dataset and loader
    dataset = TTSDataset(config['data_root'], ap)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        
        for batch in loader:
            # Prepare inputs
            mels = batch['mel'].to(device)
            wavs = batch['wav'].to(device)  # data mod to ret waveform
            
            # Forward pass
            outputs = model(wavs, mels)
            
            # Compute loss
            loss = criterion(outputs, wavs.long())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
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
            }, f"{config['checkpoint_dir']}/vocoder_{epoch+1}.pt")

if __name__ == '__main__':
    train_vocoder()