import torch
from models.synthesizer import Synthesizer
from models.speaker_encoder import SpeakerEncoder
from data.datasets import TTSDataset
from torch.utils.data import DataLoader
from utils.audio import AudioProcessor

def train_synthesizer():
    # Configuration
    config = {
        'data_root': 'path/to/tts_dataset',
        'checkpoint_dir': 'checkpoints/synthesizer',
        'speaker_encoder_path': 'checkpoints/speaker_encoder/best.pt',
        'batch_size': 32,
        'lr': 1e-3,
        'epochs': 200,
        'save_interval': 10,
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
    
    # Load speaker encoder
    speaker_encoder = SpeakerEncoder().to(device)
    speaker_encoder.load_state_dict(torch.load(config['speaker_encoder_path'])['model'])
    speaker_encoder.eval()
    
    # Initialize synthesizer
    model = Synthesizer(
        num_chars=256,  # Size of character vocabulary
        n_mels=config['audio']['num_mels'],
        speaker_embed_dim=256
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Dataset and loader
    dataset = TTSDataset(config['data_root'], ap)
    collate_fn = dataset.get_collate_fn()
    loader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=collate_fn
    )

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        
        for batch in loader:
            # Prepare inputs
            text = batch['text'].to(device)
            mels = batch['mel'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            
            # Get speaker embeddings
            with torch.no_grad():
                speaker_embeds = speaker_encoder(mels.transpose(1, 2))
            
            # Forward pass
            outputs = model(text, speaker_embeds, mels)
            
            # Compute loss
            loss = model.compute_loss(
                outputs, 
                {'mels': mels, 'stop_targets': batch['stop_targets'].to(device)},
                text_lengths,
                mel_lengths
            )['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            }, f"{config['checkpoint_dir']}/synthesizer_{epoch+1}.pt")

if __name__ == '__main__':
    train_synthesizer()