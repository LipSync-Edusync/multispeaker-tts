import torch
import torch.nn as nn
import torch.nn.functional as F

class GE2ELoss(nn.Module):
    # generalized end-2-end loss for speaker verification
    
    def __init__(self, init_w=10.0, init_b=-5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
    
    def forward(self, embeddings):
        # embs: (N, M, D) : (no. speakers, no. of utter per speak, emb dims)
        
        N, M, D = embeddings.shape
        centroids = torch.mean(embeddings, dim=1) # (N, D)
        
        # cosin sim mat
        sim_matrix = []
        for j in range(N):
            centroid = centroids[j].unsqueeze(0).expand(N, -1) # (N, D)
            sim = F.cosine_similarity(embeddings.view(-1, D), centroid.unsqueeze(1).expand(-1, M, -1).reshape(-1, D), dim=1).view(N,M)
            sim_matrix.append(sim)
        
        sim_matrix = torch.stack(sim_matrix, dim=0) # (N, N, M)
        
        sim_matrix = self.w * sim_matrix + self.b
        
        labels = torch.arange(N, device=embeddings.device)
        labels = labels.view(N,1).expand(-1, M).reshape(-1) # (N*M,)
        
        # reshape for cross-entropy
        sim_matrix = sim_matrix.view(N*N, M).transpose(0,1).reshape(-1, N) # (N*M, N)
        
        # loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


        