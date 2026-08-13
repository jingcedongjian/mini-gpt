import torch 
import torch.nn as nn
from torch.nn import functional as F
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size:int)->None:
        super().__init__()
        self.token_embedding_table=nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=vocab_size)
    def forward(self,input_indices:torch.Tensor,target_indices:torch.Tensor | None=None,)->tuple[torch.Tensor,torch.Tensor | None]:
        logits=self.token_embedding_table(input_indices)
        loss=None
        if target_indices is not None:
            batch_size, sequence_length, vocab_size = logits.shape
            logits_for_loss=logits.reshape(
                batch_size * sequence_length,vocab_size,)
            target_for_loss=target_indices.reshape(batch_size * sequence_length)
            loss=F.cross_entropy(
                logits_for_loss,
                target_for_loss,)
        return logits,loss
    @torch.no_grad()
    def generate(
        self,input_indices:torch.Tensor,
        max_new_tokens:int,)->torch.Tensor:
        for _ in range(max_new_tokens):
            logits,_=self(input_indices)
            last_token_logits=logits[:,-1,:]
            prob=F.softmax(last_token_logits,dim=-1,)
            next_token_index=torch.multinomial(
                prob,num_samples=1,
            )   
            input_indices=torch.cat((
                input_indices,next_token_index
            ),dim=1,)
        return input_indices
    