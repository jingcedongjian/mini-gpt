from pathlib import Path
import torch
text=Path("data/input.txt").read_text(encoding="utf-8")
special_chars = {'\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?'}
chars=sorted(set(text),key=lambda c:(c in special_chars,c))
vocab_size=len(chars)
#string->integer
stoi={ch: i for i, ch in enumerate(chars)}
#integer->string
itos={i:ch for i ,ch in enumerate(chars)}
def encode(input_text:str)->list[int]:
    token_ids=[stoi[char] for char in input_text]
    return token_ids
def decode(token_ids:list[int])->str:
    output_text="".join(itos[token_id] for token_id in token_ids)
    return output_text
token_ids=encode(text)
data=torch.tensor(
    token_ids,
    dtype=torch.long,)
split_index=int(0.9 *len(data))
train_data=data[:split_index]
val_data=data[split_index:]
def get_batch(
        split:str,
        batch_size:int,
        block_size:int,)->tuple[torch.Tensor,torch.Tensor]:
    score_data = train_data if split =="train" else val_data
    start_indices=torch.randint(
        0,len(score_data)-block_size,(batch_size,),)
    x=torch.stack([score_data[start:start+block_size]
        for start in start_indices])
    y=torch.stack([score_data[start+1:start+block_size+1]
                   for start in start_indices])
    return x,y



