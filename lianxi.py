import torch
from data import get_batch,vocab_size,decode
from model import BigramLanguageModel
model1=BigramLanguageModel(vocab_size=65)
learning_rate=1e-3
training_steps=10000
optimizer=torch.optim.AdamW
for step in range(training_steps):
    input_batch,target_batch=get_batch("train",4,8)
    optimizer.zero_grad()
    logits,loss=model1(input_batch,target_batch)
    loss.backward()
    optimizer.step()
    if step % 100==0:
        print(f"step:{step},loss:{loss.item():.4f}")
model1.eval()
with torch.no_grad():
    input_batch, target_batch = get_batch("val", 4, 8)
    logits,loss=model1(input_batch,target_batch)
print(f"val_loss:{loss.item():.4f}")   
input_indices=torch.zeros(
    (1,1),
    dtype=torch.long,)
max_new_tokens=200
with torch.no_grad():
    input_indices=model1.generate(input_indices=input_indices,max_new_tokens=max_new_tokens)
    
    text=decode(input_indices[0].tolist())
    print(text)