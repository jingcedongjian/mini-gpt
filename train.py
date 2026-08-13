from data import get_batch,vocab_size
from model import BigramLanguageModel
import torch
from data import decode

model_1=BigramLanguageModel(vocab_size=vocab_size)
learning_rate=1e-3
traing_steps=10000

optimzer=torch.optim.AdamW(model_1.parameters(),
                           lr=learning_rate,)
for step in range(traing_steps):
    input_batch,target_batch=get_batch("train",4,8,)
    optimzer.zero_grad()
    logits,loss=model_1(input_batch,
        target_batch,)
    loss.backward()
    optimzer.step()
    if step %100==0:
        print(
            f"step:{step},loss:{loss.item():4f}"
        )

model_1.eval()
with torch.no_grad():
    input_batch,target_batch =get_batch("val",4,8,)
    logits,loss=model_1(input_batch,target_batch)
print("val loss",loss.item())
model_1.train()
input_indices=torch.zeros((1,1),dtype=torch.long,)
input_indices=model_1.generate(input_indices,max_new_tokens=200,)
print(decode(input_indices[0].tolist()))


