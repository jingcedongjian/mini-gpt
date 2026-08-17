import torch

from data import get_batch, vocab_size, decode
from model import BigramLanguageModel


model_1 = BigramLanguageModel(vocab_size=vocab_size)

learning_rate = 1e-3
training_steps = 10000

optimizer = torch.optim.AdamW(
    model_1.parameters(),
    lr=learning_rate,
)

# 训练
model_1.train()

for step in range(training_steps):
    input_batch, target_batch = get_batch("train", 4, 8)

    optimizer.zero_grad()

    logits, loss = model_1(
        input_batch,
        target_batch,
    )

    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step:{step}, loss:{loss.item():.4f}")


# 验证
model_1.eval()

with torch.no_grad():
    input_batch, target_batch = get_batch("val", 4, 8)
    logits, loss = model_1(input_batch, target_batch)

print(f"val loss:{loss.item():.4f}")


# 生成文字
input_indices = torch.zeros(
    (1, 1),
    dtype=torch.long,
)

with torch.no_grad():
    input_indices = model_1.generate(
        input_indices,
        max_new_tokens=200,
    )

print(decode(input_indices[0].tolist()))