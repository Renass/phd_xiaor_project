import torch
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, ExponentialLR
import matplotlib.pyplot as plt

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1280*7*7, 768),
            torch.nn.GELU(),
            torch.nn.Linear(768, 768))
        
model = Model()
model.train()
optimizer = torch.optim.AdamW(model.parameters())
scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=20)
scheduler2 = ExponentialLR(optimizer, gamma=0.9)
scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[2])

learning_rates = []
for epoch in range(100):
    optimizer.step()
    learning_rates.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(learning_rates, label='Learning Rate')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Combined Learning Rate Schedule')
plt.legend()
plt.grid(True)
plt.show()