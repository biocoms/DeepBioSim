import numpy as np
import torch
import diffusion_model
import matplotlib.pyplot as plt

sampling_timesteps=50
model = diffusion_model.DiffusionModel(input_dim=784, sampling_timesteps=sampling_timesteps).cuda()

checkpoint_path = 'exp/checkpoint_20.pt'
model.load(checkpoint_path)

# Generate 100 images.
batch_size = 100
noise = torch.randn([batch_size, 1, 28, 28])
# samples have shape [batch_size, sampling_timesteps + 1, 1, 28, 28]
samples = model.generate(noise, return_all_timesteps=True).cpu().numpy()

for s in range(0, sampling_timesteps+1, 10):

    plt.figure(s+1)
    plt.clf()
    img = np.zeros([10*28, 10*28])
    for i in range(10):
        for j in range(10):
            img[i*28:(i+1)*28, j*28:(j+1)*28] = samples[i*10+j, s, 0, :, :]
    plt.imshow(img, cmap='gray')
    plt.title(f'generation step {s}')
    plt.pause(0.5)
plt.show()