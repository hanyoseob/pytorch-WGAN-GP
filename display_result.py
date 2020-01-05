import os
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

dir_result = './result/wgan/mnist/images'
lst_result = os.listdir(dir_result)

nx = 64
ny = 64
nch = 3

n = 8
m = 4

img = torch.zeros((n*m, ny, nx, nch))

for i in range(n*m):
    img[i, :, :, :] = torch.from_numpy(plt.imread(os.path.join(dir_result, lst_result[i]))[:, :, :nch])

img = img.permute((0, 3, 1, 2))

plt.figure(figsize=(n, m))
plt.axis("off")
# plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(img, padding=2, normalize=True), (1, 2, 0)))

plt.show()

