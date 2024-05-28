import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import SimpleITK as sitk
from topologylayer.nn import SumBarcodeLengths, PartialSumBarcodeLengths
from topologylayer.nn import LevelSetLayer2D

from torch_topological.nn import CubicalComplex
from torch_topological.nn import WassersteinDistance

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class complex_comp(nn.Module):
    def __init__(self, size, sublevel=True):
        super(complex_comp, self).__init__()
        self.pdfn = LevelSetLayer2D(size=size, maxdim=1, sublevel=sublevel, alg='hom')

    def forward(self, img):
        dgm = self.pdfn(img)
        return dgm

def plot_persistence_diagram(persistence, output_file_name="output"):
    """Plots the persistence diagram."""
    plt.clf()
    gd.plot_persistence_diagram(persistence)
    plt.title("Persistence Diagram")
    plt.savefig(output_file_name + "_diagram.png")


image_data = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0.4, 0, 0.2, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

image_data_2 = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0.8, 0, 0.6, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

image_data_3 = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0.8, 0, 0.6, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

image_list = [image_data, image_data_2, image_data_3]
# image_list = [image_data]
combined_images = np.array(image_list)

# Convert the numpy array to a PyTorch tensor
img_batch = torch.tensor(combined_images, device=device, dtype=torch.float32).unsqueeze(1)

pdfn = LevelSetLayer2D(size=(6, 6), maxdim=1, sublevel=True, alg='hom')
dgms = []
for img in img_batch:
    dgm = pdfn(1 - img)
    dgm_persistence = []
    for i in range(len(dgm)):
        cpu_dgm = dgm[i].cpu().detach().numpy()
        for j in range(len(cpu_dgm)):
            if cpu_dgm[j][0] != cpu_dgm[j][1]:
                dgm_persistence.append((i, (cpu_dgm[j][0], cpu_dgm[j][1])))
    dgms.append(dgm_persistence)

print("topolayer:", dgms)
plot_persistence_diagram(dgm_persistence, "topo")

persistences = []
for img in img_batch:
    cc = gd.CubicalComplex(
        dimensions=(6, 6), top_dimensional_cells=1-img.flatten()
    )
    persistence = cc.persistence()
    persistences.append(persistence)

print("gudhi:", persistences)
# cofaces = cc.cofaces_of_persistence_pairs()

# for idx, (birth, death) in enumerate(persistence):
#     if death[1] == float("inf"):
#         persistence[idx] = (birth, (death[0], image_data.max()))

# print("gudhi:", persistence)
# print("cofaces:", cofaces)
plot_persistence_diagram(persistence, "cubical")

# cubical = CubicalComplex()
# wd_loss = WassersteinDistance(q=2)
# per_cc = cubical(img_batch)
# print(len(per_cc))
# print(len(per_cc[0]))
# print(len(per_cc[0][0]))
# print(per_cc[0])
# print(per_cc[0][0])

# loss = wd_loss(per_cc, per_cc)



