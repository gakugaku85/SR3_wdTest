import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
# import SimpleITK as sitk
import torch
import cv2
from torch import nn
import SimpleITK as sitk
import time
from model.torch_topological.nn import CubicalComplex, WassersteinDistance
from icecream import ic
from skimage.filters import frangi

def match_cofaces_with_gudhi(image_data, cofaces):
    height, width = image_data.shape
    result = []

    for dim, pairs in enumerate(cofaces[0]):
        for birth, death in pairs:
            birth_y, birth_x = np.unravel_index(birth, (height, width))
            death_y, death_x = np.unravel_index(death, (height, width))
            pers = (1.00-image_data.ravel()[birth], 1.00-image_data.ravel()[death])
            result.append((dim, pers,((birth_y, birth_x), (death_y, death_x))))

    for dim, births in enumerate(cofaces[1]):
        for birth in births:
            birth_y, birth_x = np.unravel_index(birth, (height, width))
            pers = (1.0-image_data.ravel()[birth], 1.0)
            result.append((dim, pers, ((birth_y, birth_x), None)))

    return result

def persistent_homology(image_data, image_name="SR"):
    """Computes and visualizes the persistent homology for the given image data."""
    cc = gd.CubicalComplex(
        dimensions=image_data.shape, top_dimensional_cells=1 - image_data.flatten()
    )
    cc.persistence()
    cofaces = cc.cofaces_of_persistence_pairs()
    result = match_cofaces_with_gudhi(image_data=image_data, cofaces=cofaces)

    frangi_img = frangi(1-image_data)
    new_result = []

    for dim, (birth, death) , coordinates in result:
        if dim == 1:
            continue
        if image_name == "SR":
            new_result.append([birth, death])
            continue
        distance = np.abs(birth - death) / np.sqrt(2)
        weight = distance * frangi_img[coordinates[0][0], coordinates[0][1]]

        weight_threshold = 0.01
        if weight > weight_threshold:
            new_result.append([birth, death])

    return torch.tensor(new_result, dtype=torch.float32, device=device).unsqueeze(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_list = []

image = sitk.GetArrayFromImage(sitk.ReadImage("../../WDtest/img_for_slide/1.mhd"))
image = (image - image.min()) / (image.max() - image.min())


    # gudhiでの表示を行う
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
cc = gd.CubicalComplex(
    dimensions=image.shape, top_dimensional_cells=1.0-image.flatten()
)
persistence = cc.persistence()

gd.plot_persistence_diagram(persistence)
plt.title(f"Image Persistence Diagram")
plt.savefig(f"image_persistence_diagram.png")

cofaces = cc.cofaces_of_persistence_pairs()

conect = persistent_homology(image, "HR")

# combined_images = np.array(image_list)
# img_batch = torch.tensor(combined_images, device=device, dtype=torch.float32).unsqueeze(1)
cubical = CubicalComplex()
wd_loss = WassersteinDistance(q=2)
image = torch.tensor(image, device=device, dtype=torch.float32)
ic(image.shape)
per_cc = cubical(image)[0]
# print(per_cc)

loss = wd_loss(per_cc, conect)

ic(loss)