import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
# import SimpleITK as sitk
import torch
import cv2
from torch import nn
import SimpleITK as sitk
import time
from torch_topological.nn import CubicalComplex, WassersteinDistance

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            pers = (1.00-image_data.ravel()[birth], 1.0)
            result.append((dim, pers, ((birth_y, birth_x), None)))

    return result

def plot_persistence_diagram_with_coordinates(ax, cofaces_output):
    # 点をプロットし、座標情報を追加
    for dim, (birth, death), coordinates in cofaces_output:
        if dim == 0:
            color = 'red'
        else:
            color = 'blue'

        # deathがinfの場合は1に設定
        death = 1 if np.isinf(death) else death

        ax.scatter(birth, death, c=color, s=10)

        # 座標情報のテキストを作成
        if coordinates[1] is None:
            coord_text = f"({coordinates[0][0]}, {coordinates[0][1]})"
        else:
            coord_text = f"({coordinates[0][0]}, {coordinates[0][1]}) -> ({coordinates[1][0]}, {coordinates[1][1]})"

        # 点の横にテキストを追加
        ax.annotate(coord_text, (birth, death), xytext=(5, 5), textcoords='offset points', fontsize=8)

    # 対角線を描画
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.3, zorder=0)

    ax.set_title('Persistence Diagram with Coordinates')
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')

image_list = []


np.random.seed(2)
weight_thre = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

# for i in range(3):
#     # ランダムな場所から64*64に切り取る
#     x = np.random.randint(0, image.shape[0] - 64)
#     y = np.random.randint(0, image.shape[1] - 64)

#     while True:
#         x = np.random.randint(0, image.shape[0] - 64)
#         y = np.random.randint(0, image.shape[1] - 64)
#         org_image = image[x:x+64, y:y+64]
#         if (np.count_nonzero(org_image==0)/np.count_nonzero(org_image>=0) <= 0.6):#黒の割合
#             break

#     org_image = (org_image - org_image.min()) / (org_image.max() - org_image.min())
#     image_list.append(org_image)

image = sitk.GetArrayFromImage(sitk.ReadImage("img_for_slide/1.mhd"))
image = (image - image.min()) / (image.max() - image.min())
image_list.append(image)
image = sitk.GetArrayFromImage(sitk.ReadImage("img_for_slide/40.mhd"))
image = (image - image.min()) / (image.max() - image.min())
image_list.append(image)
image = sitk.GetArrayFromImage(sitk.ReadImage("img_for_slide/100.mhd"))
image = (image - image.min()) / (image.max() - image.min())
image_list.append(image)

ori_image_list = []
ori1_image = sitk.GetArrayFromImage(sitk.ReadImage("img_for_slide/1_ori.mhd"))
ori40_image = sitk.GetArrayFromImage(sitk.ReadImage("img_for_slide/40_ori.mhd"))
ori100_image = sitk.GetArrayFromImage(sitk.ReadImage("img_for_slide/100_ori.mhd"))
ori_image_list.append(ori1_image)
ori_image_list.append(ori40_image)
ori_image_list.append(ori100_image)

# from soft_frangi.soft_frangi_filter2d import SoftFrangiFilter2D
# soft_frangi_filter = SoftFrangiFilter2D(channels=1, kernel_size=7, sigmas=range(1, 10, 2), beta=0.5, c=0.5, device='cpu')
from skimage.filters import frangi, sato


plt.clf()
fig, axs = plt.subplots(3, len(image_list), figsize=(7*len(image_list), 10))

for k, weight_thre in enumerate(weight_thre):
    fig, axs = plt.subplots(3, len(image_list), figsize=(7*len(image_list), 10))

    for i, (img, ori_img) in enumerate(zip(image_list, ori_image_list)):
        # 画像の表示
        # axs[0, i].imshow(img, cmap='gray')
        # axs[0, i].axis('off')
        # axs[0, i].set_title("Image " + str(i))

        # axs[2, i].imshow(ori_img, cmap='gray')
        # axs[2, i].axis('off')
        # axs[2, i].set_title("Ori Image " + str(i))

        # ori_imgの192の画素の数をカウント
        # print(f"ori_imgの192の画素の数: {np.count_nonzero(ori_img==192)}")

        # Persistent Homologyの計算
        cc = gd.CubicalComplex(
            dimensions=img.shape, top_dimensional_cells=1.0-img.flatten()
        )
        persistence = cc.persistence()
        cofaces = cc.cofaces_of_persistence_pairs()

        result = match_cofaces_with_gudhi(img, cofaces)

        # print(f"persistence for Image {i}:", persistence)
        # print(f"cofaces for Image {i}:", cofaces)
        # print(f"result for Image {i}:", result)

        # Persistence Diagramの表示
        ax = axs[2, i]
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

        bx = axs[1, i]
        bx.set_title('frangi vs Distance')
        bx.set_xlabel('Distance')
        bx.set_ylabel('frangi')

        # cx = axs[3, i]

        filtered_diagram = [(dim, (birth, 1 if np.isinf(death) else death)) for dim, (birth, death), _ in result]

        frangi_img = frangi(1-img)
        print("frangi max min",frangi_img.max(), frangi_img.min())
        # cx.imshow(frangi_img, cmap='gray')

        img_scaled = (img * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2RGB)
        point_num = 0
        # 点をプロットし、座標情報を追加
        for (dim, (birth, death)), (_, _, coordinates) in zip(filtered_diagram, result):
            if dim == 0:
                color = 'red'
            else:
                color = 'blue'
                continue

            # birth, deathの点から対角線までの垂直に下した線のL1距離を計算
            distance = np.abs(birth - death) / np.sqrt(2)
            # print(f"distance: {distance}")
            weight = distance * frangi_img[coordinates[0][0], coordinates[0][1]]
            # print(f"weight: {weight}")

            # x軸distance、y軸frangiのグラフを描画
            if weight > weight_thre:
                ax.scatter(birth, death, c=color, s=10)
                bx.scatter(distance, frangi_img[coordinates[0][0], coordinates[0][1]], c=color, s=10)
                bx.annotate(f"({coordinates[0][0]}, {coordinates[0][1]})", (distance, frangi_img[coordinates[0][0], coordinates[0][1]]), xytext=(3, 3), textcoords='offset points', fontsize=8)
                img_rgb[coordinates[0][0], coordinates[0][1]] = [255, 0, 0]
                point_num += 1
            else:
                ax.scatter(birth, death, c='green', s=10)
                bx.scatter(distance, frangi_img[coordinates[0][0], coordinates[0][1]], c='green', s=10)
                continue
            # ax.scatter(birth, death, c=color, s=10)

            # 座標情報のテキストを作成
            if coordinates[1] is None:
                coord_text = f"({coordinates[0][0]}, {coordinates[0][1]})"
            else:
                coord_text = f"({coordinates[0][0]}, {coordinates[0][1]}) -> ({coordinates[1][0]}, {coordinates[1][1]})"

            # 点の横にテキストを追加
            ax.annotate(coord_text, (birth, death), xytext=(3, 3), textcoords='offset points', fontsize=8)

        # 対角線を描画
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k-', alpha=0.3, zorder=0)

        print(f"点の数: {point_num}")
        ax.annotate(f"weight threshold: {weight_thre}", (0.7, 0.6), xytext=(3, 3), textcoords='offset points', fontsize=12)
        ax.annotate(f"select point num: {point_num}", (0.7, 0.5), xytext=(3, 3), textcoords='offset points', fontsize=12)
        ax.annotate(f"all point num: {len(filtered_diagram)}", (0.7, 0.4), xytext=(3, 3), textcoords='offset points', fontsize=12)

        ax.set_title(f'Persistence Diagram {i}')
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')

        frangi_img = (frangi_img * 255).astype(np.uint8)
        frangi_img = cv2.cvtColor(frangi_img, cv2.COLOR_GRAY2RGB)
        concat_img = np.concatenate([img_rgb, frangi_img], axis=1)
        axs[0, i].imshow(concat_img)
        axs[0, i].axis('off')
        axs[0, i].set_title("Image " + str(i)+ "  left:original right:frangi")

    plt.tight_layout()
    plt.savefig(f"img_for_slide/image_persistence_diagram_{weight_thre}.png")

for i, img in enumerate(image_list):
    # gudhiでの表示を行う
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    cc = gd.CubicalComplex(
        dimensions=img.shape, top_dimensional_cells=1.0-img.flatten()
    )
    persistence = cc.persistence()

    gd.plot_persistence_diagram(persistence)
    plt.title(f"Image {i} Persistence Diagram")
    plt.savefig(f"image_{i}_persistence_diagram.png")

print("gudhi:", persistences)

cofaces = cc.cofaces_of_persistence_pairs()

for idx, (birth, death) in enumerate(persistence):
    if death[1] == float("inf"):
        persistence[idx] = (birth, (death[0], image_data.max()))

print("gudhi:", persistence)
print("cofaces:", cofaces)

combined_images = np.array(image_list)
img_batch = torch.tensor(combined_images, device=device, dtype=torch.float32).unsqueeze(1)
cubical = CubicalComplex()
wd_loss = WassersteinDistance(q=2)
per_cc = cubical(img_batch)
print(per_cc)

loss = wd_loss(per_cc, per_cc)