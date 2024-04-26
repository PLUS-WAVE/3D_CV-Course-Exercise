from pathlib import Path
import inspect

from PIL import Image
import numpy as np

from matplotlib import pyplot as plt


def load_imgs_masks_light_dirs(object_name="cat"):
    """
    returns:
    imgs np.array [k,h,w] np.float32 [0.0, 1.0]
    mask np.array [h,w] np.bool
    light_positions np.array [k,3] np.float32
    k: number of images
    h: image height (num rows)
    w: image width (num cols)
    """
    available_objs = [
        x.stem for x in DATA_DIR.iterdir() if x.is_dir() and "chrome" not in str(x)
    ]

    assert (
            object_name in available_objs
    ), "unknown obj {0} - please select one of {1}".format(object_name, available_objs)

    obj_dir = DATA_DIR.joinpath(object_name)

    mask = (
            np.array(
                Image.open(
                    obj_dir.joinpath("{}.{}.png".format(object_name, "mask"))
                ).convert("L")
            )
            > 0
    )

    imgs = []
    for im_path in sorted(list(obj_dir.glob("*.png"))):
        if "mask" in str(im_path):
            # we already got that one
            continue
        else:
            img = Image.open(im_path).convert("L")
            imgs.append(np.array(img))

    imgs = np.stack(imgs, axis=0).astype(np.float64) / 256.0

    # normally these would have to be recovered from the chrome ball
    # we hard-code them here to save time
    light_dirs = np.array(
        [
            [0.49816584, 0.46601385, 0.73120577],
            [0.24236702, 0.13237001, 0.96111207],
            [-0.03814999, 0.17201198, 0.98435586],
            [-0.09196399, 0.44121093, 0.89267886],
            [-0.31899811, 0.50078717, 0.80464428],
            [-0.10791803, 0.55920516, 0.82197524],
            [0.27970709, 0.42031713, 0.86319028],
            [0.09845196, 0.42847982, 0.89817162],
            [0.20550002, 0.33250804, 0.9204391],
            [0.08520805, 0.33078218, 0.93985251],
            [0.12815201, 0.043478, 0.99080105],
            [-0.13871804, 0.35998611, 0.92258729],
        ]
    )

    return imgs, mask, light_dirs


def compute_normals_albedo_map(imgs, mask, light_positions):
    """
    imgs np.array [k,h,w] np.float32 [0.0, 1.0]
    mask np.array [h,w] np.bool
    light_positions np.array [k,3] np.float32
    ---
    dims:
    k: number of images
    h: image height (num rows)
    w: image width (num cols)
    """

    S = light_positions
    I = imgs.reshape(imgs.shape[0], -1)

    # rho n = (S^T S)^-1 S^T I
    rho_n = np.linalg.inv(S.T @ S) @ S.T @ I
    # rho = ||rho_n||
    rho = np.linalg.norm(rho_n, axis=0)
    # n = rho_n / rho
    n = np.divide(rho_n, rho, out=np.zeros_like(rho_n), where=rho != 0)

    # mask out
    mask_flat = mask.flatten()
    n[:, ~mask_flat] = 0

    normals_unit = n.T.reshape(imgs.shape[1], imgs.shape[2], 3)
    rho = rho.reshape(imgs.shape[1], imgs.shape[2])

    assert normals_unit.shape == (imgs.shape[1], imgs.shape[2], 3)
    assert rho.shape == (imgs.shape[1], imgs.shape[2])

    rho = np.clip(rho, 0.0, 1.0)
    normals_unit = np.clip(normals_unit, 0.0, 1.0)

    return normals_unit, rho, mask


def relight_scene(light_pos, normals_unit, albedo, mask):
    """
    light_pos np.array [k,3] np.float32
    mask np.array [h,w] np.bool
    ----
    dims:
    h: image height (num rows)
    w: image width (num cols)
    ----
    returns:
        imgs np.array [h,w] np.float32 [0.0, 1.0]
    """
    assert light_pos.shape == (3,)
    assert np.allclose(1.0, np.linalg.norm(light_pos))
    assert normals_unit.shape[-1] == 3
    assert len(normals_unit.shape) == 3

    img = albedo * (normals_unit @ light_pos)
    # mask out
    img[~mask] = 0
    img_norm = np.clip(img, 0.0, 1.0)

    assert np.all(
        np.logical_and(0.0 <= img_norm, img_norm <= 1.0)
    ), "please normalize your image to interval [0.0,1.0]"
    return img_norm


if __name__ == '__main__':
    # find the data directory by looking at this files position on your system
    DATA_DIR = Path(inspect.getfile(lambda: None)).parent.joinpath("assets/data", "psmImages")

    assert (
        DATA_DIR.exists()
    ), "input data does not exist - please mak e sure to run ./get_data.sh in data folder"
    imgs, mask, light_positions = load_imgs_masks_light_dirs("buddha")

    normals_unit, rho, mask = compute_normals_albedo_map(imgs, mask, light_positions)

    plt.figure()
    plt.imshow(rho, cmap="gray")
    plt.title("Albedo")
    plt.show()

    plt.figure()
    plt.imshow(normals_unit)
    plt.title("Normals")
    plt.show()

    light_pos = np.array([0.5, 0.5, 0.7])
    new_albedo = 0.5
    for x in np.linspace(-2, 2, 5):
        light_pos[0] = x
        light_pos = np.array(light_pos / np.linalg.norm(light_pos))

        new_img = relight_scene(light_pos, normals_unit, new_albedo, mask)

        plt.figure()
        plt.imshow(new_img, cmap=plt.cm.gray)
        plt.title(
            "Relit image \nNew light position @ {0}\nAlbedo is now {1}".format(light_pos, new_albedo)
        )
        plt.show()
