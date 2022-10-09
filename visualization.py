import numpy as np
import torch
from torchvision.transforms import Resize
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from PIL import Image
import io
import os
from torch_tools.visualization import to_image

from utils import make_noise, one_hot


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)

@torch.no_grad()
def interpolate(G, z, shifts_r, shifts_count, dim, deformator=None, with_central_border=False):
    shifted_images = []
    for shift in np.arange(-shifts_r, shifts_r + 1e-9, shifts_r / shifts_count):
        #if deformator is not None:
        latent_shift = deformator(one_hot(deformator.input_dim, shift, dim).cuda())
        #else:
        #    latent_shift = one_hot(G.w_dim, shift, dim).cuda()
        
        w = G.mapping(z, None)
        #print(w.shape)
        #print(latent_shift.shape)
        #print(latent_shift.unsqueeze(1).shape)
        #print(w[:, 0, :] == w[:, 1, :])
        #print(w.shape, latent_shift.shape)
        if w.shape == latent_shift.shape:
            shifted_image = G.synthesis(w + latent_shift)[0]
        else:
            shifted_image = G.synthesis(w + latent_shift.unsqueeze(1))[0]
        if shift == 0.0 and with_central_border:
            shifted_image = add_border(shifted_image)

        shifted_images.append(shifted_image)
    return shifted_images

@torch.no_grad()
def interpolate2(G, z, shifts_r, shifts_count, dim, deformator=None, with_central_border=False, noise_val = 'const', first = None):
    shifted_images = []
    trunc_val = 0.7
    for shift in np.arange(-shifts_r, shifts_r + 1e-9, shifts_r / shifts_count):
        #if deformator is not None:

        
        latent_shift = deformator(one_hot(deformator.input_dim, shift, dim).cuda())
        
        
        # else:
        #     latent_shift = one_hot(G.w_dim, shift, dim).cuda()
        #print(dim)
        #print(latent_shift)s
        #print(type(z))
        w = G.mapping(z, None, truncation_psi = trunc_val)
        if w.shape == latent_shift.shape:
            shifted_image = G.synthesis(w + latent_shift, noise_mode = noise_val)[0]
        else:
            #print("no")
            # if first:
            #     #print(w.shape)
            #     latent_shift = latent_shift.unsqueeze(1).repeat(1, w.shape[1], 1)
                #latent_shift[:, first:, :] = 0
            #else:
            latent_shift = latent_shift.unsqueeze(1)
            shifted_image = G.synthesis(w + latent_shift, noise_mode = noise_val)[0]
        if shift == 0.0 and with_central_border:
            #print("no")
            shifted_image = add_border(shifted_image)

        shifted_images.append(shifted_image)
    return shifted_images

@torch.no_grad()
def interpolate3(G, z, shifts_r, shifts_count, dim, dim2, deformator=None, noise_val = 'const', first = None):
    print(first)
    trunc_val = 0.7
    shifted_images = []
    shifts1 = []
    shifts2 = []
    for shift in np.arange(-shifts_r, shifts_r + 1e-9, shifts_r / shifts_count):
        shifts1.append(deformator(one_hot(deformator.input_dim, shift, dim).cuda()))
        shifts2.append(deformator(one_hot(deformator.input_dim, shift, dim2).cuda()))
    #shifts2 = shifts2[::-1]
    #shifts1 = shifts1[::-1]

    for s1 in shifts1:
        # if first:
        #     s1 = s1.unsqueeze(1).repeat(1, 16, 1)
        #     s1[:, first:, :] = 0
        for s2 in shifts2:
            w = G.mapping(z, None, truncation_psi = trunc_val)
            #print(s1.shape, s2.shape, w.shape)
            if w.shape == s1.shape:
                shifted_image = G.synthesis(w + s1 + s2, noise_mode = noise_val)[0]
            else:
                if first:              
                    s2 = s2.unsqueeze(1).repeat(1, w.shape[1], 1)
                    s2[:, first:, :] = 0
                else:
                    s1 = s1.unsqueeze(1)
                    s2 = s2.unsqueeze(1)
                #print(w.shape, s1.shape, s2.shape)
                shifted_image = G.synthesis(w + s1 + s2, noise_mode = noise_val)[0]

            shifted_images.append(shifted_image)
            
    return shifted_images


def add_border(tensor):
    border = 3
    for ch in range(tensor.shape[0]):
        color = 1.0 if ch == 0 else -1
        tensor[ch, :border, :] = color
        tensor[ch, -border:,] = color
        tensor[ch, :, :border] = color
        tensor[ch, :, -border:] = color
    return tensor


@torch.no_grad()
def make_interpolation_chart(G, deformator=None, z=None,
                             shifts_r=10.0, shifts_count=5,
                             dims=None, dims_count=10, texts=None, **kwargs):
    with_deformation = deformator is not None
    if with_deformation:
        deformator_is_training = deformator.training
        deformator.eval()
    z = z if z is not None else make_noise(1, G.z_dim).cuda()

    if with_deformation:
        original_img = G(z, None).cpu()
    else:
        original_img = G(z, None).cpu()
    imgs = []
    if dims is None:
        dims = range(min(dims_count, deformator.input_dim))
    for i in dims:
        imgs.append(interpolate(G, z, shifts_r, shifts_count, i, deformator))

    rows_count = len(imgs) + 1
    fig, axs = plt.subplots(rows_count, **kwargs)

    axs[0].axis('off')
    axs[0].imshow(to_image(original_img, True))

    if texts is None:
        texts = dims
    for ax, shifts_imgs, text in zip(axs[1:], imgs, texts):
        ax.axis('off')
        plt.subplots_adjust(left=0.5)
        ax.imshow(to_image(make_grid(shifts_imgs, nrow=(2 * shifts_count + 1), padding=1), True))
        ax.text(-20, 21, str(text), fontsize=10)

    if deformator is not None and deformator_is_training:
        deformator.train()

    return fig


@torch.no_grad()
def inspect_all_directions(G, deformator, out_dir, max_dir, zs=None, num_z=3, shifts_r=8.0):
    os.makedirs(out_dir, exist_ok=True)

    step = 20
    max_dim = max_dir
    zs = zs if zs is not None else make_noise(num_z, G.z_dim).cuda()
    shifts_count = zs.shape[0]

    for start in range(0, max_dim - 1, step):
        imgs = []
        dims = range(start, min(start + step, max_dim))
        for z in zs:
            z = z.unsqueeze(0)
            fig = make_interpolation_chart(
                G, deformator=deformator, z=z,
                shifts_count=shifts_count, dims=dims, shifts_r=shifts_r,
                dpi=250, figsize=(int(shifts_count * 4.0), int(0.5 * step) + 2))
            fig.canvas.draw()
            plt.close(fig)
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # crop borders
            nonzero_columns = np.count_nonzero(img != 255, axis=0)[:, 0] > 0
            img = img.transpose(1, 0, 2)[nonzero_columns].transpose(1, 0, 2)
            imgs.append(img)

        out_file = os.path.join(out_dir, '{}_{}.jpg'.format(dims[0], dims[-1]))
        print('saving chart to {}'.format(out_file))
        Image.fromarray(np.hstack(imgs)).save(out_file)


def gen_animation(G, deformator, direction_index, out_file, z=None, size=None, r=8):
    import imageio

    if z is None:
        z = torch.randn([1, G.z_dim], device='cuda')
    interpolation_deformed = interpolate(
        G, z, shifts_r=r, shifts_count=5,
        dim=direction_index, deformator=deformator, with_central_border=False)

    resize = Resize(size) if size is not None else lambda x: x
    img = [resize(to_image(torch.clamp(im, -1, 1))) for im in interpolation_deformed]
    imageio.mimsave(out_file, img + img[::-1])