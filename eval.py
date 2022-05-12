import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from torch_tools.visualization import to_image
from visualization import interpolate
from loading import load_from_dir
from PIL import Image

deformator, G, shift_predictor = load_from_dir(
    root_dir = '/dataB1/tommie/recon_out/new_w_t0',
    plk = '/dataB1/tommie/training-runs/00027-stylegan2-onlycrack_crop_256_img-gpus2-batch16-gamma32/network-snapshot-002862.pkl')
#print(G.w_dim)
#print(G.z_dim)
print('deformator.annotation', deformator.annotation)
discovered_annotation = ''
for d in deformator.annotation.items():
    discovered_annotation += '{}: {}\n'.format(d[0], d[1])
rows = 10
annotated = list(deformator.annotation.values())
for inspection_dim in annotated:
    plt.figure(figsize=(5, rows), dpi=256)
    plt.title("Interpolation of direction nr {}".format(inspection_dim))
    zs = torch.randn([rows, G.z_dim] if type(G.z_dim) == int else [rows] + G.z_dim, device='cuda')
    for z, i in zip(zs, range(rows)):
        interpolation_deformed = interpolate(
            G, z.unsqueeze(0),
            shifts_r=16,
            shifts_count=3,
            dim=inspection_dim,
            deformator=deformator,
            with_central_border=True)

        plt.subplot(rows, 1, i + 1)
        plt.axis('off')
        grid = make_grid(interpolation_deformed, nrow=11, padding=0, pad_value=0.0)
        grid = torch.clamp(grid, -1, 1)
        plt.imshow(to_image(grid))
    out_file = '/dataB1/tommie/recon_out/new_w_t0/evals/chart_{}.jpg'.format(inspection_dim)
    plt.savefig(out_file)