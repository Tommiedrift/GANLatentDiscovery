import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from torch_tools.visualization import to_image
from visualization import interpolate, interpolate2, interpolate3
from loading import load_from_dir
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.transforms import Resize
import argparse
import numpy as np
import imageio
from utils import make_noise, one_hot
import os
import cv2
from torch_tools.modules import DataParallelPassthrough
import itertools
from trainer import Trainer, Params, ShiftDistribution
import math

parser = argparse.ArgumentParser(
        description='')
parser.add_argument('--root', type=str)
parser.add_argument('--pkl', type=str)
parser.add_argument('--anno', type=bool, default=False)
parser.add_argument('--dims', type=int)
parser.add_argument('--folder', type=str, default = 'eval')
parser.add_argument('--noise', type=str, default='random')
parser.add_argument('--vid', type=bool, default=False)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--multi_layer', type=bool, default=False)
parser.add_argument('--all_layer', type=bool, default=False)
parser.add_argument('--dim1', type=int, default=None)
parser.add_argument('--dim2', type=int, default=None)
parser.add_argument('--grid_len', type=int, default=11)
parser.add_argument('--magnitude', type=int, default=18)
parser.add_argument('--grid_num', type=int, default=1)
parser.add_argument('--switch', type=bool, default=False)
parser.add_argument('--version', type=str, default=True)
parser.add_argument('--res', type=int, default=256)
parser.add_argument('--rows', type=int, default=10)
parser.add_argument('--frames', type=int, default=20)
parser.add_argument('--pathfig', type=str)
parser.add_argument('--first', type=int, default=None)
parser.add_argument('--amount', type=int, default=100)
parser.add_argument('--w_var', type=str, default=None)
args = parser.parse_args()

def init():
    deformator, G, shift_predictor = load_from_dir(
        root_dir = '/dataB1/tommie/recon-runs/{}'.format(args.root),
        pkl = args.pkl,
        multi_layer = args.multi_layer,
        all_layer = args.all_layer,
        dims = args.dims,
        w_var = args.w_var)
    if args.multi_gpu:
        G = DataParallelPassthrough(G)
    if args.anno:
        discovered_annotation = ''
        for d in deformator.annotation.items():
            discovered_annotation += '{}: {}\n'.format(d[0], d[1])
        annotated = list(deformator.annotation.values())
    else:
        annotated = range(deformator.input_dim)
    if not os.path.exists('/dataB1/tommie/recon-runs/{}/{}'.format(args.root, args.folder)):
        os.mkdir('/dataB1/tommie/recon-runs/{}/{}'.format(args.root, args.folder))
    return deformator, G, shift_predictor, annotated


def make_plots(deformator, args, hor = True, center = False, pathfig = 'whatever', bound = None):
    d = 0
    for i in range(7):
        if not os.path.exists('/dataB1/tommie/recon-runs/latent_shifts/{}'.format(pathfig)):
            os.mkdir('/dataB1/tommie/recon-runs/latent_shifts/{}'.format(pathfig))
        shift = deformator(one_hot(deformator.input_dim, i, 0).cuda())
        #np.savetxt('/dataB1/tommie/recon-runs/latent_shifts/projective/shift{}.txt'.format(d), shift.cpu().detach().numpy(),  delimiter=',')
        if args.multi_layer:
            if d < deformator.input_dim / 3:
                shift = shift[:, 0, :]
            elif d < (deformator.input_dim / 3) * 2:
                shift = shift[:, 4, :]
            else:
                shift = shift[:, 8, :]
        save = '/dataB1/tommie/recon-runs/latent_shifts/{}/{}'.format(pathfig, 'shift{}.png').format(i)
        values = np.array(shift.cpu().detach().numpy())[0]
        fig = plt.figure(i, figsize=(12, 4), facecolor='white')
        plt.bar(range(deformator.shift_dim), values, 1, alpha=1)
        #plt.yticks(rotation=0, fontsize = 10)
        # if hor:
        #    
        # else:
        #     plt.bar(range(deformator.shift_dim), values)
        # if center:
        #     yabs_max = abs(max(values, key=abs)) + 0.01
        #     plt.ylim(ymin=-yabs_max, ymax=yabs_max)
        # if bound:
        #     plt.ylim(-bound, bound, emit = False)
        plt.title('shift direction {}'.format(i), fontsize=10)
        plt.xlabel('dimension', fontsize=10)
        plt.ylabel('shift amount')
        #fig.tight_layout()
        plt.savefig(save)

def create_grids(zs, args, rcas, dim1, dim2, G, deformator):
    for index, z in enumerate(zs):
        fig = plt.figure(figsize=(args.grid_len, args.grid_len), dpi=args.res)
        if rcas:
            plt.title("Interpolation Grid of Directions {} and {} \n [{} RCA = {:.2f}, {} RCA = {:.2f}]".format(dim1, dim2, dim1, rca[dim1], dim2, rca[dim2]), size = 25 / (11/args.grid_len))
        else:
            plt.title("Interpolation Grid of Directions {} and {}".format(dim1, dim2), size = 25 / (11/args.grid_len))
        plt.xlabel("dir {}".format(dim1), size = 25 / (11/args.grid_len))
        plt.ylabel("dir {}".format(dim2), size = 25 / (11/args.grid_len))
        plt.xticks([])
        plt.yticks([])
        #                            G, z,               shifts_r,       shifts_count,        dim,  dim2, deformator=None, with_central_border=False, noise_val = 'const', first = None
        interpol_grid = interpolate3(G, z.unsqueeze(0), args.magnitude, (args.grid_len-1)/2, dim2, dim1, deformator, args.noise, args.first)
        if args.switch:
            interpol_grid = interpol_grid[::-1]
        grid = ImageGrid(fig, 111,
                    nrows_ncols=(args.grid_len, args.grid_len),
                    axes_pad=0.05, label_mode = "1")
        for ax, im in zip(grid, interpol_grid):
            ax.imshow(im.permute(1, 2, 0).cpu().numpy())
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        out_file = '/dataB1/tommie/recon-runs/{}/{}/grid_{}_{}_{}_{}_{}.png'.format(args.root, args.folder, dim1, dim2, args.grid_len, args.magnitude, index)
        plt.savefig(out_file)

def create_vids(args, rcas, annotated, G, deformator):
    for inspection_dim in annotated:
        plt.figure(figsize=(5, args.rows), dpi=256)
        plt.title("Interpolation of direction nr {}".format(inspection_dim))
        #print(type(G.z_dim) == int)
        zs = torch.randn([args.rows, G.z_dim] if type(G.z_dim) == int else [args.rows] + G.z_dim, device='cuda')
        #img = []
        imgs = []
        for z, i in zip(zs, range(args.rows)):
            #print(z.shape)
            #print(z.unsqueeze(0).shape)
            interpolation_deformed2 = interpolate2(G, z.unsqueeze(0), args.magnitude, 3, inspection_dim, deformator, False, args.noise, args.first)
            plt.subplot(args.rows, 1, i + 1)
            plt.axis('off')
            grid = make_grid(interpolation_deformed2, nrow=args.rows, padding=0, pad_value=0.0)
            grid = torch.clamp(grid, -1, 1)
            plt.imshow(to_image(grid))
            if args.vid:
                interpolation_deformed2 = interpolate2(G, z.unsqueeze(0), args.magnitude, 20, inspection_dim, deformator, False, args.noise, args.first)
                resize = Resize(args.res)
                im = [resize(to_image(torch.clamp(im, -1, 1))) for im in interpolation_deformed2]
                for i in im + im[::-1]:
                    imgs.append(i)
        out_file = '/dataB1/tommie/recon-runs/{}/{}/chart_{}.png'.format(args.root, args.folder, inspection_dim)
        plt.savefig(out_file)
        print(inspection_dim)
        if args.vid:    
            s_out_file = '/dataB1/tommie/recon-runs/{}/{}/video_{}.mp4'.format(args.root, args.folder, inspection_dim)
            out = cv2.VideoWriter(s_out_file, cv2.VideoWriter_fourcc(*'mp4v'), 40, (args.res, args.res))
            for frame in imgs:
                out.write(np.array(frame)[:, :, ::-1].copy())
            out.release()

def create_dim_grid(args, rcas, dim, G, deformator):
    for j in range(args.grid_num):
        plt.figure(figsize=(5, args.rows), dpi=args.res)
        plt.title("Interpolation of direction nr {}".format(dim))
        #print(type(G.z_dim) == int)
        zs = torch.randn([args.rows, G.z_dim] if type(G.z_dim) == int else [args.rows] + G.z_dim, device='cuda')
        #img = []
        imgs = []
        for z, i in zip(zs, range(args.rows)):
            #print(z.shape)
            #print(z.unsqueeze(0).shape)
            if i == 0:
                plt.title("Interpolation of direction {}".format(str(dim+1)))
            interpolation_deformed2 = interpolate2(G, z.unsqueeze(0), args.magnitude, 3, dim, deformator, False, args.noise, args.first)
            plt.subplot(args.rows, 1, i + 1)
            plt.axis('off')
            grid = make_grid(interpolation_deformed2, nrow=args.rows, padding=0, pad_value=0.0)
            grid = torch.clamp(grid, -1, 1)
            plt.imshow(to_image(grid))
            if args.vid:
                interpolation_deformed2 = interpolate2(G, z.unsqueeze(0), args.magnitude, 20, dim, deformator, False, args.noise, args.first)
                resize = Resize(args.res)
                im = [resize(to_image(torch.clamp(im, -1, 1))) for im in interpolation_deformed2]
                for i in im + im[::-1]:
                    imgs.append(i)
        out_file = '/dataB1/tommie/recon-runs/{}/{}/chart_{}_{}.png'.format(args.root, args.folder, dim, j)
        while os.path.exists(out_file):
            out_file = out_file[:-4] + '_1.png'
        plt.tight_layout()
        plt.savefig(out_file)
        print(dim)
        if args.vid:    
            s_out_file = '/dataB1/tommie/recon-runs/{}/{}/video_{}.mp4'.format(args.root, args.folder, dim)
            out = cv2.VideoWriter(s_out_file, cv2.VideoWriter_fourcc(*'mp4v'), 40, (args.res, args.res))
            for frame in imgs:
                out.write(np.array(frame)[:, :, ::-1].copy())
            out.release()


def test_shift(latent_dim, target, b_size, trainer):
    target_indices = torch.ones(b_size, device='cuda') * float(target)
    target_indices = target_indices.long()
    if trainer.p.shift_distribution == ShiftDistribution.NORMAL:
        shifts = torch.randn(target_indices.shape, device='cuda')
    elif trainer.p.shift_distribution == ShiftDistribution.UNIFORM:
        shifts = 2.0 * torch.rand(target_indices.shape, device='cuda') - 1.0

    shifts = trainer.p.shift_scale * shifts
    shifts[(shifts < trainer.p.min_shift) & (shifts > 0)] = trainer.p.min_shift
    shifts[(shifts > -trainer.p.min_shift) & (shifts < 0)] = -trainer.p.min_shift

    try:
        latent_dim[0]
        latent_dim = list(latent_dim)
    except Exception:
        latent_dim = [latent_dim]

    z_shift = torch.zeros([b_size] + latent_dim, device='cuda')
    for i, (index, val) in enumerate(zip(target_indices, shifts)):
        z_shift[i][index] += val

    return target_indices, shifts, z_shift

@torch.no_grad()
def calc_RCA(G, deformator, shift_predictor, args):
    n_steps = 100
    all_per = []
    params = Params(**args.__dict__)
    trainer = Trainer(params, out_dir=args.root + '/trainer')
    for i in range(deformator.input_dim):
        percents = torch.empty([n_steps])
        for step in range(n_steps):
            z = make_noise(1, G.z_dim, None).cuda()
            target_indices, shifts, basis_shift = test_shift(latent_dim = deformator.input_dim, target = i, b_size = 1, trainer = trainer)
            latent_shift = deformator(basis_shift)
            imgs = G(z, None)
            w = G.mapping(z, None)
            if deformator.layer:
                shifted_image = G.synthesis(w + latent_shift)[0]
            else:
                shifted_image = G.synthesis(w + latent_shift.unsqueeze(0))[0]
            logits, _ = shift_predictor(imgs, shifted_image.unsqueeze(0))
            # print(logits)
            # print((torch.argmax(logits, dim=1) == target_indices).to(torch.float32))
            # print((torch.argmax(logits, dim=1) == target_indices).to(torch.float32).mean())
            percents[step] = (torch.argmax(logits, dim=1) == target_indices).to(torch.float32).mean()

        print(i, percents.mean())
        all_per.append(percents.mean())
    print("all")
    print(all_per)

def check_w_stats(args, G, deformator):
    for d in range(args.dims):
        latent_shift = deformator(one_hot(deformator.input_dim, 1, d).cuda())
        print(d)
        print(torch.norm(latent_shift, dim = 2, keepdim=True))
    return True

def all_dims_chart(zs, args, rcas, annotated, G, deformator):
    for index, z in enumerate(zs):
        plt.figure(figsize=(9, deformator.input_dim), dpi= args.res)#args.res)
        imgs = []
        dims = []
        res = [4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]
        for index, inspection_dim in enumerate(annotated):
            interpolation_deformed2 = interpolate2(G, z.unsqueeze(0), args.magnitude, 4, inspection_dim, deformator, False, args.noise, args.first)
            plt.subplot(deformator.input_dim, 1, inspection_dim + 1)
            #if inspection_dim == 0:
            #    plt.title("Interpolation of all directions from -{} to {}".format(args.magnitude, args.magnitude), size = 20)
            grid = make_grid(interpolation_deformed2, nrow=deformator.input_dim, padding=0, pad_value=0.0)
            grid = torch.clamp(grid, -1, 1)
            plt.ylabel(res[index], rotation = 0, size = 20, labelpad = 15)
            plt.imshow(to_image(grid))
            plt.xticks([])
            plt.yticks([])
            if args.vid:
                interpolation_deformed2 = interpolate2(G, z.unsqueeze(0), args.magnitude, 20, inspection_dim, deformator, False, args.noise, args.first)
                resize = Resize(args.res)
                im = [resize(to_image(torch.clamp(im, -1, 1))) for im in interpolation_deformed2]
                for i in im + im[::-1]:
                    imgs.append(i)
                    dims.append(inspection_dim)
        out_file = '/dataB1/tommie/recon-runs/{}/{}/chart_all_{}.png'.format(args.root, args.folder, index)
        while os.path.exists(out_file):
            print(out_file)
            out_file = out_file[:-4] + '_1.png'
        plt.tight_layout()
        plt.savefig(out_file)
        print(index)
        if args.vid:    
            s_out_file = '/dataB1/tommie/recon-runs/{}/{}/video_{}.mp4'.format(args.root, args.folder, index)
            while os.path.exists(s_out_file):
                print('test')
                s_out_file = s_out_file[:-4] + '_1.mp4'
            out = cv2.VideoWriter(s_out_file, cv2.VideoWriter_fourcc(*'mp4v'), 40, (args.res, args.res))
            res = [4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]
            for index, frame in enumerate(imgs):
                cur_dim = str(dims[index])
                cur_row1 = int(np.floor(dims[index] / (deformator.input_dim / (deformator.num_l / 2))) * 2)
                cur_row2 = cur_row1 + 1
                if args.all_layer:
                    text = '{} - {}+{}/{} - {}x{}'.format(cur_dim, str(cur_row1), str(cur_row2), str(deformator.num_l), str(res[cur_row1]), str(res[cur_row1]))
                else:
                    text = cur_dim
                img = np.array(frame)[:, :, ::-1].copy()
                img_black_text = cv2.putText(img, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
                img_final = cv2.putText(img_black_text, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                out.write(img_final)
            out.release()

def calc_map_var(args, G, deformator):
    #z = torch.randn([1, G.z_dim], device='cuda')
    #w = G.mapping(z, None)
    #covars = torch.zeros([deformator.num_l, G.z_dim, G.z_dim])

    covars_one = torch.zeros([G.z_dim, G.z_dim])
    #all_w = torch.zeros([deformator.num_l, G.z_dim, args.amount])
    all_wone = torch.zeros([G.z_dim, args.amount])
    zs = torch.randn([args.amount, G.z_dim], device='cuda')
    print('mapping')
    for index, z in enumerate(zs):
        w = G.mapping(z.unsqueeze(0), None)
        #all_w[:, :, index] = w[0]
        all_wone[:, index] = w[0, 0, :]
    print('calc cov')
    #for i in range(deformator.num_l):
    #    covars[i, :, :] = torch.cov(all_w[i, :, :])
    covars_one = torch.cov(all_wone)
    print('saving')
    fol = '/dataB1/tommie/w_stats/{}'.format(args.folder)
    if not os.path.exists(fol):
        os.mkdir(fol)
    # for index, c in enumerate(covars):
    #     torch.save(c, '{}/{}_{}.pt'.format(fol, 'w_covar_tensor', index))
    #     np.savetxt('{}/{}_{}.txt'.format(fol, 'w_covar_matrix', index), c.cpu().detach().numpy(),  delimiter=',')
    torch.save(covars_one, '{}/{}.pt'.format(fol, 'w_covar'))
    #hm = torch.mean(torch.diagonal(covars_one))
    #tests.append(hm)
    # print('------')
    # print(tests)
    # print(np.mean(tests))

def other(args, G, deformator):
    sds = []
    covar = torch.load('/dataB1/tommie/w_stats/100k/w_covar_tensor_0.pt', map_location=torch.device('cuda'))
    for i in range(30):
        shift = deformator(one_hot(deformator.input_dim, 1, i).cuda())
        #print(shift.shape, covar.shape)
        #print(var.shape, covar.shape, shift.T.shape)
        var = shift @ covar @ shift.T
        sd = torch.sqrt(var)    
        sds.append(sd)
        print(i, sd[0])

deformator, G, shift_predictor, annotated = init()

if args.version == 'vid':
    #zs = torch.randn([args.grid_len, G.z_dim], device='cuda')
    create_vids(args, None, annotated, G, deformator)
if args.version == 'all_grid':
    zs = torch.randn([args.grid_len, G.z_dim], device='cuda')
    for (dim1, dim2) in list(itertools.combinations(range(args.dims), 2))[250:]:
        create_grids(zs, args, None, dim1, dim2, G, deformator)
if args.version == 'grid':
    zs = torch.randn([args.grid_num, G.z_dim], device='cuda')
    rcas = None
    create_grids(zs, args, rcas, args.dim1, args.dim2, G, deformator)
if args.version == 'plots':
    make_plots(deformator, args, hor = True, center = False, pathfig = args.pathfig, bound = 0.2)
if args.version == 'w_stats':
    check_w_stats(args, G, deformator)
if args.version == 'calc_rca':
    calc_RCA(G, deformator, shift_predictor, args)
if args.version == 'all_dims':
    #annotated = [0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 25, 26, 30, 31, 35, 36, 40, 41]
    zs = torch.randn([args.grid_num, G.z_dim], device='cuda')
    all_dims_chart(zs, args, None, annotated, G, deformator)
    # import PIL
    # PIL.Image.MAX_IMAGE_PIXELS = 933120000
    # img = Image.open('/dataB1/tommie/recon-runs/dso_proj_all_layer/eval_f_all/chart_all_0.png')
    # basewidth = 2000
    # wpercent = (basewidth/float(img.size[0]))
    # hsize = int((float(img.size[1])*float(wpercent)))
    # img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    # img.save('/dataB1/tommie/recon-runs/dso_proj_all_layer/eval_f_all/chart_all_0_resize.png')
    
if args.version == 'compare':
    zs = torch.randn([args.grid_num, G.z_dim], device='cuda')

    deformator_2, G_2, shift_predictor_2 = load_from_dir(
            root_dir = '/dataB1/tommie/recon-runs/{}'.format('crop_proj'),
            pkl = args.pkl,
            multi_layer = False,
            all_layer = False,
            dims = 32,
            w_var = None)
    all_dims_chart(zs, args, None, annotated, G, deformator)
    all_dims_chart(zs, args, None, annotated, G_2, deformator_2)

if args.version == 'map_var':
    calc_map_var(args, G, deformator)
if args.version == 'other':
    other(args, G, deformator)
if args.version == 'dim_mul':
    create_dim_grid(args, None, args.dim1, G, deformator)




