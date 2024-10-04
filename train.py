import configargparse
from dataset import SimDataset
from model import GridModel, NCA, TrainParams
from network import Conv3DNetwork
from pathlib import Path


def parse_args():
    p = configargparse.ArgParser()
    p.add_argument('-c', '--my-config', required=True, is_config_file=True, help='config file path')

    # general params
    p.add_argument('--exp_name', type=str)

    # data params
    p.add_argument('--linear_normalize', action='store_true')
    p.add_argument('--logarithmic_normalize', action='store_true')

    # grid params:
    p.add_argument('--resolution', type=int, default=1, help='Resolution of the grid. 1 (minimum) is when each receptor is one cell away')
    p.add_argument('--static_encoding_size', type=int, default=10)
    p.add_argument('--dynamic_encoding_size', type=int, default=10)
    p.add_argument('--add_extra_layer', type=bool, default=True)
    p.add_argument('--depth_ratio', type=float, default=1.0, help='What is the depth/width ratio')
    p.add_argument('--encode_depth', type=bool, default=True, help='Should initial static state contain depth info')
    p.add_argument('--zero_dynamic_state', type=bool, default=True, help='Should the dynamic state be zeroed before each run')

    # network params
    p.add_argument('--kernel_size', type=int, default=3, help='i.e. - the neighborhood from which the network is getting info')
    p.add_argument('--channels', type=int, default=128, help='how many different channels to extract for each cell')
    p.add_argument('--mlp_width', type=int, default=256)
    p.add_argument('--mlp_depth', type=int, default=2)

    # train params
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--timestamp_batches', type=int, default=20),
    p.add_argument('--save_model_n_step', type=int, default=0, help='after how many steps to save model')
    p.add_argument('--lr', type=float, default='1e-3')
    p.add_argument('--homogeneity_loss_weight', type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    dataset = SimDataset(linear_normalize=args.linear_normalize, logarithmic_normalize=args.logarithmic_normalize)

    g = GridModel(
        static_encoding_size=args.static_encoding_size,
        dynamic_encoding_size=args.dynamic_encoding_size,
        receptors=dataset.get_receptors(),
        resolution=args.resolution,
        source_distance_multiplier=4,
        add_extra_layer=args.add_extra_layer,
        encode_depth=args.encode_depth,
        zero_dynamic_state=args.zero_dynamic_state,
        depth=args.depth_ratio
    )
    (output := Path('./experiments') / Path(args.exp_name)).mkdir(exist_ok=True, parents=True)

    network = Conv3DNetwork(
        kernel_size=args.kernel_size,
        encoding_size=args.static_encoding_size + args.dynamic_encoding_size,
        channels=args.channels,
        mlp_depth=args.mlp_depth,
        mlp_width=args.mlp_width,
        output_size=args.dynamic_encoding_size,
    )
    nca = NCA(grid_model=g, network=network, output_path=output)

    train_params = TrainParams(
        epochs=args.epochs, timestamp_batches=args.timestamp_batches, save_model_n_step=args.save_model_n_step,
        lr=args.lr, homogeneity_weight=args.homogeneity_loss_weight
    )
    nca.train(dataset=dataset, params=train_params)


if __name__ == '__main__':
    main()


