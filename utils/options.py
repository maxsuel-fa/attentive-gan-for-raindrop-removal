from argparse import ArgumentParser
from typing import Any

def parse_train_options() -> Any:
    """ Function to parse input train options """
    parser = ArgumentParser()

    parser.add_argument('--inputnc',
                        dest='input_nc',
                        type=int,
                        default=3,
                        help='number of input channels')

    parser.add_argument('--outputnc',
                        dest='output_nc',
                        type=int,
                        default=3,
                        help='number of output channels')

    parser.add_argument('--datadir',
                        dest='data_dir',
                        type=str,
                        help='path to the rainy images directory')

    parser.add_argument('--gtruthdir',
                        dest='g_truth_dir',
                        type=str,
                        help='path to the clear images directory')

    parser.add_argument('--cpdir',
                        dest='checkpoint_dir',
                        type=str,
                        help='path to the checkpoint directory')

    parser.add_argument('--ngpus',
                        dest='n_gpus',
                        type=int,
                        default=1,
                        help='number of gpus to use in the training')

    parser.add_argument('--startepoch',
                        dest='start_epoch',
                        type=int,
                        default=0,
                        help=''' if greater then zero, the model will load
                                 pre-trained weights that was trained up to
                                 this number of epochs''')

    parser.add_argument('--nepochs',
                        dest='n_epochs',
                        type=int,
                        default=5000,
                        help=''' number of epochs used
                                 in the training regime''')

    parser.add_argument('--lr',
                        type=float,
                        default=0.0002,
                        help=''' inital learning rate to be
                                 used in the training regime ''')

    parser.add_argument('--startdecay',
                        dest='start_decay',
                        type=int,
                        default=0,
                        help='''epoch where the learning rate should starts to
                                decay. If it is zero, then no decay will be applied''')

    parser.add_argument('--enddecay',
                        dest='end_decay',
                        type=int,
                        default=float('inf'),
                        help='''epoch where the learning rate should stop to
                                decay.''')

    parser.add_argument('--lambdaadv',
                        dest='lambda_adv',
                        type=float,
                        default=1.0,
                        help='importance of the adversarial loss')

    parser.add_argument('--lambdavgg',
                        dest='lambda_vgg',
                        type=float,
                        default=1.0,
                        help='importance of the perceptual loss')

    parser.add_argument('--lambdafm',
                        dest='lambda_fm',
                        type=float,
                        default=1.0,
                        help='importance of the feature matching loss')

    parser.add_argument('--batchsize',
                        dest='batch_size',
                        type=int,
                        default=1,
                        help='''batch size used in the
                                training regime''')

    return parser.parse_args()


def parse_test_options() -> Any:
    """ Function to parse input test options """
    parser = ArgumentParser()

    parser.add_argument('--datadir',
                        dest='data_dir',
                        type=str,
                        help='path to the rainy images directory')

    parser.add_argument('--gtruthdir',
                        dest='g_truth_dir',
                        type=str,
                        help='path to the clear images directory')

    parser.add_argument('--savedir',
                        dest='save_dir',
                        type=str,
                        help='directory where the generated images will be saved')


    parser.add_argument('--cpdir',
                        dest='checkpoint_dir',
                        type=str,
                        help='path to the checkpoint directory')

    return parser.parse_args()
