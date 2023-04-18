from torchvision.transforms import Resize

from data.dataset import Dataset
from models.attgan import ATTGAN
from utils.options import parse_train_options

if __name__ == '__main__':
    opt = parse_train_options()
    dataset = Dataset(opt.data_dir, opt.g_truth_dir, [Resize((256, 256))])
    model = ATTGAN(is_train=True)
    model.train(dataset, opt)

