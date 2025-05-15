#__init__.py
from dataloaders.datasets import skyscapes as skyscapes2
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'skyscapes':
        train_set = skyscapes2.SkyscapesDataset(args, split='train')
        #print("train_set:", train_set)

        val_set = skyscapes2.SkyscapesDataset(args, split='val')
        
        num_class = train_set.NUM_CLASSES

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        

        return train_loader, val_loader, num_class

    else:
        raise NotImplementedError

