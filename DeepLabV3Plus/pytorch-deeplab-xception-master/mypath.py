#mypath.py

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'skyscapes':
            return '/path/to/datasets/skyscapes/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
