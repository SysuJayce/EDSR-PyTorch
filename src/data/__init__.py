from importlib import import_module

from torch.utils.data import ConcatDataset

from dataloader import MSDataLoader


# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_train = MSDataLoader(
                args,
                MyConcatDataset(datasets),
                batch_size=args.batch_size // args.accumulation_steps,
                shuffle=True,
                pin_memory=not args.cpu
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(MSDataLoader(
                args,
                testset,
                batch_size=1,
                shuffle=False,
                pin_memory=not args.cpu
            ))

