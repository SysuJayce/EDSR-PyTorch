import torch

from src import data
from src import loss
from src import model
from src import utility
from src.option import args
from src.trainer import Trainer
from src.videotester import VideoTester

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if args.data_test == "video":
    model = model.Model(args, checkpoint)
    t = VideoTester(args, model, checkpoint)
    t.test()
else:
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()
