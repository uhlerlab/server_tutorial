import trainer
import numpy as np
import torch
import dataset
import options_parser as op

def main(options):
    seed = options.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    frames, targets = dataset.make_dataset()

    trainer.train_net(frames, targets)

if __name__ == "__main__":
    options = op.setup_options()
    main(options)
