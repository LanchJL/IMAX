import torch
import torch.optim as optim

from models.image_extractor import get_image_extractor
from models.IMAX import IMAX
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def configure_model(args, dataset):
    image_extractor = None
    is_open = False
    model = IMAX(dataset, args)
    if dataset.open_world and not args.train_only:
        is_open = True

    model = model.to(device)

    if args.update_features:
        print('Learnable image_embeddings')
        image_extractor = get_image_extractor(arch = args.image_extractor, pretrained = True)
        image_extractor = image_extractor.to(device)

    # configuring optimizer

    model_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optim_params = [{'params':model_params}]

    if args.train_VIT:
        ie_parameters = [param for name, param in image_extractor.named_parameters()]
        optim_params.append({'params': ie_parameters,
                            'lr': args.lrg})
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)

    model.is_open = is_open

    return image_extractor, model, optimizer