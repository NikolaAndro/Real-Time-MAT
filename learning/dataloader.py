import torch
from dataloaders.datasets.taskonomy import TaskonomyLoader

def get_info(dataset):
    """ Returns dictionary with mean and std"""
    if dataset == 'taskonomy':
        return TaskonomyLoader.INFO
    else:
        raise("Taskonomy is the only dataset for now to test on.")

def get_loader(args, split, treshold=1200, customized_task_set=None):
    """Returns data loader depending on dataset and split"""
    dataset = args.dataset
    loader = None

    if customized_task_set is None:
        task_set = args.task_set
    else:
        task_set = customized_task_set

    if dataset == 'taskonomy':
        if split == 'train':
            if args.debug_mode: print('TrainLoader - Loading Taskonomy Dataset.')
            loader = torch.utils.data.DataLoader(
                TaskonomyLoader(root=args.data_dir,
                                     is_training=True,
                                     threshold=treshold,
                                      task_set=task_set,
                                      model_whitelist=None,
                                      model_limit=30,
                                      output_size=None),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                pin_memory=True, drop_last=True)

        if split == 'val':
            if args.debug_mode: print('ValidationLoader - Loading Taskonomy Dataset.')
            loader = torch.utils.data.DataLoader(
                TaskonomyLoader(root=args.data_dir,
                                     is_training=False,
                                     threshold=treshold,
                                      task_set=task_set,
                                      model_whitelist=None,
                                      model_limit=30,
                                      output_size=None),
                batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                pin_memory=True, drop_last=True)

        if split == 'adv_val':
            loader = torch.utils.data.DataLoader(
                TaskonomyLoader(root=args.data_dir,
                                     is_training=False,
                                     threshold=treshold,
                                      task_set=task_set,
                                      model_whitelist=None,
                                      model_limit=30,
                                      output_size=None),
                batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                pin_memory=True, drop_last=True)
    return loader