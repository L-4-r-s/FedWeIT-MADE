def set_config(args):

    args.output_path = '/content/sample_data/outputs/'

    args.sparse_comm = True
    args.client_sparsity = 0.3
    args.server_sparsity = 0.3

    args.model ='fedweit'
    if args.task == 'non_iid_50':
        args.base_network = 'lenet'
    elif args.task == 'mnist':
        args.base_network = 'made'
    # adaptive learning rate
    args.lr_patience = 3
    args.lr_factor = 3
    args.lr_min = 1e-10

    # base network hyperparams
    if args.base_network == 'lenet':
        args.lr = 1e-3/3
        args.wd = 1e-4

    if args.base_network == 'made':
        args.lr = 1e-3 # adam learning rate
        args.wd = 1e-4

    if 'fedweit' in args.model:
        args.wd = 1e-4
        args.lambda_l1 = 1e-3
        args.lambda_l2 = 100.
        args.lambda_mask = 0

    return args

def set_data_config(args):

    args.task_path = '/content/sample_data/tasks/'

    # CIFAR10(0), CIFAR100(1), MNIST(2), SVHN(3),
    # F-MNIST((4), TrafficSign(5), FaceScrub(6), N-MNIST(7)

    if args.task in ['non_iid_50'] :
        args.datasets    = [0, 1, 2, 3, 4, 5, 6, 7]
        args.num_clients = 5
        args.num_tasks   = 10
        args.num_classes = 5
        args.frac_clients = 1.0

    elif args.task == 'mnist':
        args.only_federated = False
        args.same_masks = True #Should clients use the same masks?
        args.same_input_order = True #for FedWeITMADE: should clients use same input ordering? DOnt use when order agnostic training is active
        args.datasets = [2]
        args.hidden_layers = [500] #hidden layer shapes
        args.mnist_path = '/content/drive/MyDrive/binarized_mnist.npz'
        args.natural_input_order = False
        args.num_clients = 4
        args.num_tasks   = 2
        args.num_classes = 3
        args.frac_clients = 1.0
        args.num_masks = 1
        args.order_agn = False
        args.order_agn_step_size = 1
        args.conn_agn_step_size = 1
        args.connectivity_weights = False
        args.direct_input = True
        args.experiment = "attention"


    else:
        print('no correct task was given: {}'.format(args.task))

    return  args
