import argparse
import time
from torch import optim

import models
from utils import *

from dataset import gen_dataloader
from train import train_m2p2, eval_m2p2


parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='qps',
                    help='dataset to use (default: qps)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')
args = parser.parse_args()

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True


#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.orig_d_a, hyp_params.orig_d_v, hyp_params.orig_d_l = 73, 512, 200
hyp_params.a_len, hyp_params.v_len, hyp_params.l_len = 220, 350, 610
hyp_params.layers = args.nlevels
hyp_params.use_cuda = True
#hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
#hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = 1
#hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')


if __name__ == '__main__':
    parser.add_argument('--fd', required=False, default=9, type=int, help='fold id')
    parser.add_argument('--mod', required=False, default='avl', type=str,
                        help='modalities: a,v,l, or any combination of them')
    parser.add_argument('--dp', required=False, default=0.4, type=float, help='dropout')

    ## boolean flags
    parser.add_argument('--test_mode', default=False, action='store_true',
                        help='test mode: loading a pre-trained model and calculate loss')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='print more information')

    args = parser.parse_args()
    FOLD = int(args.fd)  # fold id
    MODS = list(args.mod)  # modalities: a, v, l
    DP = args.dp

    TEST_MODE = args.test_mode
    VERBOSE = args.verbose
    # TEST_MODE = True
    VERBOSE = True
    print("FOLD", FOLD)

    #############
    args.lonly, args.aonly, args.vonly = 1, 1, 1

    # 0 - Device configuration
    config_device()

    # 1 - load dataset
    tra_loader, val_loader, tes_loader = gen_dataloader(FOLD, MODS)

    # 2 - Initialize m2p2 models
    # initialize multiple models to output the latent embeddings for a,v,l

    # initialize m2p2 models and hyper-parameters and optimizer
    m2p2_models = {}
    m2p2_models['mult'] = models.MULTModel(hyp_params)

    m2p2_params = get_hyper_params(m2p2_models)
    m2p2_optim = optim.Adam(m2p2_params, lr=LR, weight_decay=W_DECAY)
    m2p2_scheduler = optim.lr_scheduler.StepLR(m2p2_optim, step_size=STEP_SIZE, gamma=SCHE_GAMMA)

    # if VERBOSE:
    #  print('####### total m2p2 hyper-parameters ', count_hyper_params(m2p2_params))
    #  for k, v in m2p2_models.items():
    #      print(v)
    #      print(count_hyper_params(v.parameters()))

    # 3 - Initialize concat weights: w_a, w_v, w_l
    # weight_mod = {mod: 1. / len(MODS) for mod in MODS}

    # 4 - Train or Test
    if not TEST_MODE:
        min_loss_pers = 1e5
        max_acc = 0
        #### Master Procedure Start ####
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            # train m2p2 model
            train_loss_pers, train_acc = train_m2p2(m2p2_models, MODS, tra_loader, m2p2_optim, m2p2_scheduler)
            # eval and save m2p2 model
            eval_loss_pers, eval_acc = eval_m2p2(m2p2_models, MODS, val_loader)
            if eval_loss_pers < min_loss_pers:
                print(f'[SAVE MODEL] eval pers loss: {eval_loss_pers:.5f}\tmini pers loss: {min_loss_pers:.5f}'
                      f'\teval acc: {eval_acc:.4f}\tmax_acc: {max_acc:.4f}')
                min_loss_pers = eval_loss_pers
                max_acc = eval_acc
                saveModel(FOLD, m2p2_models)

            # output loss information
            end_time = time.time()

            if VERBOSE:
                epoch_mins, epoch_secs = calc_epoch_time(start_time, end_time)
                print(f'Epoch: {epoch + 1:02}/{N_EPOCHS} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain persuasion loss:{train_loss_pers:.5f}\tTrain MAE Loss:{train_acc:.5f}')
                print(f'\tEval persuasion loss:{eval_loss_pers:.5f}\tEval MAE Loss:{eval_acc:.5f}')
        #### Master Procedure End ####
    else:
        m2p2_models = loadModel(FOLD, m2p2_models)
        test_loss_pers, test_acc = eval_m2p2(m2p2_models, MODS, tes_loader)
        print(f'Test persuasion loss:{test_loss_pers:.5f}\tTest MAE Loss:{test_acc:.5f}')
        print('MSE:', round(test_loss_pers, 3))

