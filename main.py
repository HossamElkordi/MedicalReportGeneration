import os
import argparse

import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import torch.utils.data as data

from base_cmn import BaseCMN
from datasets import NLMCXR, MIMIC
from losses import CELossTotalEval
from models import CNN, MVCNN, TNN, Classifier, ClsGenInt, ClsGen
from utils import save, load, train, test, data_to_device, data_concatenate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.manual_seed(seed=123)

RELOAD = True  # True / False
PHASE = 'INFER'  # TRAIN / TEST / INFER
DATASET_NAME = 'MIMIC'  # NIHCXR / NLMCXR / MIMIC
BACKBONE_NAME = 'DenseNet121'  # ResNeSt50 / ResNet50 / DenseNet121
MODEL_NAME = 'ClsGenInt'  # ClsGen / ClsGenInt / VisualTransformer / GumbelTransformer

if DATASET_NAME == 'MIMIC':
    EPOCHS = 50  # Start overfitting after 20 epochs
    BATCH_SIZE = 8 if PHASE == 'TRAIN' else 64  # 128 # Fit 4 GPUs
    MILESTONES = [25]  # Reduce LR by 10 after reaching milestone epochs

elif DATASET_NAME == 'NLMCXR':
    EPOCHS = 50  # Start overfitting after 20 epochs
    BATCH_SIZE = 8 if PHASE == 'TRAIN' else 64  # Fit 4 GPUs
    MILESTONES = [25]  # Reduce LR by 10 after reaching milestone epochs

else:
    raise ValueError('Invalid DATASET_NAME')


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json',
                        help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=32, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=2048, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=1000, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=7e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98), help='the weight decay.')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--noamopt_warmup', type=int, default=5000, help='.')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_agrs()

    # --- Choose Inputs/Outputs
    if MODEL_NAME in ['ClsGen', 'ClsGenInt']:
        SOURCES = ['image', 'caption', 'label', 'history']
        TARGETS = ['caption', 'label']
        KW_SRC = ['image', 'caption', 'label', 'history']
        KW_TGT = None
        KW_OUT = None

    elif MODEL_NAME == 'VisualTransformer':
        SOURCES = ['image', 'caption']
        TARGETS = ['caption']  # ,'label']
        KW_SRC = ['image', 'caption']  # kwargs of Classifier
        KW_TGT = None
        KW_OUT = None

    elif MODEL_NAME == 'GumbelTransformer':
        SOURCES = ['image', 'caption', 'caption_length']
        TARGETS = ['caption', 'label']
        KW_SRC = ['image', 'caption', 'caption_length']  # kwargs of Classifier
        KW_TGT = None
        KW_OUT = None

    else:
        raise ValueError('Invalid BACKBONE_NAME')

    # --- Choose a Dataset ---
    if DATASET_NAME == 'MIMIC':
        INPUT_SIZE = (256, 256)
        MAX_VIEWS = 2
        NUM_LABELS = 114
        NUM_CLASSES = 2

        dataset = MIMIC('/home/hoang/Datasets/MIMIC/', INPUT_SIZE, view_pos=['AP', 'PA', 'LATERAL'],
                        max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(pvt=0.9, seed=0, generate_splits=True, debug_mode=False,
                                                              train_phase=(PHASE == 'TRAIN'))

        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS,
                                                          'No' if 'history' not in SOURCES else '')

    elif DATASET_NAME == 'NLMCXR':
        INPUT_SIZE = (256, 256)
        MAX_VIEWS = 2
        NUM_LABELS = 114
        NUM_CLASSES = 2

        dataset = NLMCXR('/home/hoang/Datasets/NLMCXR/', INPUT_SIZE, view_pos=['AP', 'PA', 'LATERAL'],
                         max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(seed=123)

        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS,
                                                          'No' if 'history' not in SOURCES else '')

    else:
        raise ValueError('Invalid DATASET_NAME')

    # --- Choose a Backbone ---
    if BACKBONE_NAME == 'ResNeSt50':
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        backbone = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        FC_FEATURES = 2048

    elif BACKBONE_NAME == 'ResNet50':
        backbone = models.resnet50(pretrained=True)
        FC_FEATURES = 2048

    elif BACKBONE_NAME == 'DenseNet121':
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        FC_FEATURES = 1024

    else:
        raise ValueError('Invalid BACKBONE_NAME')

    LR = 3e-5  # Slower LR to fine-tune the model (Open-I)
    # LR = 3e-6 # Slower LR to fine-tune the model (MIMIC)
    WD = 1e-2  # Avoid overfitting with L2 regularization
    DROPOUT = 0.1  # Avoid overfitting
    NUM_EMBEDS = 256
    FWD_DIM = 256

    NUM_HEADS = 8
    NUM_LAYERS = 1

    cnn = CNN(backbone, BACKBONE_NAME)
    cnn = MVCNN(cnn)
    tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS,
              num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)

    # Not enough memory to run 8 heads and 12 layers, instead 1 head is enough
    NUM_HEADS = 1
    NUM_LAYERS = 12

    cls_model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=cnn, tnn=tnn, fc_features=FC_FEATURES,
                           embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, dropout=DROPOUT)

    # tokenizer = Tokenizer(args)
    gen_model = BaseCMN(args)

    clsgen_model = ClsGen(cls_model, gen_model, NUM_LABELS, NUM_EMBEDS)
    clsgen_model = nn.DataParallel(clsgen_model).cuda()

    if not RELOAD:
        checkpoint_path_from = 'checkpoints/{}_ClsGen_{}_{}.pt'.format(DATASET_NAME, BACKBONE_NAME, COMMENT)
        last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, clsgen_model)
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from,
                                                                                                  last_epoch,
                                                                                                  best_metric,
                                                                                                  test_metric))

    # Initialize the Interpreter module
    NUM_HEADS = 8
    NUM_LAYERS = 1

    tnn = TNN(embed_dim=NUM_EMBEDS, num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, num_layers=NUM_LAYERS,
              num_tokens=VOCAB_SIZE, num_posits=POSIT_SIZE)
    int_model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, cnn=None, tnn=tnn, embed_dim=NUM_EMBEDS,
                           num_heads=NUM_HEADS, dropout=DROPOUT)
    int_model = nn.DataParallel(int_model).cuda()

    if not RELOAD:
        checkpoint_path_from = 'checkpoints/{}_Transformer_MaxView2_NumLabel{}.pt'.format(DATASET_NAME, NUM_LABELS)
        last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, int_model)
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from,
                                                                                                  last_epoch,
                                                                                                  best_metric,
                                                                                                  test_metric))

    model = ClsGenInt(clsgen_model.module.cpu(), int_model.module.cpu(), freeze_evaluator=True)
    criterion = CELossTotalEval(ignore_index=3)

    # --- Main program ---
    train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = nn.DataParallel(model).cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)

    print('Total Parameters:', sum(p.numel() for p in model.parameters()))

    last_epoch = -1
    best_metric = 1e9

    checkpoint_path_from = 'checkpoints/{}_{}_{}_{}.pt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT)
    checkpoint_path_to = 'checkpoints/{}_{}_{}_{}.pt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT)

    if RELOAD:
        last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model, optimizer, scheduler)  # Reload
        # last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model) # Fine-tune
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from,
                                                                                                  last_epoch,
                                                                                                  best_metric,
                                                                                                  test_metric))

    if PHASE == 'TRAIN':
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(last_epoch + 1, EPOCHS):
            print('Epoch:', epoch)
            train_loss = train(train_loader, model, optimizer, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT,
                               kw_out=KW_OUT, scaler=scaler)
            val_loss = test(val_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT,
                            return_results=False)
            test_loss = test(test_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT,
                             return_results=False)

            scheduler.step()

            if best_metric > val_loss:
                best_metric = val_loss
                save(checkpoint_path_to, model, optimizer, scheduler, epoch, (val_loss, test_loss))
                print('New Best Metric: {}'.format(best_metric))
                print('Saved To:', checkpoint_path_to)
