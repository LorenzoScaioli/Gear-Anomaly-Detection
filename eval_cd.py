from argparse import ArgumentParser
import torch
from models.evaluator import *

import models.basic_model as basic # import CDEvaluator

print(torch.cuda.is_available())


"""
eval the CD model
"""

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='test', type=str)
    parser.add_argument('--print_models', default=False, type=bool, help='print models')

    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="test", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8_dedim8', type=str,
                        help='base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')

    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    ###
    # FOR BASIC model
    parser.add_argument('--output_folder', default='samples/predict', type=str)
    ###

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join('checkpoints', args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split=args.split)
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models(checkpoint_name=args.checkpoint_name)

    # Free up GPU memory
    del model
    torch.cuda.empty_cache()

    model_basic = basic.CDEvaluator(args)
    model_basic.load_checkpoint(args.checkpoint_name)
    model_basic.eval()

    for i, batch in enumerate(dataloader):
        name = batch['name']
        print('process: %s' % name)
        # score_map = model_basic._forward_pass(batch)
        # model_basic._save_predictions()

        with torch.no_grad():
            score_map = model_basic._forward_pass(batch)
            model_basic._save_predictions()

        # Optionally clean up
        del score_map
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

