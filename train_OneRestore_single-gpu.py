import os, time, torch, argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
import numpy as np
from torchvision import transforms
from makedataset import Dataset
from utils.dynamic_text import get_dynamic_label
from utils.utils import print_args, load_restore_ckpt_with_optim, load_embedder_ckpt, adjust_learning_rate, data_process, tensor_metric, load_excel, save_checkpoint
from model.loss import Total_loss

from PIL import Image

transform_resize = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
        ]) 

def main(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('> Model Initialization...')

    embedder = load_embedder_ckpt(device, freeze_model=True, ckpt_name=args.embedder_model_path)
    restorer, optimizer, cur_epoch = load_restore_ckpt_with_optim(device, freeze_model=False, ckpt_name=args.restore_model_path, lr=args.lr)
    loss = Total_loss(args)
    
    print('> Loading dataset...')
    data = Dataset(args.train_input)
    dataset = DataLoader(dataset=data, num_workers=args.num_works, batch_size=args.bs, shuffle=True)
    
    print('> Start training...')
    start_all = time.time()
    train(restorer, embedder, optimizer, loss, cur_epoch, args, dataset, device)
    end_all = time.time()
    print('Whloe Training Time:' +str(end_all-start_all)+'s.')

def train(restorer, embedder, optimizer, loss, cur_epoch, args, dataset, device):

    metric = []
    for epoch in range(cur_epoch, args.epoch):
        optimizer = adjust_learning_rate(optimizer, epoch, args.adjust_lr)
        learnrate = optimizer.param_groups[-1]['lr']
        restorer.train()

        for i, data in enumerate(dataset,0):
            # print(data)
            patch, caption= data
            pos, inp, neg = data_process(patch, caption, args, device)

            text_embedding,_,_ = embedder(inp[1],'text_encoder')
            out = restorer(inp[0], text_embedding)

            restorer.zero_grad()
            total_loss = loss(inp, pos, neg, out)
            total_loss.backward()
            optimizer.step()

            mse = tensor_metric(pos,out, 'MSE', data_range=1)
            psnr = tensor_metric(pos,out, 'PSNR', data_range=1)
            ssim = tensor_metric(pos,out, 'SSIM', data_range=1)

            print("[epoch %d][%d/%d] lr :%f Floss: %.4f MSE: %.4f PSNR: %.4f SSIM: %.4f"%(epoch+1, i+1, \
                len(dataset), learnrate, total_loss.item(), mse, psnr, ssim))
        psnr_val, ssim_val = test(args, restorer, embedder, device, epoch)
        metric.append([psnr_val, ssim_val])
        print("[epoch %d] Test images PSNR: %.4f SSIM: %.4f"
              % (epoch + 1, psnr_val, ssim_val))

        # lưu log + checkpoint
        load_excel(metric)
        # save_checkpoint(
        #     {
        #         "epoch": epoch + 1,
        #         "state_dict": restorer.state_dict(),
        #         "optimizer": optimizer.state_dict()
        #     },
        #     args.save_model_path,
        #     epoch + 1,
        #     psnr_val, ssim_val
        # )

def test(args, restorer, embedder, device, epoch=-1):
    combine_type = args.degr_type
    psnr, ssim = 0, 0
    os.makedirs(args.output,exist_ok=True)

    for i in range(len(combine_type)-1):
        file_list =  os.listdir(f'{args.test_input}/{combine_type[i+1]}/')
        for j in range(len(file_list)):
            hq = Image.open(f'{args.test_input}/{combine_type[0]}/{file_list[j]}')
            lq = Image.open(f'{args.test_input}/{combine_type[i+1]}/{file_list[j]}')
            caption = get_dynamic_label(mode='test', degradation=combine_type[i+1], filename=file_list[j])
            restorer.eval()
            with torch.no_grad():
                lq_tensor = torch.from_numpy((np.array(lq)/255).transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                hq_tensor = torch.from_numpy((np.array(hq)/255).transpose(2, 0, 1)).unsqueeze(0).float().to(device)

                starttime = time.time()

                # dynamic caption → embedding
                text_embedding, _, _ = embedder([caption], "text_encoder")

                # restore
                out = restorer(lq_tensor, text_embedding)

                endtime = time.time()
            psnr += tensor_metric(hq_tensor, out, "PSNR", data_range=1)
            ssim += tensor_metric(hq_tensor, out, "SSIM", data_range=1)
            print(f"The {file_list[j][:-4]} Time: {endtime - starttime:.3f}s.")
    total_files = len(file_list)*len(combine_type)
    
    return psnr / total_files, ssim / total_files

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "OneRestore Training")

    # load model
    parser.add_argument("--embedder-model-path", type=str, default = "./ckpts/embedder_model.tar", help = 'embedder model path')
    parser.add_argument("--restore-model-path", type=str, default = None, help = 'restore model path')
    parser.add_argument("--save-model-path", type=str, default = "./ckpts/", help = 'restore model path')

    parser.add_argument("--epoch", type=int, default = 300, help = 'epoch number')
    parser.add_argument("--bs", type=int, default = 4, help = 'batchsize')
    parser.add_argument("--lr", type=float, default = 1e-4, help = 'learning rate')
    parser.add_argument("--adjust-lr", type=int, default = 30, help = 'adjust learning rate')
    parser.add_argument("--num-works", type=int, default = 4, help = 'number works')
    parser.add_argument("--loss-weight", type=tuple, default = (0.6,0.3,0.1), help = 'loss weights')
    parser.add_argument("--degr-type", type=list, default = ['clear', 'low', 'haze', 'rain', 'snow',\
        'low_haze', 'low_rain', 'low_snow', 'haze_rain', 'haze_snow', 'low_haze_rain', 'low_haze_snow'], help = 'degradation type')
    
    parser.add_argument("--train-input", type=str, default = "./dataset.h5", help = 'train data')
    parser.add_argument("--test-input", type=str, default = "./data/CDD-11_test", help = 'test path')
    parser.add_argument("--output", type=str, default = "./result/", help = 'output path')

    argspar = parser.parse_args()

    print_args(argspar)
    main(argspar)
