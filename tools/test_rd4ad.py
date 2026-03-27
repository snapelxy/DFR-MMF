import argparse
from pathlib import Path
import torch
import numpy as np
import cv2,tqdm,json
import sys,pandas
# sys.path.append("/home/cjj/workspace/vedat-dev_mask/")
from vedacore.fileio import dump
from vedacore.misc import Config, ProgressBar, load_weights
from vedacore.parallel import MMDataParallel
from vedadet.datasets import build_dataloader, build_dataset
from vedadet.engines import build_engine
from sklearn.metrics import roc_auc_score
from skimage import measure
from sklearn.metrics import auc
from statistics import mean
# mean=(0.14677131, 0.4369392, 0.47905773)
# std=(0.13296916, 0.31260857, 0.34017286)

def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector') #_placeholder
    parser.add_argument('--config', help='test config file path',default="cfgs/rd4ad_1_placeholder_carpet.py")
    parser.add_argument('--checkpoint', help='checkpoint file',
                        default=r"workdir\rd4ad_1_placeholder_carpet\epoch_200_weights.pth")
    parser.add_argument('--out',default="./result/", help='output result file in pickle format')
    parser.add_argument('--eval',action='store_true',default=True, help='add if need to evalate the model')

    args = parser.parse_args()
    return args


def prepare(cfg, checkpoint):

    engine = build_engine(cfg.val_engine)
    load_weights(engine.model, checkpoint, map_location='cpu')

    device = torch.cuda.current_device()
    engine = MMDataParallel(
        engine.to(device), device_ids=[torch.cuda.current_device()])

    dataset = build_dataset(cfg.dataconfig.val_data, dict(test_mode=True))
    dataloader = build_dataloader(dataset, cfg["dataconfig"]["train_data"]["samples_per_gpu"],
            cfg["dataconfig"]["train_data"]["workers_per_gpu"],
            dist=False,
            seed=cfg["dataconfig"]["train_data"].get('seed', None),
            collate_fn=cfg["dataconfig"]["train_data"]["collate_fn"])

    return engine, dataloader


def compute_pro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, np.ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, np.ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pandas.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pandas.concat([df,pandas.DataFrame([{"pro": mean(pros), "fpr": fpr, "threshold": th}])], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def main(args):
    cfg = Config.fromfile(args.config)
    total_str = []
    for epi in range(200,210,20):
        tmp=Path(args.checkpoint)
        cpt =tmp.parent/f"epoch_{epi}_weights.pth"
        if not cpt.exists(): continue
        args.checkpoint = str(cpt)
        engine, data_loader = prepare(cfg, args.checkpoint)

        engine.eval()
        results = []
        dataset = data_loader.dataset
        mean=dataset.mean
        std=dataset.std
        prog_bar = ProgressBar(len(dataset))
        
        if args.eval:
            gt_list_px = []
            pr_list_px = []
            gt_list_sp = []
            pr_list_sp = []
            aupro_list = []

        for i in tqdm.tqdm(range(len(dataset))):
            data = dataset.__getitem__(i)
            img = data["img"]["input_x"]
            with torch.no_grad():
                result = engine(data)
            if args.eval or (args.out != None):
                anomaly_map,a_map_list = engine.module.model.get_anomal_map(result,256,amap_mode="a")
                anomaly_map = np.clip(a=anomaly_map,a_min=0.0 ,a_max=9999.9)
                
                if "gt_data" in data.keys():
                    gt = data["gt_data"][:,:,0]
                    gt[gt > 0.5] = 1
                    gt[gt <= 0.5] = 0
                    gt = cv2.resize(gt,(256,256))
                    if gt.sum() >0:
                        aupro_list.append(compute_pro(gt[None,...].astype(int),
                                                anomaly_map[None,...]))
                    gt_list_px.extend(gt.astype(int).ravel())
                    gt_list_sp.append(np.max(gt.astype(int)))
                    pr_list_px.extend(anomaly_map.ravel())
                    pr_list_sp.append(np.max(anomaly_map))
            if args.out != None:
                ano_map = engine.module.model.cvt2heatmap(anomaly_map*255)
                img = (img.permute(0, 2, 3, 1).cpu()*(torch.tensor(std)[None,None,None,:])+torch.tensor(mean)[None,None,None,:]).numpy()[0] * 255
                img = np.uint8(engine.module.model.min_max_norm(img)*255)
                ano_map = engine.module.model.show_cam_on_image(img, ano_map)
                img_p = Path(args.out)/(data['img_metas']['img_path'].parent.name+"_"+data['img_metas']['img_path'].name)
                img_p_temp = Path(args.out)/(data['img_metas']['img_path'].parent.name+"_"+data['img_metas']['img_path'].stem+"_b.jpg")
                if not img_p.parent.exists():
                    img_p.parent.mkdir(parents=True)
                cv2.imwrite(str(img_p), ano_map)
                tem_im =  (anomaly_map*255).astype(np.uint8)
                cv2.imwrite(img_p_temp, cv2.convertScaleAbs(tem_im, alpha=1.5))
                print(f"img writed:{str(img_p)}")

        if args.eval:
            auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
            auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
            print(f"epoch:{epi} eval result, auroc_px:{auroc_px},auroc_sp:{auroc_sp},round(np.mean(aupro_list),3):{round(np.mean(aupro_list),3)}")
            total_str.append(f"epoch:{epi} eval result, auroc_px:{auroc_px},auroc_sp:{auroc_sp},round(np.mean(aupro_list),3):{round(np.mean(aupro_list),3)}")
    for strr in total_str:
        print(strr)
    out_js = Path(args.checkpoint).parent/"eval.json"
    with open(str(out_js),"w") as jsf:
        json.dump([total_str],jsf,indent=2)
if __name__ == '__main__':
    args = parse_args()
    main(args)
