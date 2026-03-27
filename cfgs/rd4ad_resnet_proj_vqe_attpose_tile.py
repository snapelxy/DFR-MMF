import torchvision.transforms as transforms
from vedadet.datasets.rd4ad_dataset import collect_fn

noise_type = "ori"  # "blur+noise"
kernel_size = (7, 7)
sigma = (11.0, 11.0)
random_weight = 1.0
ori_imgblur_weight = 0.0
up_skip_imgblur = "zeros"   # zeros  imgblur


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
dataconfig = dict(
    train_data=dict(
        typename='RD4ADDataset',
        samples_per_gpu=8,
        workers_per_gpu=0,
        size_divisor=16,
        data_root=r"C:\DateSet\metv\tile\train\good",
        gt_root=None,
        resize={
            "h": 256, "w": 256,
            "pad_value": 0, },
        nose={
            "type": noise_type,
            "kernel_size": kernel_size,
            "sigma": sigma,
            "random_weight": random_weight,
            "ori_imgblur_weight": ori_imgblur_weight,
        },
        seed=99,
        collate_fn=collect_fn,
        trans_img=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(
                mean=mean, std=std),]),
        mean=mean,
        std=std,
    ),
    val_data=dict(
        typename='RD4ADDataset',
        samples_per_gpu=8,
        workers_per_gpu=0,
        size_divisor=8,
        data_root=r"C:\DateSet\metv\tile\test",
        gt_root=r"C:\DateSet\metv\tile\ground_truth",
        resize={
            "h": 256, "w": 256,
            "pad_value": 0,
        },
        nose={
            "type": noise_type,
            "kernel_size": kernel_size,
            "sigma": sigma,
            "random_weight": random_weight,
            "ori_imgblur_weight": ori_imgblur_weight,
        },
        seed=99,
        collate_fn=collect_fn,
        trans_img=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(
                mean=mean, std=std), ]),
        mean=mean,
        std=std,
    ),
)

# 2. model
model = dict(
    typename='RD4AD',
    encoder=dict(
        typename='wide_resnet50_2',
        pretrained=True,
        eval=True),

    proj_net=dict(
    typename='ProjectionNet',
    in_channels = [256,512,1024],
    ),

    vqvae_net_list=dict(
    typename='VectorQuantizerList',
    num_embeddings= [2048,1024,512],
    embedding_dim = [256,512,1024],
    batch=[8,8,8],
    feature_h=[64,32,16],
    feature_w=[64,32,16],
    unique_embed=False,
    ),

    neck_att=dict(
        typename='AttM_1',
        in_channels=[0, 0, 1024],
        out_channels=[0, 0, 64],
        out_kernel=[0, 0, 7],
        pos_dim = 8,
        select= [False,False,False]
        ),

    # neck_filter=dict(
    #     typename='BN_layer',
    #     layers=3,
    #     ratio = 1,
    #     in_channels_plus=[0, 0, 64],),


    neck_filter=dict(
        typename='BN_layer',
        layers=3,
        ratio = 1,
        in_channels=[256, 512, 1024+64],
        down_strid = 2,
        ),
        

    
    decoder=dict(
        typename='de_wide_resnet50_2',
        pretrained=False,
        # out_conv={
        #     "l1":[256,256,3,2,1],
        #     "l2":[512,512,3,2,1],
        #     "l3":[1024,1024,3,2,1]
        # },
        ),
)

train_engine = dict(
    typename='TrainEngine',
    model=model,
    criterion=dict(
        typename='RD4ADCriterion',
        type="CosineSimilarity",
    ),
    optimizer=dict(
        typename='Adam',
        lr=0.005,
        betas=(0.5,0.999),
        weight_decay=0.0))

# 3.2 val engine
val_engine = dict(
    typename='SeriValEngine',
    model=model,
    test_cfg=dict(
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(
            typename='nms',
            iou_thr=0.5),
        max_per_img=100),
    eval_metric=None)

# 4. hooks
hooks = [
    dict(typename='OptimizerHook'),
    # dict(
    #     typename='StepLrSchedulerHook',
    #     step=[8, 11],
    #     warmup='linear',
    #     warmup_iters=10,
    #     warmup_ratio=0.0005),
    dict(typename='EvalHook'),
    dict(
        typename='SnapshotHook',
        interval=20),
    dict(
        typename='LoggerHook',
        interval=1)]

# 5. work modes
modes = ['train']
max_epochs = 220

# 6. checkpoint
# weights = dict(filepath=r"C:\Pro\vedat\workdir\rd4ad_resnet_vqe_attpose\epoch_40_weights.pth")
# optimizer = dict(filepath=r"C:\Pro\vedat\workdir\rd4ad_resnet_vqe_attpose\epoch_40_optim.pth")
# meta = dict(filepath=r"C:\Pro\vedat\workdir\rd4ad_resnet_vqe_attpose\epoch_40_meta.pth")

# 7. misc
seed = 0
dist_params = dict(backend='nccl')
log_level = 'INFO'
