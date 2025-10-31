optimizer_wrapper = dict(
    optimizer = dict(
        type='AdamW',
        lr=2e-4, # FIXME
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1)}
    ),
)
grad_max_norm = 35
amp = False
seed = 1
print_freq = 1
eval_freq = 1
max_epochs = 5
load_from = '/home/wyq/WorkSpace/checkpoints_from/global_0201/latest.pth' # path/to/embodiedcheckpoint
find_unused_parameters = True
track_running_stats = True
flag_depthanything_as_gt = True

ignore_label = 0
empty_idx = 12   # 0 ignore, 1~11 objects, 12 empty
cls_dims = 13

pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] 
scale_range = [0.01, 0.08]
image_size = [480, 640]
resize_lim = [1.0, 1.0] 
num_frames = 1
offset = 0
grad_frames = None

_dim_ = 96
num_cams = 1
num_heads = 3
num_levels = 4
num_anchor = 16200 
num_anchor_init = 8100
num_cross_layer = 3
num_self_layer = 3
num_decoder_fillhead = 2
semantics_activation = 'identity'
use_camera_embed = False

anchor_encoder = dict(
    type='SparseGaussian3DEncoder',
    embed_dims=_dim_, 
    semantic_dim=cls_dims-1,
)

refine_layer = dict(
    type='SparseGaussian3DDeltaRefinementModule',
    embed_dims=_dim_,
    pc_range=pc_range,
    scale_range=scale_range,
    restrict_xyz=True,
    unit_xyz=[0.1, 0.1, 0.06], # FIXME
    refine_manual=[0, 1, 2],
    semantic_dim=cls_dims-1,
    semantics_activation=semantics_activation,
)

confidence_refine_layer = dict(
    type='SparseGaussian3DDeltaConfidenceRefinementModule',
    embed_dims=_dim_,
    pc_range=pc_range,
    scale_range=scale_range,
    restrict_xyz=True,
    unit_xyz=[0.1, 0.1, 0.06], # FIXME
    refine_manual=[0, 1, 2],
    semantic_dim=cls_dims-1,
    semantics_activation=semantics_activation,
)

frozen_refine_layer = dict(
    type='SparseGaussian3DDeltaForzenRefinementModule',
    embed_dims=_dim_,
    pc_range=pc_range,
    scale_range=scale_range,
    restrict_xyz=True,
    unit_xyz=[0.1, 0.1, 0.06], # FIXME
    refine_manual=[0, 1, 2],
    semantic_dim=cls_dims-1,
    semantics_activation=semantics_activation,
)

spconv_layer=dict(
    type='SparseConv3D',
    in_channels=_dim_,
    embed_channels=_dim_,
    pc_range=pc_range,
    grid_size=[0.08]*3, # FIXME
    kernel_size=3,
)

spconv_layer_fillhead=dict(
    type='SparseConv3D',
    in_channels=_dim_,
    embed_channels=_dim_,
    pc_range=pc_range,
    grid_size=[0.08]*3,
    kernel_size=3,
    dilation=2
)

model = dict(
    type='GaussianSegmentorOnline',
    reuse_instance_feature=False,
    flag_depthbranch=True,
    flag_depthanything_as_gt=flag_depthanything_as_gt,
    gaussian_scale_max=scale_range[1],
    semantic_dim=cls_dims-1,
    backbone=None,
    neck=None,
    lifter=dict(
        type='GaussianNewLifterOnline',
        reuse_instance_feature=False,
        embed_dims=_dim_,
        num_anchor=num_anchor,
        semantic_dim=cls_dims-1,
        include_v=False),
    encoder=dict(
        type='SparseGaussianFormerOnline',
        anchor_encoder=anchor_encoder,
        norm_layer=dict(type="LN", normalized_shape=_dim_),
        ffn=dict(
            type='FFN',
            embed_dims=_dim_,
            feedforward_channels=_dim_*2,
            num_fcs=2,
            act_cfg=dict(type='ReLU', inplace=True),
        ),
        deformable_model=dict(
            type='DeformableFeatureAggregation',
            embed_dims=_dim_,
            num_groups=num_heads,
            num_levels=num_levels,
            num_cams=num_cams,
            attn_drop=0.15,
            kps_generator=dict(
                type="SparseGaussian3DKeyPointsGenerator",
                num_learnable_pts=0,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
                pc_range=pc_range,
                scale_range=scale_range),
            use_deformable_func=True,
            use_camera_embed=use_camera_embed,
            residual_mode="add",
        ),
        refine_layer=refine_layer,
        confidence_refine_layer=confidence_refine_layer,
        frozen_refine_layer=frozen_refine_layer,
        num_decoder=3,
        spconv_layer=spconv_layer,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "frozen_refine",
        ] + [
            "spconv",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "frozen_refine",
        ] + [
            "spconv",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "confidence_refine",
            ]),
    head=dict(
        type='GaussianOccHeadLocal',
        empty_label=empty_idx,
        num_classes=cls_dims,
        cuda_kwargs=dict(
            scale_multiplier=3,
            H=60, W=60, D=36,
            pc_min=[-51.2, -51.2, -5.0],
            grid_size=0.08), # FIXME
        with_empty=True,
        empty_args=dict(
            mean=[0, 0, 0],
            scale=[1e5, 1e5, 1e5],
            ),
        pc_range=pc_range,
        scale_range=scale_range,
        semantics_activation=semantics_activation
    ),
    globalhead=dict(
        type='GaussianOccHeadGlobal',
        empty_label=empty_idx,
        num_classes=cls_dims,
        cuda_kwargs=dict(
            scale_multiplier=3,
            H=200, W=220, D=90,
            pc_min=[-51.2, -51.2, -5.0],
            grid_size=0.08), # FIXME
        with_empty=True,
        empty_args=dict(
            mean=[0, 0, 0],
            scale=[1e5, 1e5, 1e5],
            ),
        pc_range=pc_range,
        scale_range=scale_range,
        semantics_activation=semantics_activation
    ))


loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='GlobalFocalLoss',
            weight=100.0, # FIXME
            gamma=2.0,
            alpha=0.25,
            cls_freq=[5080655412, 722756, 44793226, 41084591, 3416464, 21897101, 10609339, 13846320, 23470172, 263393, 30949122, 9871618, 3196722886],
            ignore_label=ignore_label,
            input_dict={
                'pred': 'ce_input',
                'target': 'ce_label'}),
        dict(
            type='GlobalLovaszLoss',
            weight=1.0,
            ignore_label=ignore_label,
            input_dict={
                'lovasz_input': 'ce_input',
                'lovasz_label': 'ce_label'}),
        dict(
            type='Global_Sem_Scal_Loss',
            weight=1.0,
            ignore_label=ignore_label,
            sem_cls_range=[1, 12], # FIXME
            input_dict={
                'pred': 'ce_input',
                'ssc_target': 'ce_label'}),
        dict(
            type='Global_Geo_Scal_Loss',
            weight=1.0,
            empty_idx=empty_idx,
            ignore_label=ignore_label,
            input_dict={
                'pred': 'ce_input',
                'ssc_target': 'ce_label'}),
    ]
)

train_dataset_config = dict(
    type='Scannet_Online_SceneOcc_Dataset',
    num_frames = num_frames,
    empty_idx = empty_idx,
    phase='train',
    num_pts=num_anchor_init,
    data_tag='base', # 'mini' for mini-set
)

val_dataset_config = dict(
    type='Scannet_Online_SceneOcc_Dataset',
    num_frames = num_frames,
    empty_idx=empty_idx,
    phase='test',
    num_pts=num_anchor_init,
    data_tag='base', # 'mini' for mini-set
)

train_wrapper_config = dict(
    type='Scannet_Online_SceneOcc_DatasetWrapper',
    final_dim = [480, 640], 
    resize_lim = resize_lim,
    phase='train', 
)

val_wrapper_config = dict(
    type='Scannet_Online_SceneOcc_DatasetWrapper',
    final_dim = [480, 640],
    resize_lim = resize_lim,
    phase='test', 
)

train_loader_config = dict(
    batch_size = 1,
    shuffle = True,
    num_workers = 8,
)
    
val_loader_config = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 2,
)