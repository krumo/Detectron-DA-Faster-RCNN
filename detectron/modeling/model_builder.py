# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Detectron model construction functions.

Detectron supports a large number of model types. The configuration space is
large. To get a sense, a given model is in element in the cartesian product of:

  - backbone (e.g., VGG16, ResNet, ResNeXt)
  - FPN (on or off)
  - RPN only (just proposals)
  - Fixed proposals for Fast R-CNN, RFCN, Mask R-CNN (with or without keypoints)
  - End-to-end model with RPN + Fast R-CNN (i.e., Faster R-CNN), Mask R-CNN, ...
  - Different "head" choices for the model
  - ... many configuration options ...

A given model is made by combining many basic components. The result is flexible
though somewhat complex to understand at first.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import importlib
import logging

from caffe2.python import core
from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.modeling.detector import DetectionModelHelper
from detectron.roi_data.loader import RoIDataLoader
import detectron.modeling.fast_rcnn_heads as fast_rcnn_heads
import detectron.modeling.keypoint_rcnn_heads as keypoint_rcnn_heads
import detectron.modeling.mask_rcnn_heads as mask_rcnn_heads
import detectron.modeling.name_compat as name_compat
import detectron.modeling.optimizer as optim
import detectron.modeling.retinanet_heads as retinanet_heads
import detectron.modeling.rfcn_heads as rfcn_heads
import detectron.modeling.rpn_heads as rpn_heads
import detectron.roi_data.minibatch as roi_data_minibatch
import detectron.utils.c2 as c2_utils
import detectron.utils.blob as blob_utils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Generic recomposable model builders
#
# For example, you can create a Fast R-CNN model with the ResNet-50-C4 backbone
# with the configuration:
#
# MODEL:
#   TYPE: generalized_rcnn
#   CONV_BODY: ResNet.add_ResNet50_conv4_body
#   ROI_HEAD: ResNet.add_ResNet_roi_conv5_head
# ---------------------------------------------------------------------------- #

def generalized_rcnn(model):
    """This model type handles:
      - Fast R-CNN
      - RPN only (not integrated with Fast R-CNN)
      - Faster R-CNN (stagewise training from NIPS paper)
      - Faster R-CNN (end-to-end joint training)
      - Mask R-CNN (stagewise training from NIPS paper)
      - Mask R-CNN (end-to-end joint training)
    """
    return build_generic_detection_model(
        model,
        get_func(cfg.MODEL.CONV_BODY),
        add_roi_box_head_func=get_func(cfg.FAST_RCNN.ROI_BOX_HEAD),
        add_roi_mask_head_func=get_func(cfg.MRCNN.ROI_MASK_HEAD),
        add_roi_keypoint_head_func=get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD),
        freeze_conv_body=cfg.TRAIN.FREEZE_CONV_BODY
    )


def rfcn(model):
    # TODO(rbg): fold into build_generic_detection_model
    return build_generic_rfcn_model(model, get_func(cfg.MODEL.CONV_BODY))


def retinanet(model):
    # TODO(rbg): fold into build_generic_detection_model
    return build_generic_retinanet_model(model, get_func(cfg.MODEL.CONV_BODY))


# ---------------------------------------------------------------------------- #
# Helper functions for building various re-usable network bits
# ---------------------------------------------------------------------------- #

def create(model_type_func, train=False, gpu_id=0):
    """Generic model creation function that dispatches to specific model
    building functions.

    By default, this function will generate a data parallel model configured to
    run on cfg.NUM_GPUS devices. However, you can restrict it to build a model
    targeted to a specific GPU by specifying gpu_id. This is used by
    optimizer.build_data_parallel_model() during test time.
    """
    model = DetectionModelHelper(
        name=model_type_func,
        train=train,
        num_classes=cfg.MODEL.NUM_CLASSES,
        init_params=train
    )
    model.only_build_forward_pass = False
    model.target_gpu_id = gpu_id
    return get_func(model_type_func)(model)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    new_func_name = name_compat.get_new_name(func_name)
    if new_func_name != func_name:
        logger.warn(
            'Remapping old function name: {} -> {}'.
            format(func_name, new_func_name)
        )
        func_name = new_func_name
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'detectron.modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: {}'.format(func_name))
        raise


def build_generic_detection_model(
    model,
    add_conv_body_func,
    add_roi_box_head_func=None,
    add_roi_mask_head_func=None,
    add_roi_keypoint_head_func=None,
    freeze_conv_body=False
):
    def _single_gpu_build_func(model):
        """Build the model on a single GPU. Can be called in a loop over GPUs
        with name and device scoping to create a data parallel model.
        """
        # Add the conv body (called "backbone architecture" in papers)
        # E.g., ResNet-50, ResNet-50-FPN, ResNeXt-101-FPN, etc.
        blob_conv, dim_conv, spatial_scale_conv = add_conv_body_func(model)
        if freeze_conv_body:
            for b in c2_utils.BlobReferenceList(blob_conv):
                model.StopGradient(b, b)

        if not model.train:  # == inference
            # Create a net that can be used to execute the conv body on an image
            # (without also executing RPN or any other network heads)
            model.conv_body_net = model.net.Clone('conv_body_net')

        head_loss_gradients = {
            'rpn': None,
            'box': None,
            'mask': None,
            'keypoints': None,
            'image': None,
            'instance': None,
            'consistency': None,
        }

        if cfg.RPN.RPN_ON:
            # Add the RPN head
            head_loss_gradients['rpn'] = rpn_heads.add_generic_rpn_outputs(
                model, blob_conv, dim_conv, spatial_scale_conv
            )

        if cfg.FPN.FPN_ON:
            # After adding the RPN head, restrict FPN blobs and scales to
            # those used in the RoI heads
            blob_conv, spatial_scale_conv = _narrow_to_fpn_roi_levels(
                blob_conv, spatial_scale_conv
            )

        if not cfg.MODEL.RPN_ONLY:
            # Add the Fast R-CNN head
            head_loss_gradients['box'], blob_feats_rois_all, dim_feats_rois_all = _add_fast_rcnn_head(
                model, add_roi_box_head_func, blob_conv, dim_conv,
                spatial_scale_conv
            )

        if cfg.MODEL.MASK_ON:
            # Add the mask head
            head_loss_gradients['mask'] = _add_roi_mask_head(
                model, add_roi_mask_head_func, blob_conv, dim_conv,
                spatial_scale_conv
            )

        if cfg.MODEL.KEYPOINTS_ON:
            # Add the keypoint head
            head_loss_gradients['keypoint'] = _add_roi_keypoint_head(
                model, add_roi_keypoint_head_func, blob_conv, dim_conv,
                spatial_scale_conv
            )
        
        if cfg.TRAIN.DOMAIN_ADAPTATION:
            # Add Image-level loss
            head_loss_gradients['image'] = _add_image_level_classifier(model, blob_conv, dim_conv, spatial_scale_conv)
            # Add Instance-level loss
            head_loss_gradients['instance'] = _add_instance_level_classifier(model, blob_feats_rois_all, dim_feats_rois_all)
            # Add consistency regularization
            head_loss_gradients['consistency'] = _add_consistency_loss(model, blob_conv, dim_conv, blob_feats_rois_all, dim_feats_rois_all)

        if model.train:
            loss_gradients = {}
            for lg in head_loss_gradients.values():
                if lg is not None:
                    loss_gradients.update(lg)
            return loss_gradients
        else:
            return None

    optim.build_data_parallel_model(model, _single_gpu_build_func)
    return model

def _add_image_level_classifier(model, blob_in, dim_in, spatial_scale_in):
    from detectron.utils.c2 import const_fill
    from detectron.utils.c2 import gauss_fill
    
    def negateGrad(inputs, outputs):
        outputs[0].feed(inputs[0].data)
    def grad_negateGrad(inputs, outputs):
        scale = cfg.TRAIN.DA_IMG_GRL_WEIGHT
        grad_output = inputs[-1]
        outputs[0].reshape(grad_output.shape)
        outputs[0].data[...] = -1.0*scale*grad_output.data
    
    model.GradientScalerLayer([blob_in], ['da_grl'], -1.0*cfg.TRAIN.DA_IMG_GRL_WEIGHT)
    model.Conv('da_grl', 'da_conv_1', dim_in, 512, kernel=1, pad=0, stride=1, weight_init=gauss_fill(0.001), bias_init=const_fill(0.0))    
    model.Relu('da_conv_1', 'da_conv_1')
    model.Conv('da_conv_1', 'da_conv_2',
        512,
        1,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    if model.train:
        model.net.SpatialNarrowAs(
            ['da_label_wide', 'da_conv_2'], 'da_label'
        )
        loss_da = model.net.SigmoidCrossEntropyLoss(
            ['da_conv_2', 'da_label'],
            'loss_da',
            scale=model.GetLossScale()
        )
        loss_gradient = blob_utils.get_loss_gradients(model, [loss_da])
        model.AddLosses('loss_da')
        return loss_gradient
    else:
        return None

def _add_instance_level_classifier(model, blob_in, dim_in):
    from detectron.utils.c2 import const_fill
    from detectron.utils.c2 import gauss_fill

    def negateGrad(inputs, outputs):
        outputs[0].feed(inputs[0].data)
    def grad_negateGrad(inputs, outputs):
        scale = cfg.TRAIN.DA_INS_GRL_WEIGHT
        grad_output = inputs[-1]
        outputs[0].reshape(grad_output.shape)
        outputs[0].data[...] = -1.0*scale*grad_output.data
    model.GradientScalerLayer([blob_in], ['dc_grl'], -1.0*cfg.TRAIN.DA_INS_GRL_WEIGHT)
    model.FC('dc_grl', 'dc_ip1', dim_in, 1024,
             weight_init=gauss_fill(0.01), bias_init=const_fill(0.0))
    model.Relu('dc_ip1', 'dc_relu_1')
    model.Dropout('dc_relu_1', 'dc_drop_1', ratio=0.5, is_test=False)

    model.FC('dc_drop_1', 'dc_ip2', 1024, 1024,
             weight_init=gauss_fill(0.01), bias_init=const_fill(0.0))
    model.Relu('dc_ip2', 'dc_relu_2')
    model.Dropout('dc_relu_2', 'dc_drop_2', ratio=0.5, is_test=False)

    dc_ip3 = model.FC('dc_drop_2', 'dc_ip3', 1024, 1,
                      weight_init=gauss_fill(0.05), bias_init=const_fill(0.0))
    loss_gradient = None
    if model.train:
        dc_loss = model.net.SigmoidCrossEntropyLoss(
            [dc_ip3, 'dc_label'],
            'loss_dc',
            scale=model.GetLossScale()
        )
        loss_gradient = blob_utils.get_loss_gradients(model, [dc_loss])
        model.AddLosses('loss_dc')
    return loss_gradient


def _add_consistency_loss(model, blob_img_in, img_dim_in, blob_ins_in, ins_dim_in):
    def expand_as(inputs, outputs):
        img_prob = inputs[0].data
        ins_prob = inputs[1].data
        import numpy as np
        mean_da_conv = np.mean(img_prob, (1,2,3))
        print(mean_da_conv)
        repeated_da_conv = np.expand_dims(np.repeat(
            mean_da_conv, ins_prob.shape[0]//2), axis=1)
        outputs[0].feed(repeated_da_conv)
    def grad_expand_as(inputs, outputs):
        import numpy as np
        img_prob = inputs[0].data
        ins_prob = inputs[1].data
        grad_output = inputs[3]
        grad_input = outputs[0]
        grad_input.reshape(inputs[0].shape)
        unit = grad_output.shape[0]//2
        grad_o = grad_output.data[...]
        grad_i = np.zeros(inputs[0].shape)
        for i in range(inputs[0].shape[0]):
            grad_i[i] = np.sum(grad_o[i*unit:(i+1)*unit, 0])*np.ones(grad_i[i].shape).astype(np.float32)/(img_prob.shape[1]*img_prob.shape[2]*img_prob.shape[3])
        grad_input.data[...] = grad_i
    
    model.GradientScalerLayer([blob_img_in], ['da_grl_copy'], 1.0*cfg.TRAIN.DA_IMG_GRL_WEIGHT)
    model.ConvShared('da_grl_copy', 'da_conv_1_copy', img_dim_in, 512, kernel=1, pad=0, stride=1, weight='da_conv_1_w', bias='da_conv_1_b')
    model.Relu('da_conv_1_copy', 'da_conv_1_copy')
    model.ConvShared('da_conv_1_copy', 'da_conv_2_copy', 512, 1, kernel=1, pad=0, stride=1, weight='da_conv_2_w', bias='da_conv_2_b')
    model.net.Sigmoid('da_conv_2_copy', 'img_probs')

    model.GradientScalerLayer([blob_ins_in], ['dc_grl_copy'], 1.0*cfg.TRAIN.DA_INS_GRL_WEIGHT)
    model.FCShared('dc_grl_copy', 'dc_ip1_copy', ins_dim_in, 1024,
          weight='dc_ip1_w', bias='dc_ip1_b')
    model.Relu('dc_ip1_copy', 'dc_relu_1_copy')

    model.FCShared('dc_relu_1_copy', 'dc_ip2_copy', 1024, 1024,
          weight='dc_ip2_w', bias='dc_ip2_b')
    model.Relu('dc_ip2_copy', 'dc_relu_2_copy')

    model.FCShared('dc_relu_2_copy', 'dc_ip3_copy', 1024, 1,
                   weight='dc_ip3_w', bias='dc_ip3_b')
    model.net.Sigmoid('dc_ip3_copy', "ins_probs")
    loss_gradient = None
    if model.train:
        model.net.Python(f=expand_as, grad_f=grad_expand_as, grad_input_indices=[0], grad_output_indices=[0])(
            ['img_probs', 'ins_probs'], ['repeated_img_probs'])
        dist = model.net.L1Distance(['repeated_img_probs', 'ins_probs'], ['consistency_dist'])
        # dist = model.net.SquaredL2Distance(['repeated_img_probs', 'ins_probs'], ['consistency_dist'])
        loss_consistency = model.net.AveragedLoss(dist, 'loss_consistency')
        loss_gradient = blob_utils.get_loss_gradients(
            model, [loss_consistency])
        model.AddLosses('loss_consistency')

    return loss_gradient


def _narrow_to_fpn_roi_levels(blobs, spatial_scales):
    """Return only the blobs and spatial scales that will be used for RoI heads.
    Inputs `blobs` and `spatial_scales` may include extra blobs and scales that
    are used for RPN proposals, but not for RoI heads.
    """
    # Code only supports case when RPN and ROI min levels are the same
    assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
    # RPN max level can be >= to ROI max level
    assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
    # FPN RPN max level might be > FPN ROI max level in which case we
    # need to discard some leading conv blobs (blobs are ordered from
    # max/coarsest level to min/finest level)
    num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1
    return blobs[-num_roi_levels:], spatial_scales[-num_roi_levels:]


def _add_fast_rcnn_head(
    model, add_roi_box_head_func, blob_in, dim_in, spatial_scale_in
):
    """Add a Fast R-CNN head to the model."""
    blob_frcn, dim_frcn = add_roi_box_head_func(
        model, blob_in, dim_in, spatial_scale_in
    )
    fast_rcnn_heads.add_fast_rcnn_outputs(model, blob_frcn, dim_frcn)
    if model.train:
        loss_gradients = fast_rcnn_heads.add_fast_rcnn_losses(model)
    else:
        loss_gradients = None
    return loss_gradients, blob_frcn, dim_frcn


def _add_roi_mask_head(
    model, add_roi_mask_head_func, blob_in, dim_in, spatial_scale_in
):
    """Add a mask prediction head to the model."""
    # Capture model graph before adding the mask head
    bbox_net = copy.deepcopy(model.net.Proto())
    # Add the mask head
    blob_mask_head, dim_mask_head = add_roi_mask_head_func(
        model, blob_in, dim_in, spatial_scale_in
    )
    # Add the mask output
    blob_mask = mask_rcnn_heads.add_mask_rcnn_outputs(
        model, blob_mask_head, dim_mask_head
    )

    if not model.train:  # == inference
        # Inference uses a cascade of box predictions, then mask predictions.
        # This requires separate nets for box and mask prediction.
        # So we extract the mask prediction net, store it as its own network,
        # then restore model.net to be the bbox-only network
        model.mask_net, blob_mask = c2_utils.SuffixNet(
            'mask_net', model.net, len(bbox_net.op), blob_mask
        )
        model.net._net = bbox_net
        loss_gradients = None
    else:
        loss_gradients = mask_rcnn_heads.add_mask_rcnn_losses(model, blob_mask)
    return loss_gradients


def _add_roi_keypoint_head(
    model, add_roi_keypoint_head_func, blob_in, dim_in, spatial_scale_in
):
    """Add a keypoint prediction head to the model."""
    # Capture model graph before adding the mask head
    bbox_net = copy.deepcopy(model.net.Proto())
    # Add the keypoint head
    blob_keypoint_head, dim_keypoint_head = add_roi_keypoint_head_func(
        model, blob_in, dim_in, spatial_scale_in
    )
    # Add the keypoint output
    blob_keypoint = keypoint_rcnn_heads.add_keypoint_outputs(
        model, blob_keypoint_head, dim_keypoint_head
    )

    if not model.train:  # == inference
        # Inference uses a cascade of box predictions, then keypoint predictions
        # This requires separate nets for box and keypoint prediction.
        # So we extract the keypoint prediction net, store it as its own
        # network, then restore model.net to be the bbox-only network
        model.keypoint_net, keypoint_blob_out = c2_utils.SuffixNet(
            'keypoint_net', model.net, len(bbox_net.op), blob_keypoint
        )
        model.net._net = bbox_net
        loss_gradients = None
    else:
        loss_gradients = keypoint_rcnn_heads.add_keypoint_losses(model)
    return loss_gradients


def build_generic_rfcn_model(model, add_conv_body_func, dim_reduce=None):
    # TODO(rbg): fold this function into build_generic_detection_model
    def _single_gpu_build_func(model):
        """Builds the model on a single GPU. Can be called in a loop over GPUs
        with name and device scoping to create a data parallel model."""
        blob, dim, spatial_scale = add_conv_body_func(model)
        if not model.train:
            model.conv_body_net = model.net.Clone('conv_body_net')
        rfcn_heads.add_rfcn_outputs(model, blob, dim, dim_reduce, spatial_scale)
        if model.train:
            loss_gradients = fast_rcnn_heads.add_fast_rcnn_losses(model)
        return loss_gradients if model.train else None

    optim.build_data_parallel_model(model, _single_gpu_build_func)
    return model


def build_generic_retinanet_model(
    model, add_conv_body_func, freeze_conv_body=False
):
    # TODO(rbg): fold this function into build_generic_detection_model
    def _single_gpu_build_func(model):
        """Builds the model on a single GPU. Can be called in a loop over GPUs
        with name and device scoping to create a data parallel model."""
        blobs, dim, spatial_scales = add_conv_body_func(model)
        if not model.train:
            model.conv_body_net = model.net.Clone('conv_body_net')
        retinanet_heads.add_fpn_retinanet_outputs(
            model, blobs, dim, spatial_scales
        )
        if model.train:
            loss_gradients = retinanet_heads.add_fpn_retinanet_losses(
                model
            )
        return loss_gradients if model.train else None

    optim.build_data_parallel_model(model, _single_gpu_build_func)
    return model


# ---------------------------------------------------------------------------- #
# Network inputs
# ---------------------------------------------------------------------------- #

def add_training_inputs(model, source_roidb=None, target_roidb=None):
    """Create network input ops and blobs used for training. To be called
    *after* model_builder.create().
    """
    # Implementation notes:
    #   Typically, one would create the input ops and then the rest of the net.
    #   However, creating the input ops depends on loading the dataset, which
    #   can take a few minutes for COCO.
    #   We prefer to avoid waiting so debugging can fail fast.
    #   Thus, we create the net *without input ops* prior to loading the
    #   dataset, and then add the input ops after loading the dataset.
    #   Since we defer input op creation, we need to do a little bit of surgery
    #   to place the input ops at the start of the network op list.
    assert model.train, 'Training inputs can only be added to a trainable model'
    if source_roidb is not None:
        # To make debugging easier you can set cfg.DATA_LOADER.NUM_THREADS = 1
        model.roi_data_loader = RoIDataLoader(
            source_roidb=source_roidb,
            target_roidb=target_roidb,
            num_loaders=cfg.DATA_LOADER.NUM_THREADS,
            minibatch_queue_size=cfg.DATA_LOADER.MINIBATCH_QUEUE_SIZE,
            blobs_queue_capacity=cfg.DATA_LOADER.BLOBS_QUEUE_CAPACITY
        )
    orig_num_op = len(model.net._net.op)
    blob_names = roi_data_minibatch.get_minibatch_blob_names(is_training=True)
    for gpu_id in range(cfg.NUM_GPUS):
        with c2_utils.NamedCudaScope(gpu_id):
            for blob_name in blob_names:
                workspace.CreateBlob(core.ScopedName(blob_name))
            model.net.DequeueBlobs(
                model.roi_data_loader._blobs_queue_name, blob_names
            )
    # A little op surgery to move input ops to the start of the net
    diff = len(model.net._net.op) - orig_num_op
    new_op = model.net._net.op[-diff:] + model.net._net.op[:-diff]
    del model.net._net.op[:]
    model.net._net.op.extend(new_op)


def add_inference_inputs(model):
    """Create network input blobs used for inference."""

    def create_input_blobs_for_net(net_def):
        for op in net_def.op:
            for blob_in in op.input:
                if not workspace.HasBlob(blob_in):
                    workspace.CreateBlob(blob_in)

    create_input_blobs_for_net(model.net.Proto())
    if cfg.MODEL.MASK_ON:
        create_input_blobs_for_net(model.mask_net.Proto())
    if cfg.MODEL.KEYPOINTS_ON:
        create_input_blobs_for_net(model.keypoint_net.Proto())


# ---------------------------------------------------------------------------- #
# ********************** DEPRECATED FUNCTIONALITY BELOW ********************** #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Hardcoded functions to create various types of common models
#
#            *** This type of model definition is deprecated ***
#            *** Use the generic composable versions instead ***
#
# ---------------------------------------------------------------------------- #

import detectron.modeling.ResNet as ResNet
import detectron.modeling.VGG16 as VGG16
import detectron.modeling.VGG_CNN_M_1024 as VGG_CNN_M_1024


def fast_rcnn(model):
    logger.warn('Deprecated: use `MODEL.TYPE: generalized_rcnn`.')
    return generalized_rcnn(model)


def mask_rcnn(model):
    logger.warn(
        'Deprecated: use `MODEL.TYPE: generalized_rcnn` with '
        '`MODEL.MASK_ON: True`'
    )
    return generalized_rcnn(model)


def keypoint_rcnn(model):
    logger.warn(
        'Deprecated: use `MODEL.TYPE: generalized_rcnn` with '
        '`MODEL.KEYPOINTS_ON: True`'
    )
    return generalized_rcnn(model)


def mask_and_keypoint_rcnn(model):
    logger.warn(
        'Deprecated: use `MODEL.TYPE: generalized_rcnn` with '
        '`MODEL.MASK_ON: True and ``MODEL.KEYPOINTS_ON: True`'
    )
    return generalized_rcnn(model)


def rpn(model):
    logger.warn(
        'Deprecated: use `MODEL.TYPE: generalized_rcnn` with '
        '`MODEL.RPN_ONLY: True`'
    )
    return generalized_rcnn(model)


def fpn_rpn(model):
    logger.warn(
        'Deprecated: use `MODEL.TYPE: generalized_rcnn` with '
        '`MODEL.RPN_ONLY: True` and FPN enabled via configs'
    )
    return generalized_rcnn(model)


def faster_rcnn(model):
    logger.warn(
        'Deprecated: use `MODEL.TYPE: generalized_rcnn` with '
        '`MODEL.FASTER_RCNN: True`'
    )
    return generalized_rcnn(model)


def fast_rcnn_frozen_features(model):
    logger.warn('Deprecated: use `TRAIN.FREEZE_CONV_BODY: True` instead')
    return build_generic_detection_model(
        model,
        get_func(cfg.MODEL.CONV_BODY),
        add_roi_box_head_func=get_func(cfg.FAST_RCNN.ROI_BOX_HEAD),
        freeze_conv_body=True
    )


def rpn_frozen_features(model):
    logger.warn('Deprecated: use `TRAIN.FREEZE_CONV_BODY: True` instead')
    return build_generic_detection_model(
        model, get_func(cfg.MODEL.CONV_BODY), freeze_conv_body=True
    )


def fpn_rpn_frozen_features(model):
    logger.warn('Deprecated: use `TRAIN.FREEZE_CONV_BODY: True` instead')
    return build_generic_detection_model(
        model, get_func(cfg.MODEL.CONV_BODY), freeze_conv_body=True
    )


def mask_rcnn_frozen_features(model):
    logger.warn('Deprecated: use `TRAIN.FREEZE_CONV_BODY: True` instead')
    return build_generic_detection_model(
        model,
        get_func(cfg.MODEL.CONV_BODY),
        add_roi_box_head_func=get_func(cfg.FAST_RCNN.ROI_BOX_HEAD),
        add_roi_mask_head_func=get_func(cfg.MRCNN.ROI_MASK_HEAD),
        freeze_conv_body=True
    )


def keypoint_rcnn_frozen_features(model):
    logger.warn('Deprecated: use `TRAIN.FREEZE_CONV_BODY: True` instead')
    return build_generic_detection_model(
        model,
        get_func(cfg.MODEL.CONV_BODY),
        add_roi_box_head_func=get_func(cfg.FAST_RCNN.ROI_BOX_HEAD),
        add_roi_keypoint_head_func=get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD),
        freeze_conv_body=True
    )


# ---------------------------------------------------------------------------- #
# Fast R-CNN models
# ---------------------------------------------------------------------------- #


def VGG_CNN_M_1024_fast_rcnn(model):
    return build_generic_detection_model(
        model, VGG_CNN_M_1024.add_VGG_CNN_M_1024_conv5_body,
        VGG_CNN_M_1024.add_VGG_CNN_M_1024_roi_fc_head
    )


def VGG16_fast_rcnn(model):
    return build_generic_detection_model(
        model, VGG16.add_VGG16_conv5_body, VGG16.add_VGG16_roi_fc_head
    )


def ResNet50_fast_rcnn(model):
    return build_generic_detection_model(
        model, ResNet.add_ResNet50_conv4_body, ResNet.add_ResNet_roi_conv5_head
    )


def ResNet101_fast_rcnn(model):
    return build_generic_detection_model(
        model, ResNet.add_ResNet101_conv4_body, ResNet.add_ResNet_roi_conv5_head
    )


def ResNet50_fast_rcnn_frozen_features(model):
    return build_generic_detection_model(
        model,
        ResNet.add_ResNet50_conv4_body,
        ResNet.add_ResNet_roi_conv5_head,
        freeze_conv_body=True
    )


def ResNet101_fast_rcnn_frozen_features(model):
    return build_generic_detection_model(
        model,
        ResNet.add_ResNet101_conv4_body,
        ResNet.add_ResNet_roi_conv5_head,
        freeze_conv_body=True
    )


# ---------------------------------------------------------------------------- #
# RPN-only models
# ---------------------------------------------------------------------------- #


def VGG_CNN_M_1024_rpn(model):
    return build_generic_detection_model(
        model, VGG_CNN_M_1024.add_VGG_CNN_M_1024_conv5_body
    )


def VGG16_rpn(model):
    return build_generic_detection_model(model, VGG16.add_VGG16_conv5_body)


def ResNet50_rpn_conv4(model):
    return build_generic_detection_model(model, ResNet.add_ResNet50_conv4_body)


def ResNet101_rpn_conv4(model):
    return build_generic_detection_model(model, ResNet.add_ResNet101_conv4_body)


def VGG_CNN_M_1024_rpn_frozen_features(model):
    return build_generic_detection_model(
        model,
        VGG_CNN_M_1024.add_VGG_CNN_M_1024_conv5_body,
        freeze_conv_body=True
    )


def VGG16_rpn_frozen_features(model):
    return build_generic_detection_model(
        model, VGG16.add_VGG16_conv5_body, freeze_conv_body=True
    )


def ResNet50_rpn_conv4_frozen_features(model):
    return build_generic_detection_model(
        model, ResNet.add_ResNet50_conv4_body, freeze_conv_body=True
    )


def ResNet101_rpn_conv4_frozen_features(model):
    return build_generic_detection_model(
        model, ResNet.add_ResNet101_conv4_body, freeze_conv_body=True
    )


# ---------------------------------------------------------------------------- #
# Faster R-CNN models
# ---------------------------------------------------------------------------- #


def VGG16_faster_rcnn(model):
    assert cfg.MODEL.FASTER_RCNN
    return build_generic_detection_model(
        model, VGG16.add_VGG16_conv5_body, VGG16.add_VGG16_roi_fc_head
    )


def ResNet50_faster_rcnn(model):
    assert cfg.MODEL.FASTER_RCNN
    return build_generic_detection_model(
        model, ResNet.add_ResNet50_conv4_body, ResNet.add_ResNet_roi_conv5_head
    )


def ResNet101_faster_rcnn(model):
    assert cfg.MODEL.FASTER_RCNN
    return build_generic_detection_model(
        model, ResNet.add_ResNet101_conv4_body, ResNet.add_ResNet_roi_conv5_head
    )


# ---------------------------------------------------------------------------- #
# R-FCN models
# ---------------------------------------------------------------------------- #


def ResNet50_rfcn(model):
    return build_generic_rfcn_model(
        model, ResNet.add_ResNet50_conv5_body, dim_reduce=1024
    )


def ResNet101_rfcn(model):
    return build_generic_rfcn_model(
        model, ResNet.add_ResNet101_conv5_body, dim_reduce=1024
    )
