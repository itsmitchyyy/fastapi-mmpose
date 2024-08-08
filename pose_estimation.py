
import mmcv
import mmengine
import numpy as np

from mmpose.apis import init_model, inference_topdown
from mmengine.config import Config
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples

def process_one_image(self, img, detector, pose_estimator, visualizer=None, show_interval=0):
    # predict box
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.3)]
    bboxes = bboxes[nms(bboxes, 0.3), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

        if visualizer is not None:
            visualizer.add_datasample(
                'result',
                img,
                data_sample=data_samples,
                draw_gt=False,
                draw_bbox=False,
                draw_heatmap=False,
                show_kpt_idx=False,
                skeleton_style='mmpose',
                show=False,
                wait_time=show_interval,
                out_file=None,
                kpt_thr=0.3)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)   

def load_model(config_file, checkpoint_file):
    cfg = Config.fromfile(config_file)
    model = init_model(cfg, checkpoint_file, device='cpu')
    return model

def estimate_pose(model, image):
    image_np = np.array(image)
    result = inference_topdown(model, image_np, bbox_thr=0.3)
    return result