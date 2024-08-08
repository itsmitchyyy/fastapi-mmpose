from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import aiofiles
import cv2
import numpy as np

import io
import os
import logging
import mimetypes
import time

import mmcv
import mmengine
from mmengine.logging import print_log


from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

app = FastAPI()

# Serve frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
    
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

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Save the uploaded video to a temporary file
    video_path = 'temp_video.mp4'
    async with aiofiles.open(video_path, 'wb') as f:
        await f.write(await file.read())
        
    output_root = os.path.join(os.getcwd(), 'output')
    mmengine.mkdir_or_exist(output_root)
    output_file = os.path.join(output_root, os.path.basename(video_path))
    
        # load model
    model_config = os.path.join('config', 'rtmdet_m_640-8xb32_coco-person.py')
    checkpoint_config = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
        
    # build detector
    detector = init_detector(model_config, checkpoint_config, device='cpu')
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # load pose estimator
    pose_model = os.path.join('config', 'td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py')
    pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'

    # build pose estimator
    pose_estimator = init_pose_estimator(
        pose_model, 
        pose_checkpoint, 
        device='cpu', 
        cfg_options=dict(
        model=dict(test_cfg=dict(output_heatmaps=False))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = 3
    pose_estimator.cfg.visualizer.alpha = 0.8
    pose_estimator.cfg.visualizer.line_width = 1
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style='mmpose')

    input_type = mimetypes.guess_type(video_path)[0].split('/')[0]

    cap = cv2.VideoCapture(video_path)

    video_writer = None
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1

        print('frame_idx', frame_idx)

        if not success:
            break

        # topdown pose estimation
        pred_instances = process_one_image(frame, detector, pose_estimator, visualizer, 0.001)

        if output_file is not None:
            frame_vis = visualizer.get_image()

            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # the size of the image with visualization may very
                # depending on the presence of heatmaps
                video_writer = cv2.VideoWriter(output_file, fourcc, 25, (frame_vis.shape[1], frame_vis.shape[0]))

            video_writer.write(mmcv.rgb2bgr(frame_vis))

        if cv2.waitKey(5) & 0xFF == 27:
            break

        time.sleep(0)

    if video_writer:
            video_writer.release()

    cap.release()

    if output_file:
        print_log(f'Video saved to {output_file}',logger='current',level=logging.INFO)

    return output_file

    
@app.get("/")
async def get_index():
    return FileResponse('static/index.html')