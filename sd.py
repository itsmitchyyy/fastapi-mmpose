from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict
from PIL import Image

import io
import os
import logging
import mimetypes
import time

import mmcv
import mmengine
from mmengine.logging import print_log

# Import your MMPose functions
from pose_estimation import load_model, estimate_pose, process_video
from mmpose.apis import init_model as init_pose_estimator
from mmpose.registry import VISUALIZERS
from mmpose.utils import adapt_mmdet_pipeline

app = FastAPI()

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

class PoseResult(BaseModel):
    keypoints: List[Dict[str, float]]
    bbox: List[float]


def process_video(self, uploaded_video: str):
    assert has_mmdet, 'Please install mmdet to run the demo.'

    output_root = os.path.join(os.getcwd(), 'output')

    mmengine.mkdir_or_exist(output_root)
    output_file = os.path.join(output_root, os.path.basename(uploaded_video))

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

    input_type = mimetypes.guess_type(uploaded_video)[0].split('/')[0]

    cap = cv2.VideoCapture(uploaded_video)

    video_writer = None
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1

        print('frame_idx', frame_idx)

        if not success:
            break

        # topdown pose estimation
        pred_instances = self.process_one_image(frame, detector, pose_estimator, visualizer, 0.001)

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

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file as an image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Estimate pose
    result = estimate_pose(MODEL, image)
    
    # Convert result to a format suitable for the API response
    pose_result = {
        "keypoints": [kp.tolist() for kp in result['keypoints']],
        "bbox": result['bbox']
    }
    
    return pose_result

@app.get("/")
async def get_index():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)