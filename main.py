from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import aiofiles
import cv2
import numpy as np
from mmpose.apis import init_model, inference_topdown

app = FastAPI()

# Serve frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the MMPose model
pose_model = init_model('config/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py', 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth')

# Define a basic skeleton for visualization (using COCO keypoints as an example)
COCO_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7),
    (0, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13)
]

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        upload_path = f"static/{file.filename}"
        async with aiofiles.open(upload_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # Process the video
        video_path = upload_path
        cap = cv2.VideoCapture(video_path)
        output_path = f"static/annotated_{file.filename}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform pose detection
            pose_results, _ = inference_top_down_pose_model(pose_model, [frame_rgb])

            # Draw pose annotations on the frame
            for result in pose_results:
                keypoints = result['keypoints']
                for i, (x, y, score) in enumerate(keypoints):
                    if score > 0.5:  # confidence threshold
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                # Draw skeleton
                for start, end in COCO_SKELETON:
                    if keypoints[start][2] > 0.5 and keypoints[end][2] > 0.5:
                        cv2.line(
                            frame,
                            (int(keypoints[start][0]), int(keypoints[start][1])),
                            (int(keypoints[end][0]), int(keypoints[end][1])),
                            (0, 255, 0), 2
                        )

            # Write the frame to the output video
            out.write(frame)

        cap.release()
        out.release()
        return JSONResponse(content={"message": "Video processed successfully", "output_path": output_path})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)