"""
IBVAP - Offline Video Processor & AI Pre-Renderer
=============================================================================
Runs the local GPU AI pipeline (YOLOv8 Pose, Vehicle YOLO, Plate YOLO,
EasyOCR, YuNet Face, and BLA Breach Logic Analytics) on raw input videos and
exports high-definition H.264 MP4 videos with complete bounding boxes,
skeletons, license plates, and HUD overlays baked in.

Usage:
    python process_and_save.py
"""

import os
import sys
import time
import cv2
import torch
import numpy as np
import imageio

# Ensure base directory is in path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from main import IBVAPSurveillanceEngine, CAMERA_CONFIGS, logger


def process_video(
    engine: IBVAPSurveillanceEngine,
    input_filename: str,
    output_filename: str,
    camera_id: str = "CAM_1",
    max_frames: int = 450,
    target_width: int = 854,
    conf_thresh: float = 0.35
):
    input_path = os.path.join(BASE_DIR, input_filename) if not os.path.isabs(input_filename) else input_filename
    output_path = os.path.join(BASE_DIR, output_filename) if not os.path.isabs(output_filename) else output_filename

    if not os.path.exists(input_path):
        print(f"[ERROR] Input video not found: {input_path}")
        return False

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open input video: {input_path}")
        return False

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_frames, max_frames) if max_frames > 0 else total_frames

    print(f"\n========================================================")
    print(f"🎬 Processing: {input_filename} -> {output_filename}")
    print(f"   Camera ID     : {camera_id}")
    print(f"   Frames to Run : {frames_to_process} / {total_frames} ({frames_to_process / src_fps:.1f}s)")
    print(f"   Target FPS    : {src_fps:.1f}")
    print(f"========================================================")

    # Initialize H.264 MP4 writer using imageio (guaranteed universal browser compatibility)
    writer = imageio.get_writer(
        output_path,
        fps=int(round(src_fps)),
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=None
    )

    frame_idx = 0
    t_start = time.time()
    cam_label = CAMERA_CONFIGS.get(camera_id.replace("CAM_", ""), {}).get("label", camera_id)

    try:
        while frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # 1. Run full GPU surveillance pipeline
            results = engine.process_frame(
                frame,
                camera_id=camera_id,
                conf_thresh=conf_thresh,
                imgsz=480,
                enable_ocr=True,
                enable_face=True,
                enable_bla=True
            )

            # 2. Draw tactical overlays (skeletons, tripwires, boxes, license plates)
            annotated = engine.draw_overlays(
                frame,
                results["detections"],
                camera_id=camera_id,
                draw_skeletons=True,
                draw_bla=True
            )

            # 3. Draw live HUD header
            elapsed_fps = (frame_idx + 1) / max(time.time() - t_start, 1e-4)
            hud_text = f"{cam_label} | GPU: {engine.device_name} | {elapsed_fps:.1f} FPS | {results['inference_time_ms']}ms"
            cv2.putText(annotated, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)

            # 4. Scale down to web-friendly resolution if needed
            if target_width and annotated.shape[1] > target_width:
                scale = target_width / annotated.shape[1]
                new_h = int(annotated.shape[0] * scale)
                # Ensure even dimensions for H.264 encoder
                new_h = new_h if new_h % 2 == 0 else new_h + 1
                annotated = cv2.resize(annotated, (target_width, new_h), interpolation=cv2.INTER_AREA)

            # 5. Convert BGR to RGB for ImageIO
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)

            frame_idx += 1
            if frame_idx % 30 == 0 or frame_idx == frames_to_process:
                pct = (frame_idx / frames_to_process) * 100
                fps_now = frame_idx / max(time.time() - t_start, 1e-4)
                print(f"  [{camera_id}] Progress: {frame_idx}/{frames_to_process} ({pct:.1f}%) | Processing Speed: {fps_now:.1f} FPS", flush=True)

    finally:
        cap.release()
        writer.close()

    total_time = time.time() - t_start
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Finished: {output_filename} ({file_size_mb:.2f} MB in {total_time:.1f}s at {frame_idx / max(total_time, 1e-4):.1f} FPS)")
    return True


def main():
    print("=" * 65)
    print("   IBVAP AI VIDEO PRE-RENDERER (H.264 MP4 GENERATOR)")
    print("   Bakes YOLOv8 Pose, ANPR, & BLA directly into video frames")
    print("=" * 65)

    # 1. Initialize GPU Engine
    engine = IBVAPSurveillanceEngine(model_path="yolov8n-pose.pt")
    print(f"Engine Ready on: [{engine.device}] ({engine.device_name})\n")

    # 2. Videos to process
    jobs = [
        {
            "input": "fence.mp4",
            "output": "processed_fence.mp4",
            "camera_id": "CAM_1",
            "max_frames": 396,  # Full video (13.2s)
            "conf_thresh": 0.35
        },
        {
            "input": "vehicle.mp4",
            "output": "processed_vehicle.mp4",
            "camera_id": "CAM_2",
            "max_frames": 450,  # 15 seconds loop with vehicles and license plates
            "conf_thresh": 0.35
        },
        {
            "input": "sample.mp4",
            "output": "processed_sample.mp4",
            "camera_id": "CAM_3",
            "max_frames": 450,  # 15 seconds loop with aerial surveillance
            "conf_thresh": 0.35
        }
    ]

    for job in jobs:
        process_video(
            engine=engine,
            input_filename=job["input"],
            output_filename=job["output"],
            camera_id=job["camera_id"],
            max_frames=job["max_frames"],
            conf_thresh=job["conf_thresh"]
        )

    print("\n🎉 All processed AI videos have been generated successfully!")


if __name__ == "__main__":
    main()
