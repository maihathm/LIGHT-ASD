import os
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.io import wavfile
import python_speech_features
import ffmpeg
import warnings
from tqdm import tqdm
import logging
import traceback

from model.faceDetector import S3FD
from ASD import ASD
from loss import lossAV

warnings.filterwarnings("ignore")

# Cấu hình logging: hiển thị thời gian, level và message
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # hiển thị ở console
        # Bạn có thể thêm FileHandler nếu cần ghi log vào file:
        logging.FileHandler("debug.log", mode='w', encoding='utf-8')
    ]
)

EVAL_INFERENCE = True

def extract_audio(video_path):
    """Trích xuất audio từ video (mp4) với sample rate 16000."""
    try:
        logging.debug(f"Extracting audio from {video_path}")
        out, _ = (ffmpeg.input(video_path)
                  .output('-', format='wav', ac=1, ar=16000)
                  .run(capture_stdout=True, quiet=True))
        audio_data = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
        logging.debug(f"Audio extracted successfully, length: {len(audio_data)} samples")
        return 16000, audio_data
    except Exception as e:
        logging.exception(f"Error extracting audio from {video_path}: {str(e)}")
        return None, None

def tensor_audio(audio, numFrames, fps):
    """Tính MFCC từ audio."""
    logging.debug(f"Computing MFCC for audio with {len(audio)} samples, numFrames={numFrames}, fps={fps}")
    mfcc_feat = python_speech_features.mfcc(audio, samplerate=16000, numcep=13,
                                             winlen=0.025 * 30 / fps, winstep=0.010 * 30 / fps)
    maxAudio = int(numFrames * 4)
    if mfcc_feat.shape[0] < maxAudio:
        shortage = maxAudio - mfcc_feat.shape[0]
        logging.debug(f"MFCC features too short: {mfcc_feat.shape[0]} vs expected {maxAudio}. Padding with shortage={shortage}")
        mfcc_feat = np.pad(mfcc_feat, ((0, shortage), (0, 0)), mode='wrap')
    mfcc_feat = mfcc_feat[:int(round(numFrames * 4)), :]
    logging.debug(f"MFCC features shape after processing: {mfcc_feat.shape}")
    return torch.FloatTensor(mfcc_feat)

def tensor_video(video: torch.Tensor, face_detector) -> torch.Tensor:
    """Trích xuất khuôn mặt từ video và resize về 112x112."""
    logging.debug("Converting video tensor to numpy array for face detection.")
    video_np = video.cpu().numpy()
    faces = []
    for idx, frame in enumerate(video_np):
        bboxes = face_detector.detect_faces(frame)
        logging.debug(f"Frame {idx}: Detected {len(bboxes)} faces")
        if len(bboxes) == 0:
            continue
        # Chọn bbox có confidence cao nhất
        best_bbox = max(bboxes, key=lambda b: b[4])
        x1, y1, x2, y2, conf = best_bbox
        logging.debug(f"Frame {idx}: Best bbox {best_bbox}")
        # Giả sử x2,y2 là chiều rộng và chiều cao
        face = frame[int(y1):int(y1+y2), int(x1):int(x1+x2)]
        if face.size == 0:
            logging.warning(f"Frame {idx}: Khuôn mặt trích xuất bị rỗng, bbox={best_bbox}")
            continue
        face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
        face = cv.resize(face, (112, 112))
        faces.append(face)
    if len(faces) == 0:
        logging.error("Không phát hiện được khuôn mặt nào trong video.")
        return None
    faces_tensor = torch.FloatTensor(np.stack(faces))
    logging.debug(f"Extracted {faces_tensor.shape[0]} face(s), tensor shape: {faces_tensor.shape}")
    return faces_tensor

def group_segments(labels, min_length=2, gap_threshold=10):
    """Gom nhóm các frame có nhãn 1, bắt đầu đoạn mới nếu gap >= gap_threshold."""
    speech_indices = [i for i, lab in enumerate(labels) if lab == 1]
    if not speech_indices:
        return []
    segments = []
    start = speech_indices[0]
    prev = speech_indices[0]
    for i in speech_indices[1:]:
        if i - prev < gap_threshold:
            prev = i
        else:
            if (prev - start + 1) >= min_length:
                segments.append((start, prev))
            start = i
            prev = i
    if (prev - start + 1) >= min_length:
        segments.append((start, prev))
    logging.debug(f"Grouped speech segments: {segments}")
    return segments

def extract_speech_segments(video_frames: np.ndarray, audio: np.ndarray, labels, fps, audio_sr):
    """Tách đoạn nói từ video và audio dựa trên nhãn."""
    segments = group_segments(labels)
    if not segments:
        logging.warning("Không có segment nào được xác định từ labels.")
        return None, None
    video_segments = []
    audio_segments = []
    for start, end in segments:
        seg_frames = video_frames[start:end+1]
        video_segments.append(seg_frames)
        start_time = start / fps
        end_time = (end + 1) / fps
        start_sample = int(start_time * audio_sr)
        end_sample = int(end_time * audio_sr)
        audio_seg = audio[start_sample:end_sample]
        audio_segments.append(audio_seg)
    video_concat = np.concatenate(video_segments, axis=0)
    audio_concat = np.concatenate(audio_segments, axis=0)
    logging.debug(f"Extracted speech segments: video shape {video_concat.shape}, audio shape {audio_concat.shape}")
    return video_concat, audio_concat

def process_file(video_path, audio_path, sentence, output_dir, asd_model, face_detector, device):
    try:
        logging.info(f"Processing file: {video_path}")
        read_result = torchvision.io.read_video(video_path, pts_unit='sec')
        video_tensor = read_result[0]
        video_audio = read_result[1] if len(read_result) > 2 else None
        info = read_result[-1]
        fps = info.get('video_fps', 30)
        num_frames = video_tensor.shape[0]
        logging.debug(f"Video info: {num_frames} frames at {fps} fps")

        # Audio processing
        if not audio_path or audio_path.lower() == 'nan' or not os.path.exists(audio_path):
            sr, audio_data = extract_audio(video_path)
            if audio_data is None:
                logging.error(f"Không trích xuất được audio từ {video_path}")
                return None
        else:
            sr, audio_data = wavfile.read(audio_path)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1).astype(audio_data.dtype)
        logging.debug(f"Audio data shape: {audio_data.shape}, sample rate: {sr}")

        audio_tensor = tensor_audio(audio_data.astype(np.float32), num_frames, fps)
        face_tensor = tensor_video(video_tensor, face_detector)
        if face_tensor is None or face_tensor.shape[0] == 0:
            logging.error(f"Không phát hiện được khuôn mặt trong {video_path}")
            return None

        # Điều chỉnh kích thước tensor cho input của model.
        # NOTE: tensor_video trả về shape (N, 112, 112) -> unsqueeze để có shape (1, N, 112, 112)
        audio_tensor = audio_tensor.unsqueeze(0).to(device)   # [1, T, 13]
        face_tensor = face_tensor.unsqueeze(0).to(device)       # [1, N, 112, 112]
        logging.debug(f"Audio tensor shape: {audio_tensor.shape}, Face tensor shape: {face_tensor.shape}")

        start_inference = datetime.now()
        with torch.no_grad():
            embedA = asd_model.model.forward_audio_frontend(audio_tensor)
            embedV = asd_model.model.forward_visual_frontend(face_tensor)
            out = asd_model.model.forward_audio_visual_backend(embedA, embedV)
            x = asd_model.lossAV.FC(out)
            prob = torch.softmax(x, dim=-1)
            threshold = 0.55
            pred = torch.where(prob[:, 1] >= threshold, 1, 0)
        end_inference = datetime.now()
        inference_time = (end_inference - start_inference).total_seconds()
        logging.info(f"Inference completed in {inference_time:.2f} seconds")

        pred_np = pred.cpu().numpy().flatten()
        logging.debug(f"Prediction shape: {pred_np.shape}, unique labels: {np.unique(pred_np)}")
        video_np = video_tensor.cpu().numpy()
        video_segments, audio_segments = extract_speech_segments(video_np, audio_data, pred_np, fps, sr)
        if video_segments is None:
            logging.warning(f"Không phát hiện được đoạn nói trong {video_path}")
            return None

        if audio_segments.ndim == 1:
            audio_segments = audio_segments.reshape(-1, 1)
            logging.debug(f"Reshaped audio_segments to {audio_segments.shape}")

        out_video_path = os.path.join(output_dir, os.path.basename(video_path))
        video_out_tensor = torch.from_numpy(video_segments.astype(np.uint8))
        audio_out_tensor = torch.from_numpy(audio_segments.astype(np.float32))
        audio_array = audio_out_tensor.numpy().T
        logging.debug(f"Writing video to {out_video_path}")
        torchvision.io.write_video(
            out_video_path, 
            video_out_tensor, 
            fps=30,
            audio_array=audio_array, 
            audio_fps=sr,
            video_codec="libx264", 
            audio_codec="aac"
        )

        if EVAL_INFERENCE:
            logging.info(f"Inference: {inference_time:.2f}s, Frames: {num_frames}")
            
        return os.path.basename(video_path), sentence

    except Exception as e:
        logging.exception(f"Error processing {video_path}: {str(e)}")
        logging.debug(traceback.format_exc())
        return None

def main():
    input_csv = "data.csv"
    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    face_detector = S3FD()
    asd_model = ASD()
    model_path = os.path.join(Path(__file__).parent, "weight", "pretrain_AVA_CVPR.model")
    logging.info(f"Loading model from {model_path}")
    asd_model.loadParameters(str(model_path))
    asd_model = asd_model.to(device)
    df = pd.read_csv(input_csv)
    output_rows = []
    start_time = datetime.now()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        video_path = str(row["video_path"]).strip()
        audio_path = str(row["audio_path"]).strip()
        sentence = row["sentence"]
        result = process_file(video_path, audio_path, sentence, output_dir, asd_model, face_detector, device)
        if result is not None:
            output_rows.append(result)
    end_time = datetime.now()
    logging.info(f"Tổng thời gian xử lý: {end_time - start_time}")
    output_csv = "output.csv"
    pd.DataFrame(output_rows, columns=["video", "sentence"]).to_csv(output_csv, index=False)
    logging.info(f"Lưu file CSV đầu ra: {output_csv}")

if __name__ == "__main__":
    main()
