import cv2 as cv
from pathlib import Path
import torch
from torchvision import transforms 
from torchaudio.transforms import MFCC
from model.faceDetector import S3FD
from model.Model import ASD_Model
from collections import OrderedDict, deque
from threading import Thread
import ffmpeg
import numpy as np
from ASD import ASD
from loss import lossAV
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

MAX_FRAMES = 1
FPS = 25

class MyLossAV(lossAV):
    def forward(self, x, r=1):
        x = x.squeeze(1)
        x = self.FC(x)
        predLabel = torch.round(F.softmax(x, dim = -1))[:,1]
        return predLabel

class Video(Thread):
    
    def __init__(self):
        super().__init__()
        self._stream = None
        self._queue = deque([], maxlen=MAX_FRAMES)
        self.daemon = True
        self._width = 640
        self._height = 480
        self.convert = transforms.ToTensor()
        
    def run(self):
        out = (
        ffmpeg
        .input('/dev/video0', s=f"{self._width}x{self._height}", r=FPS)
        .output('pipe:', format='rawvideo', pix_fmt="bgr24")
        .run_async(pipe_stdout=True)
        )
        while True:
            in_bytes = out.stdout.read(self._width * self._height * 3)
            if not in_bytes:
                break
            data = (
                np
                .frombuffer(in_bytes, np.uint8)
                .reshape([-1, self._height, self._width, 3])
            )
            self._queue.append(data)
        
    @property
    def stream(self):
        return self._queue
    
    def _crop_faces(self, frame, bbox):
        x, y, w, h, _ = bbox
        face = frame[int(y):int(h), int(x):int(w)]
        face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
        face = cv.resize(face, (112, 112))
        return face
    
    @property
    def tensor(self):
        faces = []
        if len(self.stream) != MAX_FRAMES:
            return None
        for frame in self.stream.copy():
            frame = frame[0]
            bboxes = face_detector.detect_faces(frame)
            num_faces = len(bboxes)
            if num_faces > 1:
                return None
            for bbox in bboxes:
                face = self._crop_faces(frame, bbox)
                face = self.convert(face)
                faces.append(face)
        video_tensor = torch.cat(faces, dim=0)
        return video_tensor

class Audio(Thread):
    
    def __init__(self):
        super().__init__()
        self._stream = None
        self._queue = deque([], maxlen=int(1e6))
        self.daemon = True
        self._sr = 16000
        self._channels = 1
        self._mfcc_convert = MFCC(n_mfcc=13)
    
    def run(self):
        out = (
        ffmpeg
        .input('hw:0', f="alsa")
        .output('pipe:', format='s16le', acodec='pcm_s16le', ac=f"{self._channels}", ar=f'{self._sr}')
        .run_async(pipe_stdout=True)
        )
        chunk_size = 1024 #self._sr
        
        while True:
            in_bytes = out.stdout.read(chunk_size)
            if not in_bytes:
                break
            data = (
                np
                .frombuffer(in_bytes, np.uint8)
            )
            self._queue.extend(data)
        
    @property
    def stream(self):
        return self._queue
    
    @property
    def tensor(self):
        if FPS is None: return None
        num_samples = int(self._sr * (MAX_FRAMES / FPS))
        if len(self._queue) < num_samples: return None
        queue = list(self._queue)
        audio = queue[-1:num_samples:-1]
        audio = self.pcm2float(audio)
        audio = torch.tensor(audio, dtype=torch.float32)
        maxAudio = int(MAX_FRAMES * 4)
        audio = self._mfcc_convert(audio).T
        if audio.shape[0] < maxAudio:
            shortage    = maxAudio - audio.shape[0]
            audio     = np.pad(audio, ((0, shortage), (0,0)), 'wrap')
        audio = audio[:int(round(MAX_FRAMES * 4)),:]  
        #raw_data = queue[-1:num_samples:-1]
        return audio
    
    def pcm2float(self, sig, dtype='float32'):
        """Convert PCM signal to floating point with a range from -1 to 1.
        Use dtype='float32' for single precision.
        Parameters
        ----------
        sig : array_like
            Input array, must have integral type.
        dtype : data type, optional
            Desired (floating point) data type.
        Returns
        -------
        numpy.ndarray
            Normalized floating point data.
        See Also
        --------
        float2pcm, dtype
        """
        sig = np.asarray(sig)
        if sig.dtype.kind not in 'iu':
            raise TypeError("'sig' must be an array of integers")
        dtype = np.dtype(dtype)
        if dtype.kind != 'f':
            raise TypeError("'dtype' must be a floating point type")

        i = np.iinfo(sig.dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        return (sig.astype(dtype) - offset) / abs_max
    
if __name__ == "__main__":
    print("Cuda available:", torch.cuda.is_available())
    curr_dir = Path(__file__).parent
    face_detector = S3FD()
    
    model_path = curr_dir.joinpath("weight", "pretrain_AVA_CVPR.model")
    weights = torch.load(model_path)

    weights_nw = OrderedDict()
    for k in weights.keys():
        if not "loss" in k:
            k_nw = k.split("model.")[1]
        else:
            k_nw = k
        weights_nw[k_nw] = weights[k]
    
    del weights

    model = ASD_Model()
    asd = ASD()
    lossAV = MyLossAV()
    model.lossAV = lossAV
    model.lossV = asd.lossV 
    model.load_state_dict(weights_nw)
    model.eval()
    model = model.to("cuda")
    
    video = Video()
    audio = Audio()
    
    pil_conv = transforms.ToPILImage()
    
    video.start()
    audio.start()
    
    while True:
        try:
            video_data = video.tensor
            audio_data = audio.tensor
            if video_data is None or audio_data is None: continue
        except IndexError:
            continue
        
        
        #audio_data = torch.unsqueeze(audio_data, 0).to("cuda")
        audio_data = torch.unsqueeze(audio_data, 0).to("cuda")
        video_data = torch.unsqueeze(video_data, 0).to("cuda")
        outsAV, _ = model(audio_data, video_data)
        label = model.lossAV(outsAV)
        print(label)
        
        