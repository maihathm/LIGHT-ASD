import cv2 as cv
import torch
from pathlib import Path
from model.faceDetector import S3FD
from model.Model import ASD_Model
from collections import OrderedDict, deque
from torchvision import transforms
import  queue
from threading import Thread
import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
RECORD_SECONDS = 5
MAX_FRAMES = 30

class Video(Thread):
    def __init__(self) -> None:
        super().__init__()
        self._th = Thread(target=self.features, daemon=True)
        self._video_features = None
        self._convert = transforms.ToTensor()
        self._deque = deque([], maxlen=MAX_FRAMES)
        self.face_dectecor = S3FD()
        self.cap = cv.VideoCapture(0)
        self._frame = None
        self._frames =  []
        self._th.start()
    
    def run(self) -> None:
        while True:
            ret, frame = self.cap.read()
            if ret:
                self._shape = frame.shape
                self._frame = frame
                self._deque.append(self._frame)
    
    @property
    def frame(self):
        return self._frame
    
    @property
    def features(self):
        while True:
            tensors = []
            if len(self._deque) == self._deque.maxlen:
                deque = self._deque.copy()
                for i in range(len(deque)):
                    data = self._convert(deque[i])
                    C, H, W = data.size()
                    data = data.view([1, C, H, W])
                    tensors.append(data)
                self._video_features =  torch.cat(tensors, dim=0)

class Audio(Thread):
    def __init__(self) -> None:
        super().__init__()
        self.mic = pyaudio.PyAudio()
        self.frames = []
    
    def run(self) -> None:
        while True:
            self.frames = []
            self.stream = self.mic.open(format=FORMAT,
                                        channels=CHANNELS,
                                        rate=RATE,
                                        input=True,
                                        frames_per_buffer=CHUNK)
            print("* recording")
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = self.stream.read(CHUNK)
                self.frames.append(data)
            print("* done recording")
            self.stream.stop_stream()
            self.stream.close()
            self.mic.terminate()
    
    @property
    def frame(self):
        return self._frame

if __name__ == "__main__":
    print("Cuda available:", torch.cuda.is_available())
    curr_dir = Path(__file__).parent


video = Video()
video.start()
print(video)
while True:
    video_features = video.features