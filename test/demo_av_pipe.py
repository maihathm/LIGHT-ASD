import ffmpeg
import numpy as np

probe = ffmpeg.probe("/dev/video0")
video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
width = int(video_stream['width'])
height = int(video_stream['height'])

out = (
    ffmpeg
    .input('/dev/video0')
    .output('pipe:', format='rawvideo', pix_fmt="yuv420p")
    .run_async(pipe_stdout=True)
)

while True:
    in_bytes = out.stdout.read(width * height * 3)
    if not in_bytes:
        break
    video = (
        np
        .frombuffer(in_bytes, np.uint8)
        .reshape([-1, height, width, 3])
    )
    print(video.shape)