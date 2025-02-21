import ffmpeg

video = ffmpeg.input('/dev/video0', r=10, input_format='yuyv422', s="1280x720")
audio = ffmpeg.input('hw:0', f='alsa')
stream = ffmpeg.concat(video, audio ,v=1, a=1).node
video, audio = stream[0], stream[1]
out = ffmpeg.output(video, audio, 'output.avi')

ffmpeg.run(out, capture_stdout=True)