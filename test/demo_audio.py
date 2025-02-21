import ffmpeg

stream = ffmpeg.input('hw:0', f='alsa')
stream = ffmpeg.output(stream, 'output.wav')
ffmpeg.run(stream)

