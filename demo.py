import cv2 as cv
import numpy as np

mat = np.full([640, 480, 3], 255, dtype=np.uint8)
cv.imshow("test", mat)

import torch
from pathlib import Path
from model.faceDetector import S3FD
from model.Model import ASD_Model
from collections import OrderedDict

if __name__ == "__main__":
    print("Cuda available:", torch.cuda.is_available())
    curr_dir = Path(__file__).parent

    face_dectecor = S3FD()

    model_path = curr_dir.joinpath("weight", "pretrain_AVA_CVPR.model")
    weights = torch.load(model_path)

    weights_nw = OrderedDict()
    for k in weights.keys():
        try:
            k_nw = k.split("model.")[1]
            weights_nw[k_nw] = weights[k]
        except IndexError:
            continue
    
    del weights

    model = ASD_Model()
    model.load_state_dict(weights_nw)
    model.eval()


    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            bboxes = face_dectecor.detect_faces(frame)
            for bbox in bboxes:
                x, y, w, h, _ = bbox
                cv.rectangle(frame, [int(x), int(y)], [int(w), int(h)], (255, 0, 0))
            cv.imshow("Demo", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()