import cv2
import numpy as np
from insightface.app import FaceAnalysis

_model = None


def _get_model():
    # loaded the first time it is needed, so Flask starts fast
    global _model
    if _model is None:
        _model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _model.prepare(ctx_id=-1, det_size=(640, 640))
    return _model


def get_embedding(image_bytes):
    """Returns (vector, error). The photo must have exactly one face."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, "not a valid image"

    faces = _get_model().get(img)
    if len(faces) == 0:
        return None, "no face found"
    if len(faces) > 1:
        return None, "more than one face"

    return faces[0].normed_embedding.astype(np.float32), None


def get_faces(frame):
    """Every face in one video frame: a box and a signature for each."""
    found = _get_model().get(frame)
    return [
        {
            "box": [float(v) for v in f.bbox],
            "vector": f.normed_embedding.astype(np.float32),
        }
        for f in found
    ]