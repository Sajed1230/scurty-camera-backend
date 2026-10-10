import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

img = cv2.imread("test1.jpeg")
faces = app.get(img)

print("faces found:", len(faces))
if faces:
    print("numbers per face:", len(faces[0].normed_embedding))
    print("first 5 numbers:", faces[0].normed_embedding[:5])