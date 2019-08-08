import runway
import dlib
import numpy as np
from PIL import Image
import scipy

@runway.setup
def setup():
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
  return detector, predictor

command_inputs = {
  'image': runway.image,
  'size': runway.number(min=1, default=128, max=1024),
  'padding': runway.number(default=0.5, min=0, max=1, step=0.01)
}

@runway.command('align', inputs=command_inputs, outputs={'image': runway.image})
def align(model, inputs):
  detector, predictor = model
  img = inputs['image']
  img_np = np.array(img)
  rects = detector(img_np)
  if len(rects) == 0:
    return np.zeros((256, 256), dtype=np.uint8)
  faces = dlib.full_object_detections()
  for detection in rects:
    faces.append(predictor(img_np, detection))
  return dlib.get_face_chip(img_np, faces[0], size=inputs['size'], padding=inputs['padding'])

if __name__ == "__main__":
  runway.run(port=5222)