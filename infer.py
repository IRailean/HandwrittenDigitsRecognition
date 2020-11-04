import torch
from argparse import ArgumentParser
import cv2
from torchvision import transforms
from torchvision import models
from torchvision.models import densenet121
from fastai.vision.all import *

from  .get_digits import get_digits_from_image

def make_parser():
    parser = ArgumentParser(description="MongoDB to PostgreSQL migrator")

    parser.add_argument('--image', '-ip', type=str, required=True,
                        help='path to the image')
    parser.add_argument('--learner_path', '-lp', type=str, required=True,
                        help='path to the learner')
    return parser

def num_from_digits(digits):
  reversed = digits[::-1]
  sum = 0
  for i in range(len(reversed)):
    sum += int(reversed[i]) * 10**i
  return sum

def inference(image, learner):
  digits = get_digits_from_image(image)
  if digits is None:
    return 0
  y = []
  for d in digits:
    arr = torch.tensor(d[0] * 255.0).squeeze(2).float()

    if torch.cuda.is_available():
      learner.model.cuda()
      arr.cuda()
    
    res = learner.predict(arr)

    if res[0] == '0':
      if torch.max(res[2]) > 0.7:
        y.append(res[0])
      else:
        y.append(torch.topk(res[2], 2)[1][1])
    else:
      y.append(res[0])

  return num_from_digits(y)

def main():
    parser = make_parser()
    args = parser.parse_args()
    image = args.image
    if isinstance(image, str):
      inference(cv2.imread(image), load_learner(learner_path))
    elif isinstance(image, (np.ndarray, np.generic)):
      inference(image, load_learner(learner_path))

    learner_path = args.learner_path

    return inference(cv2.imread(image_path), load_learner(learner_path))

if __name__ == "__main__":
    main()
