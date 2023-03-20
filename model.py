import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5n.pt').eval().to(device)