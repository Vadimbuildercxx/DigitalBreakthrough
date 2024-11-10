import numpy as np
from transformers import AutoFeatureExtractor, SwinForImageClassification
import torch.nn.functional as F
from PIL import Image
import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

processor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224")
swin_model = SwinForImageClassification.from_pretrained(r'Q:\pythonProject\animalshack\weights\swin', local_files_only=True)
swin_model = swin_model.to(device)
swin_model.eval()

