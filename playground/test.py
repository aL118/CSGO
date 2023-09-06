from config import *
import json
import torch
import torchvision.models as models

path = 'my_models'
file = 'first_attempt'
#tp_load_model('my_models','first_attempt')

# pytorch
model_path = os.path.join(path, file+'.json')
json_file = open(model_path, 'r')
json_string = json_file.read()
json_file.close()
config = json.loads(json_string)
print(config)
# torch.save(model.state_dict(), 'model_weights.pth')