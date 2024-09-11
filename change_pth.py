import torch
from collections import OrderedDict

model_path = 'runs/STgram-MFN(m=0.7,s=30)/model/'
input_name = 'best_checkpoint.pth.tar2'
output_name = 'best_checkpoint.pth.tarxx'

# state_dict = torch.load('runs/STgram-MFN(m=0.7,s=30)/model/best_checkpoint.pth.tar')
state_dict = torch.load(model_path + input_name, map_location=torch.device('cpu'))
state_dict_pth = state_dict['clf_state_dict']

new_state_dict = OrderedDict()
for k, v in state_dict_pth.items():
    name = k.replace('module.', '')  # remove `module.`
    if name not in ['mobilefacenet.arcface.weight']:
        new_state_dict[name] = v
    else:
        print('delete mobilefacenet.arcface.weight')

state_dict['clf_state_dict'] = new_state_dict
torch.save(state_dict, model_path + output_name)

