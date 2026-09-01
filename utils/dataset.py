
import torch
from torch.utils.data import Dataset

class inferencedataset(Dataset):
    def __init__(self,prompts,gts):
            self.prompts = prompts
            self.gts = gts
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.prompts[index],self.gts[index]

class trainingdataset(Dataset):
    def __init__(self,repeated_prompts,rollout_responses,repeated_ground_truths):
        self.repeated_prompts = repeated_prompts
        self.rollout_responses = rollout_responses
        self.repeated_ground_truths = repeated_ground_truths
    
    def __len__(self):
        return len(self.repeated_prompts)
    
    def __getitem__(self, index):
        return self.repeated_prompts[index],self.rollout_responses[index],self.repeated_ground_truths[index]
