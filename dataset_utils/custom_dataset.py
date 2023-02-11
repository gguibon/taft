from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, listjsons):
      self.data = listjsons

    def __len__(self):
      return len(self.data)
    
    def __getitem__(self, idx):
      return self.data[idx]

    def get_labels(self):
      labels = [ el['labels'].item() for el in self.data ]
      return labels