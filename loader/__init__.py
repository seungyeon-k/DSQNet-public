import torch
from loader.primitive_dataset import PrimitiveDataset

def get_dataloader(data_cfg):

  # dataset
  dataset = get_dataset(data_cfg)
  normalizer = dataset.normalizer

  # dataloader   
  loader = torch.utils.data.DataLoader(
      dataset, 
      batch_size = data_cfg["batch_size"], 
      num_workers = data_cfg["num_workers"], 
      shuffle = True)
  return loader, normalizer

def get_dataset(data_cfg):

  name = data_cfg['loader']
  dataset = _get_dataset_instance(name)

  return dataset(data_cfg)

def _get_dataset_instance(name):
  try:
      return {
          "primitivedataset": get_primitive_dataset
      }[name]
  except:
      raise ("Loss {} not available".format(name))

def get_primitive_dataset(data_cfg):
  dataset = PrimitiveDataset(data_cfg)
  return dataset
  