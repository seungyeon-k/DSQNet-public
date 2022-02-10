import numpy as np
import open3d as o3d
import os
import pandas
import sys
import torch
from torch.utils.data.sampler import SubsetRandomSampler
sys.path.append("..")
from functions import lie
from training.loader.basic_dataset import BasicDataset
from training.loader.single_primitive_dataset import SinglePrimitiveDataset
from training.loader.delta_position_dataset import DeltaPositionDataset
from training.loader.delta_position_orientation_dataset import DeltaPositionOrientationDataset
from training.loader.delta_position_orientation_param_dataset import DeltaPosOriParamDataset
from training.loader.delta_position_orientation_param_noisy_dataset import DeltaPosOriParamNoisyDataset
from training.loader.total_with_sampling_dataset import TotalWithSamplingDataset
from training.loader.total_with_sampling_with_normal_dataset import TotalWithSamplingWithNormalDataset
from training.loader.total_with_sampling_with_boundary_dataset import TotalWithSamplingWithBoundaryDataset
from training.loader.final_sq_dataset import SQDataset
from training.loader.primitive_dataset import PrimitiveDataset

def get_dataloader(data_cfg):

  # dataset
  split = data_cfg['split']
  dataset = get_dataset(data_cfg, split)
  normalizer = dataset.normalizer

  # dataloader   
  loader = torch.utils.data.DataLoader(
      dataset, 
      batch_size = data_cfg["batch_size"], 
      num_workers = data_cfg["num_workers"], 
      shuffle = True)
  return loader, normalizer

def get_dataset(data_cfg, split):

  name = data_cfg['loader']
  dataset = _get_dataset_instance(name)

  return dataset(split, data_cfg)

def _get_dataset_instance(name):
  try:
      return {
          "basic": get_basic_dataset,
          "single": get_single_primitive_dataset,
          "deltaposition": get_delta_position_dataset,
          "deltapositionori": get_delta_position_orientation_dataset,
          "deltapositionoriparam": get_delta_position_orientation_param_dataset,
          "deltapositionoriparamnoisy": get_delta_position_orientation_param_noisy_dataset,
          "totalwithsampling": get_total_with_sampling_dataset,
          "totalwithsamplingwithnormal": get_total_with_sampling_with_normal_dataset,
          "totalwithsamplingwithboundary": get_total_with_sampling_with_boundary_dataset,
          "sqdataset": get_sq_dataset,
          "primitivedataset": get_primitive_dataset
      }[name]
  except:
      raise ("Loss {} not available".format(name))

def get_basic_dataset(split, data_cfg):
  dataset = BasicDataset(split, data_cfg)
  return dataset

def get_single_primitive_dataset(split, data_cfg):
  dataset = SinglePrimitiveDataset(split, data_cfg)
  return dataset

def get_delta_position_dataset(split, data_cfg):
  dataset = DeltaPositionDataset(split, data_cfg)
  return dataset

def get_delta_position_orientation_dataset(split, data_cfg):
  dataset = DeltaPositionOrientationDataset(split, data_cfg)
  return dataset

def get_delta_position_orientation_param_dataset(split, data_cfg):
  dataset = DeltaPosOriParamDataset(split, data_cfg)
  return dataset

def get_delta_position_orientation_param_noisy_dataset(split, data_cfg):
  dataset = DeltaPosOriParamNoisyDataset(split, data_cfg)
  return dataset

def get_total_with_sampling_dataset(split, data_cfg):
  dataset = TotalWithSamplingDataset(split, data_cfg)
  return dataset

def get_total_with_sampling_with_normal_dataset(split, data_cfg):
  dataset = TotalWithSamplingWithNormalDataset(split, data_cfg)
  return dataset

def get_total_with_sampling_with_boundary_dataset(split, data_cfg):
  dataset = TotalWithSamplingWithBoundaryDataset(split, data_cfg)
  return dataset

def get_sq_dataset(split, data_cfg):
  dataset = SQDataset(split, data_cfg)
  return dataset
  
def get_primitive_dataset(split, data_cfg):
  dataset = PrimitiveDataset(split, data_cfg)
  return dataset
  