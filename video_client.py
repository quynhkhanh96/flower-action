from federated_learning import FedAvgVideoClient
from federated_learning.client.update.video_base import VideoLocalUpdate 
from evaluation.video_recognition import evaluate_video_recognizer
import flwr
import torch 
import argparse
from datasets import *
from mmcv import Config
from mmaction.models import build_model
import yaml 
from utils.parsing import Dict2Class