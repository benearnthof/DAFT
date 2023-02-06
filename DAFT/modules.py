import torch
import torch.nn as nn
import torch.nn.parallel 
import torch.utils.data

def conv3d(in_channels, out_channels, kernel_size=3, stride=1):
  """
  A thin wrapper around conv3d to streamline padding
  """
  if kernel_size != 1:
    padding = 1
  else:
    padding = 0
  return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)


class ConvBlock(nn.Module):
  """
  Wrap up convolution, batch norm, & activation in one block.
  This is only used once to setup the base model used for benchmarking, the 
  Heterogenous Resnet.
  """
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    super().__init__()
    self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    self.bn = nn.BatchNorm3d(out_channels)
    self.relu = nn.ReLU() # inplace=True may save memory

  def forward(self, x):
    out = self.conv(x)
    out = self.bn(out)
    out = self.relu(out)
    return out

class ResBlock(nn.Module):
  """
  Residual Block used in all models.
  Performs two convolution operations, each of which are followed by BatchNorm
  and uses relu as nonlinearity.
  To ensure that the dimensions of the residual matches the output tensor 
  a downsampling step is performed before adding the residual to the output.
  """
  def __init__(self, in_channels, out_channels, stride=1): # batch norm momentum left at default
    super().__init__()
    self.conv1 = conv3d(in_channels, out_channels, stride=stride)
    self.bn1 = nn.BatchNorm3d(out_channels)
    self.conv2 = conv3d(out_channels, out_channels)
    self.bn2 = nn.BatchNorm3d(out_channels)
    self.relu = nn.ReLU()
    # need to match dimensionality of input and output tensors for residual connection
    if stride != 1 or in_channels != out_channels:
      self.downsample = nn.Sequential(
        conv3d(in_channels, out_channels, kernel_size=1, stride = stride),
        nn.BatchNorm3d(out_channels)
      )
    else:
      self.downsample = None

  def forward(self, x):
    if self.downsample is not None: 
      residual = self.downsample(x)
    else:
      residual = x
    
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    out += residual
    out = self.relu(out)
    return out

class FiLM(nn.Module):
  """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  From https://arxiv.org/pdf/1709.07871.pdf (Perez et al.)
  """
  # In the original FiLM paper, they use a chain of GRUs to extract information from 
  # Text and use a FiLM layer to condition the output of the CNN that extracts 
  # information from images on the information contained in the text
  # we simply learn two scale and shift parameters gamma & beta to accomplish this
  # DAFT uses a very similar idea to condition the 3D-Convnet on the supplementary
  # tabular patient data
  def forward(self, x, gammas, betas):
    gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
    betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
    return (gammas * x) + betas

# DAFT: Dynamic Affine Featuremap Transform:
# The core Idea is taken from FiLM; we learn pretty much the exact same operation
# gammas is the v_scale parameter to scale the incoming feature map
# betas is the v_shift parameter to shift the incoming feature map
# DAFT uses a third `location` parameter in their code to specify the 
# location in the network at which this operation is being done. 

