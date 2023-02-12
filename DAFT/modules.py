from abc import ABCMeta, abstractmethod
from collections import OrderedDict

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

class FilmBase(nn.Module, metaclass=ABCMeta):
  """
  Wrapper for models using a Film-Like structure.
  Very similar to ResBlock, but will sandwich in the Film scaling and shifting 
  operation at the location specified.
  """
  # bn momentum left at default
  # scale and shift left out since we always want to scale and shift
  # activation using gelu as default, relu for convlayers
  def __init__(self, in_channels, out_channels, stride, ndim_tabular, location):
    super().__init__()
    if location not in set(range(5)):
      raise ValueError(f"Invalid location parameter, needs to be in {0, 1, 2, 3, 4}")
    # initialize components for residual block
    self.conv1 = conv3d(in_channels, out_channels, stride=stride)
    self.bn1 = nn.BatchNorm3d(out_channels) # affine left at default of True
    self.conv2 = conv3d(out_channels, out_channels)
    self.bn2 = nn.BatchNorm3d(out_channels)
    self.relu = nn.ReLU() # inplace left at default of False
    self.global_pool = nn.AdaptiveAvgPool3d(1) # target output size of 1x1x1
    # need to match dimensionality of input and output tensors for residual connection
    if stride != 1 or in_channels != out_channels:
      self.downsample = nn.Sequential(
        conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
        nn.BatchNorm3d(out_channels) # no momentum
      )
    else:
      self.downsample = None
    # location specific dimensionality of Film output tensor
    self.film_dims = 0
    if location in {0, 1, 2}:
      self.film_dims = in_channels
    else: 
      self.film_dims = out_channels
    # activation function for Film layer
    self.scale_activation = nn.GELU()

  @abstractmethod
  def rescale_features(self, feature_map, x_aux):
    """Scale and shift features"""
    raise NotImplementedError("Subclasses need to provide this method.")

  def forward(self, feature_map, x_aux):
    """
    This is a bit messy and the reason this meta class exists.
    Location defines the point at which we will rescale the image features
    According to the tabular data. 
    The exact procedure for this needs to be defined in `rescale_features`.
    """
    if self.location == 0:
      feature_map = self.rescale_features(feature_map, x_aux)

    residual = feature_map

    if self.location == 1:
      residual = self.rescale_features(residual, x_aux)
    
    if self.location == 2:
      feature_map = self.rescale_features(feature_map, x_aux)

    out = self.conv1(feature_map)
    out = self.bn1(out)

    if self.location == 3:
      out = self.rescale_features(out, x_aux)
    
    out = self.relu(out)

    if self.location == 4:
      out = self.rescale_features(out, x_aux)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(residual)
    
    out += residual
    out = self.relu(out)
    return out
  

class FilmBlock(FilmBase):
  """
  Class that extends FilmBase to include the scaling and shifting operation on 
  the feature maps to account for tabular auxiliary data. 
  In the original paper the auxiliary data had 15 features. 
  """
  def __init__(self, in_channels, out_channels, stride=2, ndim_tabular=15, location=0, bottleneck_dim=7):
    super().__init__(
      in_channels=in_channels,
      out_channels=out_channels,
      stride=stride,
      ndim_tabular=ndim_tabular,
      location=location,
    )
    self.bottleneck_dim = bottleneck_dim
    # we always scale and shift
    self.split_size = self.film_dims
    self.film_dims = 2 * self.film_dims
    # initialize embedding net for tabular data
    embedding_layers = [
      ("embedding_in", nn.Linear(ndim_tabular, self.bottleneck_dim, bias=False)),
      ("embedding_relu", nn.ReLU()),
      ("embedding_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)),
    ]
    self.embedding_net = nn.Sequential(OrderedDict(embedding_layers))

  def rescale_features(self, feature_map, x_aux):
    embedding = self.embedding_net(x_aux)
    assert (embedding.size(0) == feature_map.size(0)) and (embedding.dim() == 2), f"Invalid output size of embedding network: {embedding.size()}"
    scale, shift = torch.split(embedding, self.split_size, dim=1)
    # independently obtain scale and shift as learnable vectors that match feature_map in dimension
    scale = scale.view(*scale.size(), 1, 1, 1).expand_as(feature_map)
    shift = shift.view(*shift.size(), 1, 1, 1).expane_as(feature_map)
    # we always gate scale with the GELU activation
    scale = self.scale_activation(scale)
    # analog to `(gammas * x) + betas` in FiLM
    return (scale * feature_map) + shift

class DAFTBlock(FilmBlock):
  """
  Class that extends `FilmBlock` with a new `rescale_features` method.
  The only change is added global pooling of the respective feature map
  that is performed before concatenating the pooled information to the tabular
  information and passing it through the embedding network
  """
  def __init__(self, in_channels, out_channels, stride=2, ndim_tabular=15, location=0, bottleneck_dim=7):
    super().__init__(
      in_channels=in_channels,
      out_channels=out_channels,
      stride=stride,
      ndim_tabular=ndim_tabular,
      location=location,
      bottleneck_dim=bottleneck_dim,
    )
    # split size and film_dims initialized in superclass
    # extend embedding input to match tabular dim + input from global pooling
    embedding_input_dims = self.film_dims / 2 # TODO: double check if /2 is needed since we init film_dims in superclass
    # initialize embedding net for tabular data
    embedding_layers = [
      ("embedding_in", nn.Linear(ndim_tabular + embedding_input_dims, self.bottleneck_dim, bias=False)),
      ("embedding_relu", nn.ReLU()),
      ("embedding_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)),
    ]
    self.embedding_net = nn.Sequential(OrderedDict(embedding_layers))

  def rescale_features(self, feature_map, x_aux):
    pooled_features = self.global_pool(feature_map)
    pooled_features = pooled_features.view(pooled_features.size(0), -1)
    pooled_features = torch.cat((pooled_features, x_aux), dim=1)

    embedding = self.embedding_net(pooled_features)

    scale, shift = torch.split(embedding, self.split_size, dim=1)
    # scale and shift are learned independently
    scale = scale.view(*scale.size(), 1, 1, 1).expand_as(feature_map)
    shift = shift.view(*shift.size(), 1, 1, 1).expand_as(feature_map)
    # we always gate shift with GELU
    scale = self.scale_activation(scale)
    # analog to `(gammas * x) + betas` in FiLM
    return (scale * feature_map) + shift

