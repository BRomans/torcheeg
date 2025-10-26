from .dgcnn import DGCNN
from .lggnet import LGGNet
from .gin import GIN
try:
    from .rgnn import RGNN
except ImportError:
    pass