from dataclasses import dataclass
from typing import Optional

__all__ = ["ColorScheme", "LIGHT_THEME", "DARK_THEME"]


# Color theme dataclass
@dataclass
class ColorScheme:
    '''Dataclass to store color theme for computation graph'''
    TensorNode: str = "lightyellow"
    ModuleNode: str = "darkseagreen1"
    FunctionNode: str = "aliceblue"

    activation    : str = "indianred1"
    adaptive      : str = ""
    batchnorm     : str = ""
    channelshuffle: str = ""
    container     : str = ""
    conv          : str = "deepskyblue1"
    distance      : str = ""
    dropout       : str = ""
    flatten       : str = ""
    fold          : str = ""
    instancenorm  : str = ""
    lazy          : str = ""
    linear        : str = "deepskyblue1"
    normalization : str = ""  #"darkgoldenrod1"
    padding       : str = ""
    pixelshuffle  : str = ""
    pooling       : str = ""
    rnn           : str = ""
    sparse        : str = ""
    transformer   : str = ""
    upsampling    : str = ""

    def __getitem__(self, node_type: str) -> str:
        '''Returns color for given node type

        Args:
            node_type (str):
                Type of node

        Returns:
            str: Color for given node type
        '''
        return self.__dict__[node_type]

    def get(self, node_type: str, fallback: Optional[str] = None) -> str:
        '''Returns color for given node type

        Args:
            node_type (str):
                Type of node
            fallback (str, optional):
                Fallback color if node_type is not found. Defaults to None.

        Returns:
            str: Color for given node type
        '''
        return self.__dict__.get(node_type, fallback)


LIGHT_THEME = ColorScheme()
DARK_THEME = ColorScheme(
    TensorNode     = "darkgoldenrod4",
    ModuleNode     = "darkseagreen4",
    FunctionNode   = "cadetblue4",

    activation     = "firebrick",
    adaptive       = "",
    batchnorm      = "",
    channelshuffle = "",
    container      = "",
    conv           = "dodgerblue4",
    distance       = "",
    dropout        = "darkolivegreen4",
    flatten        = "",
    fold           = "",
    instancenorm   = "",
    lazy           = "",
    linear         = "dodgerblue4",
    normalization  = "",  # "darkgoldenrod3",
    padding        = "",
    pixelshuffle   = "",
    pooling        = "",
    rnn            = "",
    sparse         = "palegreen4",
    transformer    = "",
    upsampling     = "",
)
