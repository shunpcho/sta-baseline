from .container import Container, open
from .frame import Frame as Frame
from .packet import Packet as Packet
from .streams import Stream as Stream
from .streams import VideoStream as VideoStream

def open(path: str, mode: str = "r") -> Container: ...
