import numpy as np
from dataclasses import dataclass

@dataclass
class Measurement:
    """Class for descirbing the camera measurement of the x-axis position of the center of the QR code and its height.
    
    Attributes:
        timestamp: The timestamp of the measurement.
        cx: The x-axis position of the center of the QR code.
        height: The height of the QR code.
    """
    timestamp: float
    cx: float
    height: float

class QrCode:
    """Class for describing the QR code initialized with the id and global poistion. 
    It computes the mean center and height of the code as seen by the camera given measurements.

    Attributes:
        id: The id of the QR code.
        sx: The x global coordinate of the QR code.
        sy: The y global coordinate of the QR code.
        cx: The x coordinate of the center of the QR code.
        height: The height of the QR code.
    """

    def __init__(self, id: int, sx: float, sy: float):
        """
        Constructor for the QrCode class.

        Args:
            id: The id of the QR code.
            sx: The x global coordinate of the QR code.
            sy: The y global coordinate of the QR code.
            cx: The x coordinate of the center of the QR code.
            measurements: list of measurements from the camera module.
        """
        self.id = id
        self.sx = sx
        self.sy = sy
        self.measurements = []

    def update_measurements(self, measurements: list[Measurement]):
        """Updates the QR code with new measurements.

        Args:
            measurements: list of measurements from the camera module.
        """
        self.measurements = measurements

    def add_measurement(self, measurement: Measurement):
        """Adds a new measurement to the QR code.

        Args:
            measurement: A new measurement from the camera module.
        """
        self.measurements.append(measurement)
    