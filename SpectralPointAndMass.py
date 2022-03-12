from dataclasses import dataclass

# Define Helper Classes for Spectral Stuff

@dataclass
class SpectralPoint:
    value: complex = complex(0)
    # time: float = 0.0
    freq: float = 0.0 
    # time_reassigned: float = 0.0
    freq_reassigned: float = 0.0

@dataclass
class SpectralMass:
	left_bin: int = 0
	right_bin: int = 0
	center_bin: int = 0
	mass: float = 0