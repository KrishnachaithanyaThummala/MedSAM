# MedNet exact code map
Source commit: 6adb072630811b1d39568a87732aed96dc39c9ae
Original source: https://github.com/Md-Ferdous/MedNet/blob/6adb072630811b1d39568a87732aed96dc39c9ae/MedNet.py

- `AlbumentationsTransform`: lines 20-26
  - `__init__`: lines 21-22
  - `__call__`: lines 23-26
- `FocalLoss`: lines 31-43
  - `__init__`: lines 32-36
  - `forward`: lines 37-43
- `ChannelAttention`: lines 48-62
  - `__init__`: lines 49-56
  - `forward`: lines 57-62
- `SpatialAttention`: lines 64-73
  - `__init__`: lines 65-68
  - `forward`: lines 69-73
- `CBAM`: lines 75-81
  - `__init__`: lines 76-79
  - `forward`: lines 80-81
- `DepthwiseSeparableConv`: lines 86-94
  - `__init__`: lines 87-92
  - `forward`: lines 93-94
- `ResidualDSCBAMBlock`: lines 96-111
  - `__init__`: lines 97-109
  - `forward`: lines 110-111
- `ResidualCBAMBackbone`: lines 116-129
  - `__init__`: lines 117-123
  - `forward`: lines 124-129
- `Darma`: lines 131-144
  - `__init__`: lines 132-138
  - `forward`: lines 139-144
- `EarlyStopping`: lines 149-165
  - `__init__`: lines 150-155
  - `__call__`: lines 156-165
- `set_seed`: lines 170-177
- `train_model`: lines 207-287
- `test_model`: lines 292-314
- `main`: lines 319-397
