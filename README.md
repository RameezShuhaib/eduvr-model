# eduvr-model
A Project on Deep Learning for analysing confidence level in an audio.

## Usage
```python
from eduvr.eduvrmodel import Model
model = Model()
model.setModel('saved_model.HDF5')
model.setAudio('example.m4a')
model.getLevel()
```
