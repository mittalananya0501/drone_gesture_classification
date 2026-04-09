This project demonstrates hand gesture recognition using a live camera feed to control drone-like commands such as UP, DOWN, LEFT, RIGHT, TAKEOFF, LAND, FLIP.

Setup:

```bash
conda create -n gesture-env python=3.10
conda activate gesture-env
pip install -r requirements.txt
```

Run:

```bash
python mediaPipeAndCNN/dataCollect.py
python mediaPipeAndCNN/train_model.py
python mediaPipeAndCNN/gesturePredict.py
```
