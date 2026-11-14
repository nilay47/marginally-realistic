"""CVPR27 extension: joint physical consistency in video+sensor generation.

Importing this package sets the training-config defaults for physvid BEFORE
physvid is imported (physvid reads them at import time). Override by exporting
PV_T / PV_DT / PV_WIN yourself. Always import physvid as `from cvpr27 import physvid`.
"""
import os

os.environ.setdefault("PV_T", "64")
os.environ.setdefault("PV_DT", "0.04")
os.environ.setdefault("PV_WIN", "8")
