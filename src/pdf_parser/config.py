import os

MODEL = os.getenv("LAYOUTPARSER_MODEL", "mask_rcnn_X_101_32x8d_FPN_3x")
MODEL_THRESHOLD_RESTRICTIVE = float(os.getenv("MODEL_THRESHOLD_RESTRICTIVE", "0.5"))
OCR_AGENT = os.getenv("OCR_AGENT", "gcv")
