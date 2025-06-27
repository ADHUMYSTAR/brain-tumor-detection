# Brain Tumor Detection using Deep Learning

A web-based brain tumor detection system using TensorFlow, FastAPI, and HTML/CSS.

## Features
- Upload MRI scans via frontend
- Backend prediction with pretrained CNN model
- 4-class classification (e.g. glioma, meningioma, pituitary, no tumor)
- Accuracy: ~91% on test set

## Run locally
```bash
uvicorn backend.main:app --reload
