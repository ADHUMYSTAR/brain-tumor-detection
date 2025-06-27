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

![image](https://github.com/user-attachments/assets/ea9fcab1-314d-4d78-8943-49e802c4b758)
