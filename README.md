# 🧠 Brain Tumor Detection using Deep Learning

A **web-based brain tumor detection system** built with **TensorFlow, FastAPI, and HTML/CSS**.  
This project allows users to upload MRI scans, processes them through a **pretrained CNN model**, and predicts one of four classes:  

- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

---

## 🚀 Features
✅ Upload MRI scans via frontend  
✅ Backend prediction with **FastAPI**  
✅ Pretrained CNN model for classification  
✅ 4-class classification with ~91% test accuracy  
✅ Simple **UI view** for user interaction  

---

## 📸Preview
<img width="1873" height="886" alt="Screenshot 2025-06-27 170351" src="https://github.com/user-attachments/assets/b708d5e3-2445-4abd-a283-bba8c7cbccb8" />
<img width="1873" height="888" alt="Screenshot 2025-06-27 164405" src="https://github.com/user-attachments/assets/b6482920-cbab-431e-92ee-e06c16312812" />
<img width="1894" height="894" alt="Screenshot 2025-06-27 164312" src="https://github.com/user-attachments/assets/78aaa213-d53f-496c-a1bd-c089b266a655" />


## ⚙️ Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
2️⃣ Create & activate virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
3️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Run the backend
bash
Copy
Edit
uvicorn backend.main:app --reload
