# Quick Start Guide - SecureSwipe AI

## 🚀 Running the Application (3 Simple Steps)

### Step 1: Install Dependencies

Open a terminal/PowerShell in the project directory and run:

```bash
pip install -r requirements.txt
```

**Note:** This will install:
- Flask (web framework)
- OpenCV (image processing)
- YOLO/Ultralytics (face detection)
- DeepFace (face recognition)
- Pandas (data management)
- And other dependencies

⏱️ *This may take 2-5 minutes, especially for TensorFlow and model downloads*

### Step 2: Start the Server

**Option A - Using the batch file (Windows):**
```bash
START.bat
```

**Option B - Using Python directly:**
```bash
python app.py
```

**Option C - Using the run script:**
```bash
python run.py
```

### Step 3: Open in Browser

Once the server starts, you'll see:
```
 * Running on http://0.0.0.0:5000
```

Open your web browser and go to:
```
http://localhost:5000
```

The application will automatically redirect you to the login page.

---

## 📋 First Time Setup

### First Run Notes:

1. **Model Download**: On first run, the YOLO face detection model will be downloaded from HuggingFace. This happens automatically and may take 1-2 minutes. You'll see:
   ```
   Loading YOLOv8 Face Detection model...
   ```

2. **Data Directories**: The following directories will be created automatically:
   - `data/` - Stores users.csv, transactions.csv, faces.pkl
   - `uploads/` - Temporary file storage

3. **Default Admin**: A default admin user is created in the system (University ID: ADMIN001)

---

## 🔧 Troubleshooting

### Issue: "Module not found" errors
**Solution:** Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: "Port 5000 already in use"
**Solution:** Either:
- Close the application using port 5000, OR
- Edit `app.py` line 478 and change `port=5000` to a different port (e.g., `port=5001`)

### Issue: Camera not working
**Solution:**
- Allow browser camera permissions
- Try Chrome or Edge browser
- Make sure no other app is using the camera

### Issue: Face recognition not working
**Solution:**
- Ensure good lighting
- Look directly at the camera
- Make sure your face is clearly visible
- Re-register if needed

---

## 📱 Using the Application

1. **New User Registration:**
   - Go to `/register` or click "New Registration" from login
   - Fill in University ID, Name, Email
   - Capture at least 3 face images (different angles)
   - You'll automatically get 8 meal swipes

2. **Login:**
   - Allow camera access
   - Click "Recognize Face"
   - You'll be redirected to dashboard

3. **Dashboard:**
   - View balance and transactions
   - Use meal swipes
   - Check transaction history

4. **Admin Panel:**
   - Login as admin user
   - Click "Admin" button or go to `/admin`
   - View all users, transactions, and statistics

---

## 🛑 Stopping the Server

Press `CTRL+C` in the terminal where the server is running.

---

## ✅ Verification Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Server starts without errors
- [ ] Browser opens `http://localhost:5000`
- [ ] Login page loads
- [ ] Camera access works

---

**Need Help?** Check the main `README.md` for detailed documentation.

