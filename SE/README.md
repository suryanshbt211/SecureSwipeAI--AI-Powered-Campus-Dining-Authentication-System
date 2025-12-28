# SecureSwipe AI - Real-Time Facial Recognition Framework for Contactless Campus Dining

A full-stack web application for secure, contactless meal swipe authentication using YOLO-based facial recognition.

## Features

- **Real-Time Facial Recognition Login**: Secure authentication using live camera feed
- **User Dashboard**: View meal swipe balance, transaction history, and manage account
- **New Registration**: Register with university ID and multi-angle facial images (default 8 meal swipes)
- **Admin Panel**: Comprehensive monitoring and management portal
- **Data Management**: Uses Pandas DataFrames for efficient data handling (no database required)

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Face Recognition**: YOLOv8 + DeepFace (Facenet model)
- **Data Storage**: Pandas DataFrames (CSV files)
- **Libraries**: OpenCV, Ultralytics, DeepFace, HuggingFace Hub

## Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Ensure you have the required model**:
   - The YOLOv8 face detection model will be automatically downloaded from HuggingFace on first run
   - This may take a few minutes on first startup

## Running the Application

1. **Start the Flask server**:
```bash
python app.py
```

2. **Access the application**:
   - Open your browser and go to: `http://localhost:5000`
   - The application will redirect to the login page

## Application Structure

```
.
├── app.py                 # Flask backend server
├── se.py                  # YOLO face recognition model
├── requirements.txt       # Python dependencies
├── templates/            # HTML templates
│   ├── login.html        # Login page with face recognition
│   ├── dashboard.html    # User dashboard
│   ├── register.html     # Registration page
│   └── admin.html        # Admin panel
├── data/                 # Data storage (created automatically)
│   ├── users.csv         # User data
│   ├── transactions.csv  # Transaction records
│   └── faces.pkl         # Face embeddings database
└── uploads/              # Temporary upload directory
```

## Usage Guide

### For Users

1. **Registration**:
   - Go to `/register`
   - Fill in University ID, Name, and Email
   - Capture at least 3 face images from different angles
   - You'll receive 8 meal swipes automatically

2. **Login**:
   - Go to `/login`
   - Allow camera access
   - Click "Recognize Face" to authenticate

3. **Dashboard**:
   - View your remaining balance
   - See transaction history
   - Use meal swipes

### For Admins

1. **Access Admin Panel**:
   - Login with an admin account (default admin user is created)
   - Click "Admin" button on dashboard or go to `/admin`

2. **Admin Features**:
   - View all users and their balances
   - Monitor all transactions
   - View system statistics
   - Filter transactions by user

## Default Admin Account

A default admin user is automatically created:
- **University ID**: ADMIN001
- **Role**: admin
- **Note**: You'll need to register this admin account with facial recognition through the registration page if you want to use it

## API Endpoints

### Authentication
- `POST /api/recognize` - Face recognition login
- `POST /api/logout` - Logout

### Registration
- `POST /api/register` - Register new user

### User
- `GET /api/user/profile` - Get user profile and transactions

### Transactions
- `POST /api/transaction/swipe` - Process meal swipe

### Admin
- `GET /api/admin/users` - Get all users
- `GET /api/admin/transactions` - Get all transactions
- `GET /api/admin/stats` - Get system statistics

## Data Storage

All data is stored using Pandas DataFrames and saved to CSV files:
- **Users**: `data/users.csv`
- **Transactions**: `data/transactions.csv`
- **Face Embeddings**: `data/faces.pkl` (binary pickle file)

The data directory is created automatically on first run.

## Security Notes

- Face recognition uses cosine similarity threshold of 0.5 (50%)
- Sessions are used for authentication
- All face images are processed and stored as embeddings only (not raw images)
- Make sure to set a strong `SECRET_KEY` in production

## Troubleshooting

1. **Camera not working**:
   - Ensure camera permissions are granted in browser
   - Try a different browser (Chrome recommended)
   - Check if camera is being used by another application

2. **Face not recognized**:
   - Ensure good lighting
   - Look directly at camera
   - Make sure face is clearly visible
   - Try re-registering with better quality images

3. **Model download issues**:
   - Check internet connection (required for first-time model download)
   - Model is cached after first download

## License

This project is for educational and research purposes.

## Support

For issues or questions, please check the code comments or review the API endpoints documentation above.

