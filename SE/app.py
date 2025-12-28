from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
import cv2
import base64
import os
import pickle
from datetime import datetime
from werkzeug.utils import secure_filename
import json
from se import FaceRegistrationSystem
import uuid

app = Flask(__name__)
app.secret_key = 'secure_swipe_ai_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app)

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)

# Initialize face recognition system
face_system = FaceRegistrationSystem()

# ============================================================================
# DATA MANAGEMENT - DataFrame Structure
# ============================================================================
# 
# The system uses a two-part storage approach:
# 1. USER METADATA (DataFrame/CSV): Stores Name, ID, Email, Meal Swipes Count
#    - Stored in: data/users.csv
#    - Columns: user_id, university_id, name, email, balance (meal_swipes), 
#               registered_faces, role, created_at
#
# 2. FACE EMBEDDINGS (Pickle file): Stores face recognition embeddings
#    - Stored in: data/faces.pkl
#    - Format: {user_id: [list of numpy embedding arrays]}
#    - Why separate? Embeddings are 128-dimensional numpy arrays that don't
#      serialize well in CSV. This separation is best practice.
#
# NOTE: Raw images are NOT stored - only their mathematical representations
#       (embeddings) are kept for face recognition.
# ============================================================================

USERS_FILE = 'data/users.csv'
TRANSACTIONS_FILE = 'data/transactions.csv'
FACES_FILE = 'data/faces.pkl'

# Default meal swipes for new users
DEFAULT_MEAL_SWIPES = 8

def init_dataframes():
    """
    Initialize or load DataFrames for user and transaction data.
    
    Returns:
        tuple: (users_df, transactions_df) - Two pandas DataFrames
    """
    # Users DataFrame - Stores user metadata
    if os.path.exists(USERS_FILE):
        users_df = pd.read_csv(USERS_FILE)
        # Ensure data types are correct
        users_df['balance'] = users_df['balance'].astype(float)
        users_df['registered_faces'] = users_df['registered_faces'].astype(int)
    else:
        # Create new DataFrame with proper structure
        users_df = pd.DataFrame(columns=[
            'user_id',           # Unique identifier (e.g., 'user-abc123')
            'university_id',     # Student/Staff ID (e.g., '900428564')
            'name',              # Full name
            'email',             # Email address
            'balance',           # Meal swipes count (default: 8)
            'registered_faces',  # Number of face embeddings registered
            'role',              # 'user' or 'admin'
            'created_at'         # Registration timestamp
        ])
        # Create default admin user
        admin_data = {
            'user_id': 'admin-001',
            'university_id': 'ADMIN001',
            'name': 'Admin User',
            'email': 'admin@university.edu',
            'balance': 0,
            'registered_faces': 0,
            'role': 'admin',
            'created_at': datetime.now().isoformat()
        }
        users_df = pd.concat([users_df, pd.DataFrame([admin_data])], ignore_index=True)
        users_df.to_csv(USERS_FILE, index=False)
        print(f"✓ Created new users DataFrame with {len(users_df)} user(s)")
    
    # Transactions DataFrame - Stores transaction history
    if os.path.exists(TRANSACTIONS_FILE):
        transactions_df = pd.read_csv(TRANSACTIONS_FILE)
    else:
        transactions_df = pd.DataFrame(columns=[
            'transaction_id',    # Unique transaction ID
            'user_id',           # Links to users_df
            'university_id',     # For quick lookup
            'transaction_type',  # 'meal_swipe', 'initial_balance', 'admin_added', etc.
            'amount',            # Change in balance (+ or -)
            'balance_after',     # Balance after transaction
            'timestamp',         # When transaction occurred
            'status'             # 'completed', 'failed', etc.
        ])
        transactions_df.to_csv(TRANSACTIONS_FILE, index=False)
        print(f"✓ Created new transactions DataFrame")
    
    return users_df, transactions_df

def save_dataframes(users_df, transactions_df):
    """
    Save DataFrames to CSV files.
    
    Args:
        users_df: Users DataFrame
        transactions_df: Transactions DataFrame
    """
    users_df.to_csv(USERS_FILE, index=False)
    transactions_df.to_csv(TRANSACTIONS_FILE, index=False)

def init_face_embeddings():
    """
    Initialize the face embeddings storage file if it doesn't exist.
    
    Returns:
        dict: Dictionary mapping user_id to list of embeddings
    """
    if not os.path.exists(FACES_FILE):
        faces_data = {}
        with open(FACES_FILE, 'wb') as f:
            pickle.dump(faces_data, f)
        print(f"✓ Created new face embeddings storage file")
        return faces_data
    
    # Load existing embeddings
    with open(FACES_FILE, 'rb') as f:
        faces_data = pickle.load(f)
    return faces_data

def save_face_embeddings(faces_data):
    """
    Save face embeddings to pickle file.
    
    Args:
        faces_data: Dictionary mapping user_id to list of numpy embedding arrays
    """
    with open(FACES_FILE, 'wb') as f:
        pickle.dump(faces_data, f)

def get_user_by_id(user_id):
    """
    Get user information from DataFrame by user_id.
    
    Args:
        user_id: The user's unique identifier
        
    Returns:
        dict: User information or None if not found
    """
    global users_df
    user = users_df[users_df['user_id'] == user_id]
    if not user.empty:
        return user.iloc[0].to_dict()
    return None

def get_user_by_university_id(university_id):
    """
    Get user information from DataFrame by university_id.
    
    Args:
        university_id: The user's university ID
        
    Returns:
        dict: User information or None if not found
    """
    global users_df
    user = users_df[users_df['university_id'] == university_id]
    if not user.empty:
        return user.iloc[0].to_dict()
    return None

def update_user_balance(user_id, new_balance):
    """
    Update a user's meal swipe balance in the DataFrame.
    
    Args:
        user_id: The user's unique identifier
        new_balance: New balance value (must be >= 0)
        
    Returns:
        bool: True if successful, False if user not found
    """
    global users_df
    user_idx = users_df[users_df['user_id'] == user_id].index
    if user_idx.empty:
        return False
    
    users_df.loc[user_idx[0], 'balance'] = max(0, float(new_balance))
    return True

# Load initial data
users_df, transactions_df = init_dataframes()
faces_data = init_face_embeddings()  # Initialize face embeddings storage

def decode_base64_image(base64_string):
    """Decode base64 image string to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def get_face_embedding_from_image(image):
    """Extract face embedding from image using DeepFace"""
    from deepface import DeepFace
    try:
        # Save temporary image
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(temp_path, image)
        
        embedding_objs = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            enforce_detection=False
        )
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if len(embedding_objs) > 0:
            return np.array(embedding_objs[0]["embedding"])
        return None
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None

def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def recognize_user_from_image(image):
    """Recognize user from image by comparing with registered faces"""
    global users_df
    
    # Get embedding from input image
    embedding = get_face_embedding_from_image(image)
    if embedding is None:
        return None, 0.0
    
    # Load face embeddings database
    faces_data = init_face_embeddings()
    if not faces_data:
        return None, 0.0
    
    # Compare with all registered faces
    max_similarity = -1
    best_match_user_id = None
    
    for user_id, embeddings_list in faces_data.items():
        for stored_embedding in embeddings_list:
            similarity = cosine_similarity(embedding, stored_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_user_id = user_id
    
    # Threshold for matching (0.5 = 50% similarity)
    if max_similarity > 0.5:
        return best_match_user_id, max_similarity
    
    return None, max_similarity

@app.route('/')
def index():
    """Home page - redirects to login"""
    return redirect(url_for('login'))

@app.route('/login')
def login():
    """Login page"""
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/register')
def register():
    """Registration page"""
    return render_template('register.html')

@app.route('/admin')
def admin():
    """Admin login page"""
    # Check if admin is already logged in
    if 'admin_logged_in' in session and session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard'))
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    """Admin dashboard page"""
    # Check if admin is logged in
    if 'admin_logged_in' not in session or not session.get('admin_logged_in'):
        return redirect(url_for('admin'))
    return render_template('admin_dashboard.html')

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """API endpoint for face recognition login"""
    global users_df
    
    try:
        data = request.json
        base64_image = data.get('image')
        
        if not base64_image:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        # Decode image
        image = decode_base64_image(base64_image)
        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image format'}), 400
        
        # Detect face first using YOLO
        results = face_system.yolo_model(image, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            return jsonify({'success': False, 'error': 'No face detected'}), 400
        
        # Get first detected face bounding box
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        face_region = image[y1:y2, x1:x2]
        
        # Recognize user
        user_id, confidence = recognize_user_from_image(face_region)
        
        if user_id:
            # Get user info
            user = users_df[users_df['user_id'] == user_id].iloc[0].to_dict()
            
            # Set session
            session['user_id'] = user_id
            session['university_id'] = user['university_id']
            session['name'] = user['name']
            session['role'] = user.get('role', 'user')
            
            return jsonify({
                'success': True,
                'user_id': user_id,
                'name': user['name'],
                'confidence': float(confidence * 100)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Face not recognized. Please register first.',
                'confidence': float(confidence * 100)
            }), 401
    
    except Exception as e:
        print(f"Recognition error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/register', methods=['POST'])
def api_register():
    """API endpoint for user registration"""
    global users_df, transactions_df
    
    try:
        data = request.json
        university_id = data.get('university_id', '').strip()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        images = data.get('images', [])  # Array of base64 images
        
        # Validation
        if not university_id or not name or not email:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        if len(images) < 3:
            return jsonify({'success': False, 'error': 'Please upload at least 3 images from different angles'}), 400
        
        # Check if university_id already exists
        if not users_df[users_df['university_id'] == university_id].empty:
            return jsonify({'success': False, 'error': 'University ID already registered'}), 400
        
        # Process images and extract embeddings
        embeddings_list = []
        for base64_img in images:
            image = decode_base64_image(base64_img)
            if image is None:
                continue
            
            # Detect face using YOLO
            results = face_system.yolo_model(image, verbose=False)
            if len(results) == 0 or len(results[0].boxes) == 0:
                continue
            
            # Get first detected face
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            face_region = image[y1:y2, x1:x2]
            
            # Extract embedding
            embedding = get_face_embedding_from_image(face_region)
            if embedding is not None:
                embeddings_list.append(embedding)
        
        if len(embeddings_list) < 3:
            return jsonify({'success': False, 'error': 'Could not extract enough valid face embeddings. Please ensure faces are clearly visible.'}), 400
        
        # Create user entry in DataFrame
        user_id = f"user-{uuid.uuid4().hex[:8]}"
        new_user = {
            'user_id': user_id,
            'university_id': university_id,
            'name': name,
            'email': email,
            'balance': DEFAULT_MEAL_SWIPES,  # Default meal swipes
            'registered_faces': len(embeddings_list),
            'role': 'user',
            'created_at': datetime.now().isoformat()
        }
        
        # Add user to DataFrame
        users_df = pd.concat([users_df, pd.DataFrame([new_user])], ignore_index=True)
        
        # Save face embeddings separately (not in DataFrame - they're numpy arrays)
        global faces_data
        faces_data = init_face_embeddings()  # Load existing embeddings
        faces_data[user_id] = embeddings_list  # Store embeddings for this user
        save_face_embeddings(faces_data)  # Save to pickle file
        
        # Create initial balance transaction
        transaction = {
            'transaction_id': f"txn-{uuid.uuid4().hex[:8]}",
            'user_id': user_id,
            'university_id': university_id,
            'transaction_type': 'initial_balance',
            'amount': DEFAULT_MEAL_SWIPES,
            'balance_after': DEFAULT_MEAL_SWIPES,
            'timestamp': datetime.now().isoformat(),
            'status': 'completed'
        }
        transactions_df = pd.concat([transactions_df, pd.DataFrame([transaction])], ignore_index=True)
        
        # Save data
        save_dataframes(users_df, transactions_df)
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'message': f'Registration successful! You have been credited with {DEFAULT_MEAL_SWIPES} meal swipes.'
        })
    
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user/profile', methods=['GET'])
def api_user_profile():
    """Get current user profile"""
    global users_df, transactions_df
    
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    user = users_df[users_df['user_id'] == user_id].iloc[0].to_dict()
    
    # Get recent transactions
    user_transactions = transactions_df[transactions_df['user_id'] == user_id].copy()
    user_transactions = user_transactions.sort_values('timestamp', ascending=False).head(10)
    
    return jsonify({
        'success': True,
        'user': {
            'user_id': user['user_id'],
            'university_id': user['university_id'],
            'name': user['name'],
            'email': user['email'],
            'balance': float(user['balance']),
        },
        'transactions': user_transactions.to_dict('records')
    })

@app.route('/api/transaction/swipe', methods=['POST'])
def api_swipe_meal():
    """Process meal swipe transaction"""
    global users_df, transactions_df
    
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    user_idx = users_df[users_df['user_id'] == user_id].index[0]
    
    # Check balance
    if users_df.loc[user_idx, 'balance'] <= 0:
        return jsonify({'success': False, 'error': 'Insufficient balance'}), 400
    
    # Deduct one meal swipe
    current_balance = users_df.loc[user_idx, 'balance']
    new_balance = max(0, current_balance - 1)  # Ensure balance doesn't go negative
    users_df.loc[user_idx, 'balance'] = new_balance
    
    # Create transaction record
    transaction = {
        'transaction_id': f"txn-{uuid.uuid4().hex[:8]}",
        'user_id': user_id,
        'university_id': session['university_id'],
        'transaction_type': 'meal_swipe',
        'amount': -1,
        'balance_after': new_balance,
        'timestamp': datetime.now().isoformat(),
        'status': 'completed'
    }
    transactions_df = pd.concat([transactions_df, pd.DataFrame([transaction])], ignore_index=True)
    
    # Save data
    save_dataframes(users_df, transactions_df)
    
    return jsonify({
        'success': True,
        'balance': float(new_balance),
        'transaction_id': transaction['transaction_id']
    })

@app.route('/api/admin/users', methods=['GET'])
def api_admin_users():
    """Get all users (admin only)"""
    global users_df
    
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    users = users_df.to_dict('records')
    return jsonify({'success': True, 'users': users})

@app.route('/api/admin/transactions', methods=['GET'])
def api_admin_transactions():
    """Get all transactions (admin only)"""
    global transactions_df
    
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    # Get filter parameters
    user_id = request.args.get('user_id')
    limit = int(request.args.get('limit', 100))
    
    filtered_df = transactions_df.copy()
    if user_id:
        filtered_df = filtered_df[filtered_df['user_id'] == user_id]
    
    filtered_df = filtered_df.sort_values('timestamp', ascending=False).head(limit)
    
    return jsonify({
        'success': True,
        'transactions': filtered_df.to_dict('records')
    })

@app.route('/api/admin/stats', methods=['GET'])
def api_admin_stats():
    """Get system statistics (admin only)"""
    global users_df, transactions_df
    
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    stats = {
        'total_users': len(users_df),
        'total_balance': float(users_df['balance'].sum()),
        'total_transactions': len(transactions_df),
        'meal_swipes_today': len(transactions_df[
            (transactions_df['transaction_type'] == 'meal_swipe') &
            (transactions_df['timestamp'].str.startswith(datetime.now().strftime('%Y-%m-%d')))
        ]),
        'active_users': len(users_df[users_df['balance'] > 0])
    }
    
    return jsonify({'success': True, 'stats': stats})

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """Logout user"""
    session.clear()
    return jsonify({'success': True})

@app.route('/api/admin/logout', methods=['POST'])
def api_admin_logout():
    """Admin logout endpoint"""
    session.pop('admin_logged_in', None)
    session.clear()
    return jsonify({'success': True})

@app.route('/api/admin/login', methods=['POST'])
def api_admin_login():
    """Admin login endpoint"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        # Check credentials
        if username == 'adminxyz' and password == 'admin123':
            session['admin_logged_in'] = True
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'error': 'Invalid username or password'}), 401
    except Exception as e:
        print(f"Admin login error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/search-user', methods=['GET'])
def api_admin_search_user():
    """Search user by name or university ID"""
    global users_df
    
    if 'admin_logged_in' not in session or not session.get('admin_logged_in'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        query = request.args.get('q', '').strip().lower()
        if not query:
            return jsonify({'success': False, 'error': 'Please provide a search query'}), 400
        
        # Search by name or university_id
        matching_users = users_df[
            (users_df['name'].str.lower().str.contains(query, na=False)) |
            (users_df['university_id'].str.lower().str.contains(query, na=False))
        ]
        
        if matching_users.empty:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        # Return first match
        user = matching_users.iloc[0].to_dict()
        return jsonify({
            'success': True,
            'user': {
                'user_id': user['user_id'],
                'university_id': user['university_id'],
                'name': user['name'],
                'email': user['email'],
                'balance': float(user['balance'])
            }
        })
    except Exception as e:
        print(f"Search user error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/add-user', methods=['POST'])
def api_admin_add_user():
    """Add a new user"""
    global users_df, transactions_df
    
    if 'admin_logged_in' not in session or not session.get('admin_logged_in'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        data = request.json
        university_id = data.get('university_id', '').strip()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        balance = int(data.get('balance', 8))
        
        # Validation
        if not university_id or not name or not email:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Check if university_id already exists
        if not users_df[users_df['university_id'] == university_id].empty:
            return jsonify({'success': False, 'error': 'University ID already exists'}), 400
        
        # Create user
        user_id = f"user-{uuid.uuid4().hex[:8]}"
        new_user = {
            'user_id': user_id,
            'university_id': university_id,
            'name': name,
            'email': email,
            'balance': balance,
            'registered_faces': 0,
            'role': 'user',
            'created_at': datetime.now().isoformat()
        }
        
        users_df = pd.concat([users_df, pd.DataFrame([new_user])], ignore_index=True)
        
        # Create initial balance transaction
        if balance > 0:
            transaction = {
                'transaction_id': f"txn-{uuid.uuid4().hex[:8]}",
                'user_id': user_id,
                'university_id': university_id,
                'transaction_type': 'admin_added',
                'amount': balance,
                'balance_after': balance,
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            transactions_df = pd.concat([transactions_df, pd.DataFrame([transaction])], ignore_index=True)
        
        # Save data
        save_dataframes(users_df, transactions_df)
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'message': 'User added successfully'
        })
    except Exception as e:
        print(f"Add user error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/delete-user', methods=['POST'])
def api_admin_delete_user():
    """Delete a user"""
    global users_df, transactions_df
    
    if 'admin_logged_in' not in session or not session.get('admin_logged_in'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        data = request.json
        user_id = data.get('user_id', '').strip()
        
        if not user_id:
            return jsonify({'success': False, 'error': 'User ID is required'}), 400
        
        # Check if user exists
        user_idx = users_df[users_df['user_id'] == user_id].index
        if user_idx.empty:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        # Prevent deleting admin users
        user_role = users_df.loc[user_idx[0], 'role']
        if user_role == 'admin':
            return jsonify({'success': False, 'error': 'Cannot delete admin users'}), 400
        
        # Delete user
        users_df = users_df.drop(user_idx)
        
        # Delete associated transactions (optional - you might want to keep them for records)
        # transactions_df = transactions_df[transactions_df['user_id'] != user_id]
        
        # Delete face embeddings if they exist
        global faces_data
        faces_data = init_face_embeddings()
        if user_id in faces_data:
            del faces_data[user_id]
            save_face_embeddings(faces_data)
        
        # Save data
        save_dataframes(users_df, transactions_df)
        
        return jsonify({'success': True, 'message': 'User deleted successfully'})
    except Exception as e:
        print(f"Delete user error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/add-meal-swipes', methods=['POST'])
def api_admin_add_meal_swipes():
    """Add meal swipes to a user"""
    global users_df, transactions_df
    
    if 'admin_logged_in' not in session or not session.get('admin_logged_in'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        data = request.json
        user_id = data.get('user_id', '').strip()
        amount = int(data.get('amount', 1))
        
        if not user_id:
            return jsonify({'success': False, 'error': 'User ID is required'}), 400
        
        if amount <= 0:
            return jsonify({'success': False, 'error': 'Amount must be greater than 0'}), 400
        
        # Check if user exists
        user_idx = users_df[users_df['user_id'] == user_id].index
        if user_idx.empty:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        # Update balance
        new_balance = users_df.loc[user_idx[0], 'balance'] + amount
        users_df.loc[user_idx[0], 'balance'] = new_balance
        
        # Create transaction record
        transaction = {
            'transaction_id': f"txn-{uuid.uuid4().hex[:8]}",
            'user_id': user_id,
            'university_id': users_df.loc[user_idx[0], 'university_id'],
            'transaction_type': 'admin_added',
            'amount': amount,
            'balance_after': new_balance,
            'timestamp': datetime.now().isoformat(),
            'status': 'completed'
        }
        transactions_df = pd.concat([transactions_df, pd.DataFrame([transaction])], ignore_index=True)
        
        # Save data
        save_dataframes(users_df, transactions_df)
        
        return jsonify({
            'success': True,
            'new_balance': float(new_balance),
            'message': f'Added {amount} meal swipe(s)'
        })
    except Exception as e:
        print(f"Add meal swipes error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/delete-meal-swipes', methods=['POST'])
def api_admin_delete_meal_swipes():
    """Remove meal swipes from a user"""
    global users_df, transactions_df
    
    if 'admin_logged_in' not in session or not session.get('admin_logged_in'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 403
    
    try:
        data = request.json
        user_id = data.get('user_id', '').strip()
        amount = int(data.get('amount', 1))
        
        if not user_id:
            return jsonify({'success': False, 'error': 'User ID is required'}), 400
        
        if amount <= 0:
            return jsonify({'success': False, 'error': 'Amount must be greater than 0'}), 400
        
        # Check if user exists
        user_idx = users_df[users_df['user_id'] == user_id].index
        if user_idx.empty:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        # Check balance
        current_balance = users_df.loc[user_idx[0], 'balance']
        if current_balance < amount:
            return jsonify({
                'success': False,
                'error': f'Insufficient balance. Current balance: {current_balance}'
            }), 400
        
        # Update balance
        new_balance = current_balance - amount
        users_df.loc[user_idx[0], 'balance'] = new_balance
        
        # Create transaction record
        transaction = {
            'transaction_id': f"txn-{uuid.uuid4().hex[:8]}",
            'user_id': user_id,
            'university_id': users_df.loc[user_idx[0], 'university_id'],
            'transaction_type': 'admin_removed',
            'amount': -amount,
            'balance_after': new_balance,
            'timestamp': datetime.now().isoformat(),
            'status': 'completed'
        }
        transactions_df = pd.concat([transactions_df, pd.DataFrame([transaction])], ignore_index=True)
        
        # Save data
        save_dataframes(users_df, transactions_df)
        
        return jsonify({
            'success': True,
            'new_balance': float(new_balance),
            'message': f'Removed {amount} meal swipe(s)'
        })
    except Exception as e:
        print(f"Delete meal swipes error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

