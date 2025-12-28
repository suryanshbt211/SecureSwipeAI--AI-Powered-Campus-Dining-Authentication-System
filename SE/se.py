# Install: pip install deepface opencv-python huggingface_hub ultralytics tf-keras

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2
import numpy as np
from deepface import DeepFace
import pickle
import os

class FaceRegistrationSystem:
    def __init__(self):
        # Load YOLOv8 Face Detection model
        print("Loading YOLOv8 Face Detection model...")
        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection", 
            filename="model.pt"
        )
        self.yolo_model = YOLO(model_path)
        
        # Storage for registered faces
        self.known_face_embeddings = []
        self.known_face_names = []
        self.database_file = "face_database.pkl"
        
        # Load existing database
        self.load_database()
        
        # Registration mode
        self.registration_mode = False
        self.registration_name = ""
        
    def load_database(self):
        """Load registered faces from disk"""
        if os.path.exists(self.database_file):
            with open(self.database_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_embeddings = data['embeddings']
                self.known_face_names = data['names']
            print(f"Loaded {len(self.known_face_names)} registered faces")
        else:
            print("No existing database found. Starting fresh.")
    
    def save_database(self):
        """Save registered faces to disk"""
        data = {
            'embeddings': self.known_face_embeddings,
            'names': self.known_face_names
        }
        with open(self.database_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Database saved with {len(self.known_face_names)} faces")
    
    def get_face_embedding(self, face_img):
        """Extract face embedding using DeepFace"""
        try:
            # DeepFace.represent returns a list of embeddings
            embedding_objs = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet",  # Options: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace
                enforce_detection=False
            )
            if len(embedding_objs) > 0:
                return np.array(embedding_objs[0]["embedding"])
            return None
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)
    
    def register_face(self, frame, face_bbox):
        """Register a new face with embeddings"""
        x1, y1, x2, y2 = face_bbox
        
        # Extract and save face region temporarily
        face_region = frame[y1:y2, x1:x2]
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, face_region)
        
        # Get embedding
        embedding = self.get_face_embedding(temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if embedding is not None:
            self.known_face_embeddings.append(embedding)
            self.known_face_names.append(self.registration_name)
            self.save_database()
            print(f"✓ Registered: {self.registration_name}")
            return True
        else:
            print("✗ Could not extract face embedding. Please try again.")
            return False
    
    def recognize_face(self, frame, face_bbox):
        """Compare face against registered faces"""
        x1, y1, x2, y2 = face_bbox
        
        # Extract face region
        face_region = frame[y1:y2, x1:x2]
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, face_region)
        
        # Get embedding
        embedding = self.get_face_embedding(temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if embedding is None:
            return "Unknown"
        
        # No registered faces yet
        if len(self.known_face_embeddings) == 0:
            return "Not Registered"
        
        # Compare with all registered faces
        max_similarity = -1
        best_match_name = "Not Registered"
        
        for idx, known_embedding in enumerate(self.known_face_embeddings):
            similarity = self.cosine_similarity(embedding, known_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                if similarity > 0.5:  # Threshold for matching
                    best_match_name = self.known_face_names[idx]
        
        if max_similarity > 0.5:
            confidence = max_similarity * 100
            return f"{best_match_name} ({confidence:.1f}%)"
        
        return "Not Registered"
    
    def run(self):
        """Main loop for real-time face recognition"""
        cap = cv2.VideoCapture(0)
        
        print("\n=== Face Registration & Recognition System ===")
        print("Controls:")
        print("  'r' - Register your face")
        print("  'q' - Quit")
        print("  'd' - Delete all registrations")
        print("=============================================\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces using YOLOv8
            results = self.yolo_model(frame, verbose=False)
            
            # Process each detected face
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    if self.registration_mode:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        label = "REGISTERING..."
                        color = (255, 0, 0)
                        
                        success = self.register_face(frame, (x1, y1, x2, y2))
                        if success:
                            self.registration_mode = False
                    else:
                        name = self.recognize_face(frame, (x1, y1, x2, y2))
                        
                        if name == "Not Registered":
                            color = (0, 0, 255)  # RED
                        else:
                            color = (0, 255, 0)  # GREEN
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = name
                    
                    # Draw label
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 35), (x1 + label_size[0] + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, "Press 'r' to Register | 'q' to Quit | 'd' to Delete All", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Face Recognition System", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("\n--- Registration Mode ---")
                self.registration_name = input("Enter your name: ").strip()
                if self.registration_name:
                    self.registration_mode = True
                    print("Look at the camera...")
            elif key == ord('d'):
                confirm = input("\nDelete all registrations? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    self.known_face_embeddings = []
                    self.known_face_names = []
                    if os.path.exists(self.database_file):
                        os.remove(self.database_file)
                    print("✓ All registrations deleted")
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    system = FaceRegistrationSystem()
    system.run()
