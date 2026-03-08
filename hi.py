"""
ESP32-CAM Face Recognition Script
Detects and recognizes specific faces from stored images
"""

import cv2
import serial
import serial.tools.list_ports
import time
import urllib.request
import numpy as np
import os
import pickle
from pathlib import Path

# Configuration
ESP32_IP = "192.168.4.1"  # Update this to your ESP32's IP
SERIAL_PORT = 'COM8'
BAUD_RATE = 9600

# ============================================
# STEP 1: Create folders for storing face images
# ============================================
FACE_DB_FOLDER = "face_database"
KNOWN_FACES_FOLDER = os.path.join(FACE_DB_FOLDER, "known_faces")
ENCODINGS_FILE = os.path.join(FACE_DB_FOLDER, "face_encodings.pkl")

# Create folders if they don't exist
os.makedirs(KNOWN_FACES_FOLDER, exist_ok=True)
os.makedirs(FACE_DB_FOLDER, exist_ok=True)

# ============================================
# STEP 2: Face Recognition Class
# ============================================
class FaceRecognizer:
    def __init__(self):
        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # Load face recognition model (LBPH face recognizer)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Dictionary to map label IDs to names
        self.label_to_name = {}
        self.name_to_label = {}
        
        # Load existing encodings if available
        self.load_encodings()
    
    def load_encodings(self):
        """Load previously saved face encodings"""
        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, 'rb') as f:
                    data = pickle.load(f)
                    
                    if isinstance(data, dict):
                        self.label_to_name = data.get('label_to_name', {})
                        self.name_to_label = {v: k for k, v in self.label_to_name.items()}
                        
                        # Load model if exists
                        if 'model_data' in data and data['model_data']:
                            import base64
                            model_bytes = base64.b64decode(data['model_data'])
                            temp_file = "temp_model.xml"
                            with open(temp_file, 'wb') as mf:
                                mf.write(model_bytes)
                            self.recognizer.read(temp_file)
                            os.remove(temp_file)
                            print(f"✅ Loaded {len(self.label_to_name)} known faces")
                    else:
                        print("⚠️ Unknown data format in encodings file")
                        
            except Exception as e:
                print(f"❌ Error loading encodings: {e}")
    
    def save_encodings(self):
        """Save face encodings to file"""
        try:
            # Save the model to a temporary file
            temp_model = "temp_model.xml"
            self.recognizer.write(temp_model)
            
            # Read the model file as bytes
            with open(temp_model, 'rb') as f:
                model_bytes = f.read()
            
            # Convert to base64 for safe storage in PKL
            import base64
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
            
            # Prepare data for saving
            data = {
                'model_data': model_b64,
                'label_to_name': self.label_to_name,
                'timestamp': time.time(),
                'version': '2.0'
            }
            
            # Save to PKL file
            with open(ENCODINGS_FILE, 'wb') as f:
                pickle.dump(data, f)
            
            # Clean up temp file
            if os.path.exists(temp_model):
                os.remove(temp_model)
            
            print(f"✅ Saved {len(self.label_to_name)} face encodings")
            return True
            
        except Exception as e:
            print(f"❌ Error saving encodings: {e}")
            return False
    
    def add_face(self, name, image_path=None, frame=None):
        """Add a new face to the database"""
        if frame is None:
            # Load image from file
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Could not load image: {image_path}")
                return False
        else:
            img = frame
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        
        if len(faces) == 0:
            print(f"❌ No face detected in image for {name}")
            return False
        
        # Get the largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to standard size
        face_roi = cv2.resize(face_roi, (200, 200))
        
        # Assign label
        if name not in self.name_to_label:
            label = len(self.name_to_label)
            self.name_to_label[name] = label
            self.label_to_name[label] = name
        else:
            label = self.name_to_label[name]
        
        # Train with this face
        self.recognizer.update([face_roi], np.array([label]))
        
        print(f"✅ Added face for: {name}")
        self.save_encodings()
        return True
    
    def recognize_face(self, frame):
        """Recognize faces in the frame - FIXED METHOD NAME"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        
        recognized_names = []
        face_locations = []
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # Predict
            label, confidence = self.recognizer.predict(face_roi)
            
            # Lower confidence value means better match
            # 0-50 is excellent, 50-80 is good, 80+ is poor
            if confidence < 80 and label in self.label_to_name:
                name = self.label_to_name[label]
                confidence_text = f"{100 - confidence:.1f}%"
            else:
                name = "Unknown"
                confidence_text = ""
            
            recognized_names.append(name)
            face_locations.append((x, y, w, h, confidence_text))
        
        return recognized_names, face_locations

# ============================================
# STEP 3: Initialize Serial Connection
# ============================================
print("Available COM ports:")
ports = serial.tools.list_ports.comports()
for port in ports:
    print(f"  {port.device}: {port.description}")

# Initialize serial connection
ser = None
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"✅ Serial connected on {SERIAL_PORT} at {BAUD_RATE} baud")
except Exception as e:
    print(f"❌ Serial connection failed: {e}")
    ser = None

# ============================================
# STEP 4: Initialize ESP32-CAM Connection
# ============================================
def test_connection(ip, port=80):
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False

print(f"\n🔍 Testing connection to ESP32-CAM at {ESP32_IP}...")
if test_connection(ESP32_IP):
    print("✅ ESP32-CAM is reachable")
else:
    print(f"❌ Cannot reach ESP32-CAM at {ESP32_IP}")
    print("   Check if you're connected to the right WiFi network")
    print("   Trying alternative IPs...")
    
    # Try alternative IPs
    alternative_ips = ["192.168.4.1", "192.168.1.210", "192.168.0.210"]
    for ip in alternative_ips:
        if test_connection(ip):
            ESP32_IP = ip
            print(f"✅ Found ESP32-CAM at {ESP32_IP}")
            break

# Initialize video stream
stream_url = f"http://{ESP32_IP}:81/stream"
cap = cv2.VideoCapture(stream_url)

def get_frame_from_esp32():
    try:
        url = f"http://{ESP32_IP}:80/capture"
        with urllib.request.urlopen(url, timeout=3) as response:
            img_array = np.array(bytearray(response.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_array, -1)
            return frame
    except:
        return None

# ============================================
# STEP 5: Initialize Face Recognizer
# ============================================
recognizer = FaceRecognizer()

# Menu for adding faces
print("\n" + "="*50)
print("FACE RECOGNITION SYSTEM")
print("="*50)
print("1. Start recognition with existing database")
print("2. Add new faces to database from folder")
print("3. Capture face from ESP32-CAM to add")
choice = input("Enter your choice (1/2/3): ")

if choice == '2':
    # Add faces from image files
    print("\n📸 Place face images in the 'known_faces' folder")
    print("   Name format: person_name.jpg or person_name.png")
    
    image_files = list(Path(KNOWN_FACES_FOLDER).glob('*.*'))
    if len(image_files) == 0:
        print("❌ No images found in known_faces folder")
        print("   Please add images and restart the program")
        exit()
    else:
        for img_file in image_files:
            name = img_file.stem  # filename without extension
            print(f"Processing {name}...")
            recognizer.add_face(name, str(img_file))
        
        print(f"\n✅ Added {len(image_files)} faces to database!")

elif choice == '3':
    # Capture face from ESP32-CAM
    print("\n📸 Capturing face from ESP32-CAM")
    print("Look at the camera and press SPACE to capture, ESC to cancel")
    
    while True:
        frame = get_frame_from_esp32()
        if frame is None:
            ret, frame = cap.read()
            if not ret:
                print(".", end="", flush=True)
                time.sleep(0.5)
                continue
        
        # Show frame
        cv2.imshow("Capture Face - Press SPACE to capture", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space key
            name = input("\nEnter person's name: ")
            if name:
                recognizer.add_face(name, frame=frame)
                print(f"✅ Added {name} to database")
            break
        elif key == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()

# ============================================
# STEP 6: Main Recognition Loop
# ============================================
print("\n🎯 Starting face recognition... Press 'q' to quit")
print("-" * 50)

last_status = ""
stable_count = 0
frame_count = 0
no_frame_count = 0

while True:
    # Get frame from ESP32-CAM
    ret, frame = cap.read()
    
    if not ret:
        frame = get_frame_from_esp32()
        if frame is None:
            no_frame_count += 1
            if no_frame_count % 10 == 0:
                print(f"⏳ Waiting for frames... ({no_frame_count})")
            time.sleep(0.5)
            continue
        else:
            no_frame_count = 0
        ret = True
    
    if not ret:
        print("❌ Failed to capture frame")
        break

    # Process every 2nd frame
    frame_count += 1
    if frame_count % 2 != 0:
        # Still show the frame
        if frame is not None:
            cv2.imshow("ESP32-CAM Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Recognize faces - USING CORRECT METHOD NAME: recognize_face()
    names, face_locations = recognizer.recognize_face(frame)
    
    # Draw results
    for i, (x, y, w, h, confidence) in enumerate(face_locations):
        # Determine color based on recognition
        if i < len(names) and names[i] != "Unknown":
            color = (0, 255, 0)  # Green for known faces
        else:
            color = (0, 0, 255)  # Red for unknown
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Add name label
        if i < len(names):
            label = f"{names[i]} {confidence}"
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Determine overall status
    if len(names) > 0:
        if any(name != "Unknown" for name in names):
            known_names = [name for name in names if name != "Unknown"]
            status = f"KNOWN_FACE: {', '.join(known_names)}"
        else:
            status = "UNKNOWN_FACE"
    else:
        status = "NO_FACE"

    # Send status via serial after stabilization
    if status != last_status:
        stable_count = 0
        last_status = status
    else:
        stable_count += 1
    
    if stable_count >= 5 and ser and ser.is_open:  # Changed to >= for more frequent updates
        try:
            ser.write((status + "\n").encode())
            print(f"📤 Sent: {status}")
            stable_count = 0  # Reset to avoid repeated sends
        except Exception as e:
            print(f"❌ Serial write error: {e}")

    # Add information to frame
    cv2.putText(frame, f"Status: {status[:30]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"ESP32: {ESP32_IP}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame, f"Faces: {len(face_locations)}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show FPS or frame count
    cv2.putText(frame, f"Frame: {frame_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show frame
    cv2.imshow("ESP32-CAM Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
print("\n🧹 Cleaning up...")
cap.release()
cv2.destroyAllWindows()
if ser and ser.is_open:
    ser.close()
    print("✅ Serial port closed")
print("✅ Done!")