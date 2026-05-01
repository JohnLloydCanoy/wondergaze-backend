import cv2
import mediapipe as mp
import pyautogui
import time
import os

# --- INITIALIZATION ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def analyze_screen_with_gemini(image_path):
    """
    This is where we will hook up the Gemini 1.5 API.
    It will take the screenshot, send it to Google, and return the UI coordinates.
    """
    print(f"\n[SYSTEM] Screenshot successfully saved to: {image_path}")
    print("[SYSTEM] Preparing to send to Gemini Vision API for UI analysis...")
    # The API code will go here in the next step!

def start_vision_engine():
    print("Starting WonderGaze Vision Engine (Gaze-and-Snap Mode)...")
    cap = cv2.VideoCapture(0)
    
    # Blink Logic Variables
    last_blink_time = 0
    blink_count = 0
    click_cooldown = 0
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success: continue

            # Flip for a natural mirror view
            image = cv2.flip(image, 1)
            image_h, image_w, _ = image.shape
            
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    
                    # --- BLINK DETECTION (The Trigger) ---
                    # Landmarks 145 & 159 are the top and bottom of the right eyelid
                    top_eyelid = face_landmarks.landmark[159].y
                    bottom_eyelid = face_landmarks.landmark[145].y
                    eye_distance = bottom_eyelid - top_eyelid
                    
                    current_time = time.time()
                    
                    # If distance is extremely small, the eye is closed
                    if eye_distance < 0.004: 
                        if current_time - last_blink_time > 0.2: # Debounce
                            blink_count += 1
                            last_blink_time = current_time
                            print(f"Blink {blink_count} detected.")
                    
                    # Reset if too much time passes between blinks
                    if current_time - last_blink_time > 1.0:
                        blink_count = 0
                        
                    # --- THE ACTION: DOUBLE BLINK SNAPSHOT ---
                    if blink_count == 2 and (current_time - click_cooldown > 2.0):
                        print("\n>>> INTENTIONAL DOUBLE BLINK DETECTED! <<<")
                        
                        # 1. Define where to save the image
                        screenshot_path = os.path.join(os.getcwd(), "wondergaze_capture.png")
                        
                        # 2. Take the screenshot
                        pyautogui.screenshot(screenshot_path)
                        
                        # 3. Pass it to our upcoming Gemini function
                        analyze_screen_with_gemini(screenshot_path)
                        
                        # 4. Reset variables so it doesn't loop infinitely
                        blink_count = 0
                        click_cooldown = current_time

                    # Draw visual markers (Just the eyelid points for debugging)
                    top_y = int(top_eyelid * image_h)
                    bottom_y = int(bottom_eyelid * image_h)
                    x_coord = int(face_landmarks.landmark[159].x * image_w)
                    
                    cv2.circle(image, (x_coord, top_y), 2, (0, 255, 0), -1)
                    cv2.circle(image, (x_coord, bottom_y), 2, (0, 0, 255), -1)

            cv2.imshow('WonderGaze AI - Agentic Trigger', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_vision_engine()