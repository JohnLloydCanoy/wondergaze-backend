import cv2
import mediapipe as mp
import pyautogui
import time
import os
import json
import threading
from PIL import Image
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- SECURITY & API CONFIGURATION ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("[FATAL ERROR] Gemini API key not found. Please check your .env file.")
    exit()

client = genai.Client(api_key=api_key)

# --- FASTAPI SERVER SETUP ---
app = FastAPI()

# Allow your Next.js frontend to talk to this Python backend without security errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the latest Gemini JSON data
latest_ui_data = []

# --- API ENDPOINTS ---

@app.get("/api/ui-elements")
def get_ui_elements():
    """Next.js will constantly ping this URL to fetch the latest screen data."""
    return {"status": "success", "data": latest_ui_data}

class ClickTarget(BaseModel):
    ymin: int
    xmin: int
    ymax: int
    xmax: int

@app.post("/api/execute-click")
def execute_click(target: ClickTarget):
    global latest_ui_data
    
    # 1. Get your actual screen resolution (e.g., 1920x1080)
    screen_w, screen_h = pyautogui.size()
    
    # 2. Find the exact center of the Gemini 1000x1000 bounding box
    center_x_1000 = target.xmin + ((target.xmax - target.xmin) / 2)
    center_y_1000 = target.ymin + ((target.ymax - target.ymin) / 2)
    
    # 3. Translate that to real Windows pixels
    real_x = int((center_x_1000 / 1000) * screen_w)
    real_y = int((center_y_1000 / 1000) * screen_h)
    
    print(f"\n[SYSTEM] Executing physical click at coordinates: ({real_x}, {real_y})")
    
    # 4. Perform the actual OS-level click
    pyautogui.click(real_x, real_y)
    
    # 5. Clear the data so the Next.js overlay hides itself automatically
    latest_ui_data = []
    
    return {"status": "success"}

# --- INITIALIZATION ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def analyze_screen_with_gemini(image_path):
    global latest_ui_data
    print(f"\n[SYSTEM] Screenshot successfully saved to: {image_path}")
    print("[SYSTEM] Sending to Gemini 2.5 Flash for EXHAUSTIVE UI analysis...")
    print("[SYSTEM] Waiting for response (Expect 2-4 seconds)...")
    
    try:
        img = Image.open(image_path)
        
        prompt = """
        You are a highly precise AI accessibility screen reader. Analyze this screenshot of a computer desktop.
        You MUST find EVERY SINGLE clickable interactive element on this screen. Do not skip small elements.
        
        You must specifically look for and include:
        1. OS System Controls (Minimize, Maximize, Close 'X' buttons in the top right corners).
        2. Browser Controls (URL address bar, Back/Forward buttons, refresh button).
        3. Taskbar & System Tray (Start button, tiny icons in the bottom right like Wi-Fi, Clock, Volume).
        4. In-text links (Tiny citation numbers like '1', '3 Sources', etc.).
        5. All standard web buttons, search bars, and tabs.
        
        Return the result STRICTLY as a raw JSON array of objects. Do NOT wrap it in markdown block quotes like ```json.
        Each object must have:
        - "id": A unique integer starting from 1.
        - "label": A precise description (e.g., "Edge Browser Close Button", "Wi-Fi Icon", "Citation Link 1").
        - "box_2d": An array of four numbers [ymin, xmin, ymax, xmax] representing the exact bounding box of the element. Use a 1000x1000 normalized grid where [0,0] is top-left and [1000,1000] is bottom-right.
        """
        
        start_time = time.time()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, img]
        )
        end_time = time.time()
        
        print("\n=== GEMINI ANALYSIS COMPLETE ===")
        print(f"Latency: {round(end_time - start_time, 2)} seconds")
        
        # Clean the text response to ensure it is pure JSON, then store it globally
        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        latest_ui_data = json.loads(clean_json)
        
        print(f"[SYSTEM] Successfully parsed {len(latest_ui_data)} interactive elements!")
        print("[SYSTEM] Data is now available at http://localhost:8000/api/ui-elements\n")
        
        return clean_json
        
    except Exception as e:
        print(f"\n[ERROR] Gemini API failed or returned invalid JSON: {e}")
        return None

def start_vision_engine():
    print("Starting WonderGaze Vision Engine (Background Thread)...")
    cap = cv2.VideoCapture(0)
    
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

            image = cv2.flip(image, 1)
            image_h, image_w, _ = image.shape
            
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    
                    top_eyelid = face_landmarks.landmark[159].y
                    bottom_eyelid = face_landmarks.landmark[145].y
                    eye_distance = bottom_eyelid - top_eyelid
                    
                    current_time = time.time()
                    
                    if eye_distance < 0.007: 
                        if current_time - last_blink_time > 0.2: 
                            blink_count += 1
                            last_blink_time = current_time
                            print(f"Blink {blink_count} detected.")
                    
                    if current_time - last_blink_time > 1.5: 
                        blink_count = 0
                        
                    if blink_count == 2 and (current_time - click_cooldown > 2.0):
                        print("\n>>> INTENTIONAL DOUBLE BLINK DETECTED! <<<")
                        
                        print("[SYSTEM] Hiding camera preview for clean capture...")
                        try:
                            cv2.destroyWindow('WonderGaze AI - Agentic Trigger')
                            cv2.waitKey(1) 
                            time.sleep(0.3) 
                        except:
                            pass 
                        
                        screenshot_path = os.path.join(os.getcwd(), "wondergaze_capture.png")
                        pyautogui.screenshot(screenshot_path)
                        
                        # Call Gemini to parse the screen
                        analyze_screen_with_gemini(screenshot_path)
                        
                        blink_count = 0
                        click_cooldown = current_time

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
    os._exit(0) # Closes the server if you close the camera

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Start the camera loop in a separate background thread
    vision_thread = threading.Thread(target=start_vision_engine, daemon=True)
    vision_thread.start()
    
    # 2. Start the FastAPI server on the main thread
    print("\n[SYSTEM] Starting FastAPI Server...")
    print("[SYSTEM] API Endpoint available at: http://localhost:8000/api/ui-elements")
    uvicorn.run(app, host="0.0.0.0", port=8000)