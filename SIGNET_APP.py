import tkinter as tk
from tkinter import ttk, messagebox, font
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
import sys
import multiprocessing as mp # CHANGED: Use multiprocessing
import queue # Keep this for the Empty exception
import time
import serial
import serial.tools.list_ports
from PIL import Image, ImageTk

# --- Configuration ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- AI & Processing Logic (Now in a Process) ---

def find_extreme_points_from_masks(left_mask, right_mask):
    """ Finds the innermost point (tip) for each instrument's polygon mask. """
    left_tip = max(left_mask, key=lambda point: point[0])
    right_tip = min(right_mask, key=lambda point: point[0])
    return tuple(left_tip.astype(int)), tuple(right_tip.astype(int))

# CHANGED: Inherit from mp.Process instead of threading.Thread
class TrackerProcess(mp.Process): 
    def __init__(self, config, frame_queue):
        super().__init__()
        self.config = config
        self.frame_queue = frame_queue
        # self.running = True # We can't use this flag easily across processes
        self.STABLE_CIRCLE_RADIUS = 180
        self.SMOOTHING_FACTOR = 0.05
        self.stable_focus_point = None
        self.PREDICT_KWARGS = {'device': self.config['device'], 
                             'verbose': False, 
                             'imgsz': 640, 
                             'half': True}

    def run(self):
        """Main AI processing loop. Assumes all inputs are validated."""
        
        try:
            model_path = resource_path(os.path.join("models", self.config["model_name"]))
            model = YOLO(model_path)
            self.class_names = model.names
            
            # CRITICAL FIX 1: Open the camera with cv2.CAP_DSHOW for stability
            cap = cv2.VideoCapture(self.config["video_source"], cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.config["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.config["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"AI Process: Camera provides {self.config['width']}x{self.config['height']}")

            serial_port = None
            if self.config["com_port"] != "None":
                serial_port = serial.Serial(self.config["com_port"], 115200, timeout=1)
                print(f"AI Process: Serial port {self.config['com_port']} connected.")
        
        except Exception as e:
            print(f"AI Process Error during initialization: {e}")
            self.frame_queue.put(("error", f"Error on start: {e}"))
            return

        print("AI Process: Starting tracking loop...")
        # We can't use self.running, so we just loop until the process is terminated
        while True: 
            try:
                start_time = time.time()
                ret, full_frame = cap.read()
                if not ret:
                    print("AI Process: End of video stream.")
                    break

                if self.config["video_mode"] == "3D (Left Half)":
                    frame = full_frame[0:self.config["height"], 0:self.config["width"]//2]
                else:
                    frame = full_frame

                overlay_blue = frame.copy()
                overlay_green = frame.copy()
                overlay_yellow = frame.copy()

                results = model.predict(frame, **self.PREDICT_KWARGS)
                detections = []
                if results[0].masks:
                    for mask, box in zip(results[0].masks.xy, results[0].boxes):
                        if box.conf[0] > self.config["confidence"]:
                            detections.append({'mask': mask, 'box': box.xyxy[0].cpu().numpy()})

                if len(detections) >= 2:
                    detections.sort(key=lambda d: (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]), reverse=True)
                    top_two = detections[:2]

                    center_x1 = (top_two[0]['box'][0] + top_two[0]['box'][2]) / 2
                    center_x2 = (top_two[1]['box'][0] + top_two[1]['box'][2]) / 2
                    
                    if center_x1 < center_x2:
                        left_inst, right_inst = top_two[0], top_two[1]
                    else:
                        left_inst, right_inst = top_two[1], top_two[0]

                    left_tip, right_tip = find_extreme_points_from_masks(left_inst['mask'], right_inst['mask'])
                    realtime_midpoint = (int((left_tip[0] + right_tip[0]) / 2), int((left_tip[1] + right_tip[1]) / 2))

                    if self.stable_focus_point is None:
                        self.stable_focus_point = realtime_midpoint
                    else:
                        dist = np.linalg.norm(np.array(self.stable_focus_point) - np.array(realtime_midpoint))
                        if dist > self.STABLE_CIRCLE_RADIUS:
                            self.stable_focus_point = (
                                int(self.stable_focus_point[0] * (1 - self.SMOOTHING_FACTOR) + realtime_midpoint[0] * self.SMOOTHING_FACTOR),
                                int(self.stable_focus_point[1] * (1 - self.SMOOTHING_FACTOR) + realtime_midpoint[1] * self.SMOOTHING_FACTOR)
                            )

                    if serial_port and serial_port.is_open:
                        data_string = f"<{self.stable_focus_point[0]},{self.stable_focus_point[1]}>\n"
                        serial_port.write(data_string.encode('utf-8'))

                    PALE_BLUE = (255, 230, 204)
                    YELLOW = (0, 255, 255)
                    TRANSPARENT_GREEN = (0, 255, 0)

                    cv2.fillPoly(overlay_green, [left_inst['mask'].astype(np.int32)], TRANSPARENT_GREEN)
                    cv2.fillPoly(overlay_green, [right_inst['mask'].astype(np.int32)], TRANSPARENT_GREEN)
                    frame = cv2.addWeighted(overlay_green, 0.3, frame, 0.7, 0)
                    
                    cv2.circle(overlay_blue, realtime_midpoint, 60, PALE_BLUE, -1)
                    frame = cv2.addWeighted(overlay_blue, 0.7, frame, 0.3, 0) 

                    cv2.circle(overlay_yellow, self.stable_focus_point, self.STABLE_CIRCLE_RADIUS, YELLOW, 3)
                    frame = cv2.addWeighted(overlay_yellow, 0.5, frame, 0.5, 0)
                    
                    cv2.line(frame, left_tip, right_tip, PALE_BLUE, 2)
                    cv2.circle(frame, left_tip, 10, PALE_BLUE, -1)
                    cv2.circle(frame, right_tip, 10, PALE_BLUE, -1)
                    cv2.putText(frame, f"ROBOT FOCUS: {self.stable_focus_point}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_queue.put(("frame", rgb_frame))
            
            except (KeyboardInterrupt, SystemExit):
                break # Allow exiting
            except Exception as e:
                print(f"AI Process Error in loop: {e}")
                self.frame_queue.put(("error", f"Error in loop: {e}"))
                break

        cap.release()
        if serial_port and serial_port.is_open:
            serial_port.close()
        self.frame_queue.put(("done", None))
        print("AI Process: Stopped.")

    # We can't use this flag method, so the stop is handled by self.terminate()
    # def stop(self):
    #     self.running = False

# --- Validation Process ---
# CHANGED: Inherit from mp.Process
class ValidationProcess(mp.Process):
    def __init__(self, config, frame_queue):
        super().__init__()
        self.config = config
        self.frame_queue = frame_queue

    def run(self):
        """Runs a series of quick checks to validate the user's settings."""
        try:
            # 1. Check GPU
            self.frame_queue.put(("status", "Checking GPU..."))
            if not torch.cuda.is_available():
                self.frame_queue.put(("validation_error", "Validation FAILED: PyTorch cannot detect NVIDIA GPU."))
                return
            self.frame_queue.put(("status", "GPU OK."))

            # 2. Check Model
            self.frame_queue.put(("status", "Loading model..."))
            model_path = resource_path(os.path.join("models", self.config["model_name"]))
            if not os.path.exists(model_path):
                 self.frame_queue.put(("validation_error", f"Validation FAILED: Model file not found at {model_path}"))
                 return
            _ = YOLO(model_path) # Try to load the model
            self.frame_queue.put(("status", "Model OK."))

            # 3. Check Camera
            self.frame_queue.put(("status", "Opening camera..."))
            # CRITICAL FIX 3: Also use CAP_DSHOW in the validation step
            cap = cv2.VideoCapture(self.config["video_source"], cv2.CAP_DSHOW)
            if not cap.isOpened():
                self.frame_queue.put(("validation_error", f"Validation FAILED: Could not open Camera {self.config['video_source']}."))
                return
            cap.release() # Immediately release it
            self.frame_queue.put(("status", "Camera OK."))

            # 4. Check COM Port
            if self.config["com_port"] != "None":
                self.frame_queue.put(("status", "Checking COM port..."))
                # CRITICAL FIX 2: Do NOT open the port. Just check if it exists.
                # This prevents the hardware deadlock.
                available_ports = [port.device for port in serial.tools.list_ports.comports()]
                if self.config["com_port"] not in available_ports:
                    self.frame_queue.put(("validation_error", f"Validation FAILED: COM Port {self.config['com_port']} not found."))
                    return
                # OLD, BUGGY CODE:
                # try:
                #     sp = serial.Serial(self.config["com_port"], 115200, timeout=1)
                #     sp.close()
                # except serial.SerialException as e:
                #     self.frame_queue.put(("validation_error", f"Validation FAILED: COM Port {self.config['com_port']} error: {e}"))
                #     return
                self.frame_queue.put(("status", "COM Port OK."))

            # 5. Success
            self.frame_queue.put(("validation_success", "✅ Validation Successful. Ready to Start."))

        except Exception as e:
            self.frame_queue.put(("validation_error", f"Validation FAILED: An unexpected error occurred: {e}"))

# --- Main GUI Application ---

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Surgical Tracker v2.0")
        self.root.geometry("1280x900")
        self.root.configure(bg="#2E2E2E")

        self.tracking_active = False
        self.ai_process = None # CHANGED: Renamed from ai_thread
        self.frame_queue = mp.Queue() # CHANGED: Use mp.Queue
        self.validation_passed = False 

        # --- Style Configuration ---
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('.', background='#2E2E2E', foreground='white', fieldbackground='#555555', selectbackground='#4a6984')
        self.style.configure('TButton', padding=6, relief="flat", background="#007ACC", foreground="white")
        self.style.map('TButton', background=[('active', '#005F9E'), ('disabled', '#555555')])
        self.style.configure('TLabel', padding=5, background='#2E2E2E', foreground='white')
        self.style.configure('TRadiobutton', padding=5, background='#2E2E2E', foreground='white')
        self.style.configure('TMenubutton', padding=6, background='#555555', foreground='white', relief='flat')
        self.style.configure('Green.TButton', background='#008A00', foreground='white')
        self.style.map('Green.TButton', background=[('active', '#006A00'), ('disabled', '#555555')])

        # --- GUI Layout ---
        
        self.video_label = ttk.Label(root, text="Waiting for video feed...", style='TLabel', anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(fill=tk.X)

        # ... (GUI layout widgets are unchanged) ...
        # 2a. Video Mode (2D/3D)
        video_mode_frame = ttk.Frame(control_frame)
        ttk.Label(video_mode_frame, text="Video Mode:").pack(side=tk.LEFT, padx=5)
        self.video_mode = tk.StringVar(value="3D (Left Half)")
        self.radio_2d = ttk.Radiobutton(video_mode_frame, text="2D", variable=self.video_mode, value="2D (Full 1920x1080)")
        self.radio_3d = ttk.Radiobutton(video_mode_frame, text="3D (Left Half)", variable=self.video_mode, value="3D (Left Half)")
        self.radio_2d.pack(side=tk.LEFT)
        self.radio_3d.pack(side=tk.LEFT)
        video_mode_frame.pack(side=tk.LEFT, padx=10)

        # 2b. Model Selection
        ttk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=(10, 5))
        self.models_list = self.find_models()
        self.model_var = tk.StringVar(value=self.models_list[0] if self.models_list else "No Models Found")
        self.model_menu = ttk.OptionMenu(control_frame, self.model_var, self.models_list[0] if self.models_list else None, *self.models_list)
        self.model_menu.pack(side=tk.LEFT, padx=5)

        # 2c. Camera Selection
        ttk.Label(control_frame, text="Camera:").pack(side=tk.LEFT, padx=(10, 5))
        self.camera_list = self.find_cameras()
        self.camera_names = [name for name, _ in self.camera_list]
        self.camera_var = tk.StringVar(value=self.camera_names[0])
        self.camera_menu = ttk.OptionMenu(control_frame, self.camera_var, self.camera_names[0], *self.camera_names)
        self.camera_menu.pack(side=tk.LEFT, padx=5)

        # 2d. COM Port Selection
        ttk.Label(control_frame, text="COM Port:").pack(side=tk.LEFT, padx=(10, 5))
        self.com_ports_list = self.find_com_ports()
        self.com_var = tk.StringVar(value=self.com_ports_list[0])
        self.com_menu = ttk.OptionMenu(control_frame, self.com_var, self.com_ports_list[0], *self.com_ports_list)
        self.com_menu.pack(side=tk.LEFT, padx=5)

        # 2e. NEW Button Layout
        self.start_button = ttk.Button(control_frame, text="Start Tracking", command=self.start_tracking, state=tk.DISABLED, style="Green.TButton")
        self.start_button.pack(side=tk.RIGHT, padx=10)
        
        self.validate_button = ttk.Button(control_frame, text="Validate Settings", command=self.run_validation)
        self.validate_button.pack(side=tk.RIGHT, padx=5)
        
        self.status_bar = ttk.Label(root, text="Ready. Please validate settings first.", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_video_feed() # Start the GUI's update loop

    def find_models(self):
        models_dir = resource_path("models")
        if not os.path.exists(models_dir):
            messagebox.showerror("Error", "The 'models' folder is missing. Please create it and add your .pt model files.")
            return ["No 'models' folder"]
        models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        return models if models else ["No models found"]

    def find_cameras(self):
        print("GUI: Populating static camera list (0, 1, 2, 3).")
        # CRITICAL FIX: We no longer probe for cameras at startup.
        # Probing (opening/closing the camera) in the main GUI thread
        # causes a race condition and freezes the app.
        # This is a much more stable solution. The user must
        # know which camera index (0, 1, 2...) is their device.
        arr = [("Camera 0", 0), ("Camera 1", 1), ("Camera 2", 2), ("Camera 3", 3)]
        print(f"GUI: Found {len(arr)} cameras.")
        return arr
        
        # OLD, BUGGY CODE that caused freezes:
        # print("GUI: Scanning for available cameras...")
        # index = 0
        # arr = []
        # max_cams_to_check = 5 
        # while index < max_cams_to_check:
        #     # Use cv2.CAP_DSHOW for better Windows compatibility
        #     cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) 
        #     if cap.isOpened():
        #         arr.append((f"Camera {index}", index))
        #         cap.release()
        #     index += 1
        # print(f"GUI: Found {len(arr)} cameras.")
        # return arr if arr else [("No Cameras Found", -1)]

    def find_com_ports(self):
        ports = serial.tools.list_ports.comports()
        port_names = [port.device for port in ports]
        return ["None"] + port_names if port_names else ["None"] 

    def get_current_config(self):
        """Helper function to read all GUI settings."""
        selected_cam_name = self.camera_var.get()
        selected_cam_id = -1
        for name, cam_id in self.camera_list:
            if name == selected_cam_name:
                selected_cam_id = cam_id
                break
        
        config = {
            "model_name": self.model_var.get(),
            "com_port": self.com_var.get(),
            "video_mode": self.video_mode.get(),
            "video_source": selected_cam_id,
            "confidence": 0.5, 
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        return config

    def run_validation(self):
        """Launches the ValidationProcess to check settings."""
        print("GUI: Validation button clicked.")
        self.validation_passed = False 
        self.set_controls_enabled(False) 
        self.start_button.config(state=tk.DISABLED) 
        self.status_bar.config(text="Validating...")
        
        config = self.get_current_config()

        if config["video_source"] == -1:
            messagebox.showerror("Error", "No valid camera selected or found.")
            self.set_controls_enabled(True)
            self.status_bar.config(text="Ready.")
            return
        if "No models" in config["model_name"]:
            messagebox.showerror("Error", "No AI model selected or found.")
            self.set_controls_enabled(True)
            self.status_bar.config(text="Ready.")
            return

        # Start the validation process
        # CHANGED: Use ValidationProcess
        vp = ValidationProcess(config, self.frame_queue)
        vp.daemon = True # Set as daemon so it exits with main app
        vp.start()

    def start_tracking(self):
        """Launches the main TrackerProcess."""
        if self.tracking_active:
            self.stop_tracking()
            return
            
        print("GUI: Start button clicked.")
        if not self.validation_passed:
            messagebox.showwarning("Warning", "Please run a successful validation before starting.")
            return

        self.tracking_active = True
        self.set_controls_enabled(False)
        self.validate_button.config(state=tk.DISABLED)
        self.start_button.config(text="Stop Tracking")
        self.status_bar.config(text=f"Tracking running... Using {self.model_var.get()}")

        config = self.get_current_config()
        # CHANGED: Use TrackerProcess
        self.ai_process = TrackerProcess(config, self.frame_queue)
        self.ai_process.daemon = True # Set as daemon
        self.ai_process.start()

    def stop_tracking(self):
        print("GUI: Stop command issued.")
        if self.ai_process and self.ai_process.is_alive():
            # CHANGED: Use terminate() to forcefully stop the process
            self.ai_process.terminate()
            self.ai_process.join(1) # Wait 1s for it to close
            
        self.tracking_active = False
        self.set_controls_enabled(True)
        self.validate_button.config(state=tk.NORMAL)
        self.start_button.config(text="Start Tracking", state=tk.NORMAL if self.validation_passed else tk.DISABLED)
        self.status_bar.config(text="Stopped.")
        self.video_label.config(image=None, text="Waiting for video feed...")
        
        # Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break


    def set_controls_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.radio_2d.config(state=state)
        self.radio_3d.config(state=state)
        self.model_menu.config(state=state)
        self.camera_menu.config(state=state)
        self.com_menu.config(state=state)
        self.validate_button.config(state=state)
        # Start button is handled by validation status

    def update_video_feed(self):
        """Checks the queue for new frames and messages."""
        try:
            while not self.frame_queue.empty():
                msg_type, data = self.frame_queue.get_nowait()
                
                if msg_type == "frame":
                    self.img = Image.fromarray(data)
                    w, h = self.video_label.winfo_width(), self.video_label.winfo_height()
                    if w > 10 and h > 10: 
                        self.img = self.img.resize((w, h), Image.Resampling.LANCZOS)
                    self.photo = ImageTk.PhotoImage(image=self.img)
                    self.video_label.config(image=self.photo, text="")
                
                elif msg_type == "validation_error":
                    messagebox.showerror("Validation Failed", data)
                    self.status_bar.config(text=f"Validation FAILED. Check settings.")
                    self.set_controls_enabled(True) 
                    self.validation_passed = False
                
                elif msg_type == "validation_success":
                    self.status_bar.config(text=data)
                    self.set_controls_enabled(True) 
                    self.start_button.config(state=tk.NORMAL) 
                    self.validation_passed = True
                
                elif msg_type == "error":
                    messagebox.showerror("AI Thread Error", data)
                    self.stop_tracking()
                
                elif msg_type == "status":
                    self.status_bar.config(text=data)
                
                elif msg_type == "done":
                    if self.tracking_active:
                        self.stop_tracking()

        except queue.Empty:
            pass 

        self.root.after(15, self.update_video_feed) # ~66 FPS target

    def on_closing(self):
        print("GUI: Closing application...")
        if self.tracking_active:
            self.stop_tracking()
        self.root.quit()
        self.root.destroy()

if __name__ == '__main__':
    # CRITICAL: This is required for multiprocessing to work in a frozen .exe
    mp.freeze_support() 
    
    if not os.path.exists(resource_path("models")):
        print("Fatal Error: 'models' folder not found. Creating one.")
        os.makedirs(resource_path("models"))
        print("Please add your .pt model files to the 'models' folder and restart.")
    
    root = tk.Tk()
    app = App(root)
    root.mainloop()