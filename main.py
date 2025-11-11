import cv2
import torch
import numpy as np
import time
from collections import deque

try:
    import Jetson.GPIO as GPIO
except ImportError:
    print("Warning: Jetson.GPIO not found. Running in simulation mode.")
    GPIO = None

# Import YOLOv5
import yolov5

class JetsonPWMServo:
    """PWM servo controller for Jetson GPIO"""
    
    def __init__(self, pan_pin, frequency=50):
        """
        Initialize PWM servo controller
        
        Args:
            pan_pin: GPIO pin for pan servo (e.g., 33)
            frequency: PWM frequency in Hz (50Hz is standard for servos)
        """
        self.pan_pin = pan_pin
        self.frequency = frequency
        self.pan_pwm = None
        
        self.init_pwm()
    
    def init_pwm(self):
        """Initialize GPIO and PWM for servo control"""
        if GPIO is None:
            print("GPIO not available. Servo control disabled.")
            return
        
        try:
            # Set GPIO mode
            GPIO.setmode(GPIO.BOARD)
            
            # Setup pan pin
            GPIO.setup(self.pan_pin, GPIO.OUT)
            
            # Create PWM object for pan servo
            self.pan_pwm = GPIO.PWM(self.pan_pin, self.frequency)
            
            # Start PWM with neutral position (90 degrees)
            self.pan_pwm.start(7.5)  # 1.5ms / 20ms = 7.5%
            
            print(f"PWM Servo initialized on pin {self.pan_pin} (pan)")
        except Exception as e:
            print(f"Error initializing PWM: {e}")
            self.pan_pwm = None
    
    def angle_to_duty_cycle(self, angle):
        """
        Convert servo angle (0-180) to PWM duty cycle
        Typical servo: 1ms (5%) = 0°, 1.5ms (7.5%) = 90°, 2ms (10%) = 180°
        """
        # Clamp angle
        angle = max(0, min(180, angle))
        # Convert: duty_cycle = (angle / 180) * 5 + 5
        # This maps 0° to 5%, 90° to 7.5%, 180° to 10%
        duty_cycle = (angle / 180.0) * 5.0 + 5.0
        return duty_cycle
    
    def set_angles(self, pan_angle):
        """Set servo angle via PWM"""
        try:
            if self.pan_pwm is not None:
                pan_duty = self.angle_to_duty_cycle(pan_angle)
                self.pan_pwm.ChangeDutyCycle(pan_duty)
        except Exception as e:
            print(f"Error setting servo angle: {e}")
    
    def cleanup(self):
        """Clean up PWM and GPIO"""
        if GPIO is None:
            return
        
        try:
            if self.pan_pwm is not None:
                self.pan_pwm.stop()
            GPIO.cleanup()
        except Exception as e:
            print(f"Error during cleanup: {e}")


class ServoCamera:
    """Camera tracker that uses YOLO to detect persons and servo to keep them centered"""
    
    def __init__(self, 
                 model_name='yolov5',
                 camera_index=0,
                 pan_pin=33,
                 target_class=0,  # Class 0 is 'person' in COCO dataset
                 confidence_threshold=0.5,
                 use_gpu=True):
        """
        Initialize the servo camera tracker for Jetson
        
        Args:
            model_name: YOLOv5 model ('yolov5su' recommended for performance)
            camera_index: Camera index (0 for default, or use GStreamer pipeline)
            pan_pin: GPIO pin for pan servo control
            target_class: Class ID to detect (0 = person)
            confidence_threshold: Confidence threshold for detections
            use_gpu: Use GPU for inference (set to True on Jetson)
        """
        self.target_class = target_class
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        
        # Load YOLOv5su model optimized for performance
        print("Loading YOLOv5su model...")
        self.model = yolov5.load(model_name)
        self.model.conf = confidence_threshold
        
        # Use GPU if available and requested
        if use_gpu:
            if torch.cuda.is_available():
                try:
                    self.model.to('cuda')
                    print("Model loaded on GPU")
                except Exception as e:
                    print(f"Warning: failed to move model to GPU: {e}. Using CPU.")
            else:
                print("CUDA not available — using CPU")
        else:
            print("Model loaded on CPU")
        
        print(f"Model loaded: {model_name}")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Warning: Camera index {camera_index} not opened. Trying fallback 0.")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera. Check camera connection or pipeline.")
        
        # For better performance on Jetson, set camera properties
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # If properties not available, read one frame to determine size
        if self.frame_width == 0 or self.frame_height == 0:
            ret_tmp, tmp_frame = self.cap.read()
            if ret_tmp and tmp_frame is not None:
                h, w = tmp_frame.shape[:2]
                self.frame_width = w
                self.frame_height = h
            else:
                print("Warning: unable to determine camera resolution, defaulting to 640x480")
                self.frame_width = 640
                self.frame_height = 480
        
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        print(f"Camera initialized: {self.frame_width}x{self.frame_height}")
        
        # Initialize PWM servo controller
        self.servo = JetsonPWMServo(pan_pin)
        
        # PID control parameters for servo
        self.kp_pan = 0.5  # Pan (horizontal) proportional gain
        self.pan_angle = 90  # Initial pan angle (center)
        self.pan_min, self.pan_max = 0, 180
        
        # Smoothing buffer for bounding boxes
        self.bbox_buffer = deque(maxlen=5)
        
        # Initialize servo to center position
        self.servo.set_angles(self.pan_angle)
    
    def send_servo_command(self, pan_angle):
        """
        Send PWM command to servo to move to specified angle
        """
        if self.servo is None:
            return
        
        # Clamp angle to valid range
        pan_angle = max(self.pan_min, min(self.pan_max, pan_angle))
        
        try:
            self.servo.set_angles(pan_angle)
        except Exception as e:
            print(f"Error sending servo command: {e}")
    
    def get_smoothed_bbox(self, bbox):
        """Apply smoothing to bounding box to reduce jitter"""
        if bbox is not None:
            self.bbox_buffer.append(bbox)
        
        if len(self.bbox_buffer) == 0:
            return None
        
        # Average the buffered bounding boxes
        bboxes = np.array(list(self.bbox_buffer))
        smoothed_bbox = np.mean(bboxes, axis=0)
        return smoothed_bbox
    
    def detect_persons(self, frame):
        """
        Detect persons in frame using YOLOv5su
        Returns the bounding box of the detected person closest to center
        """
        # Run inference
        results = self.model(frame)
        
        # Try to extract detections tensor in a robust way
        try:
            detections = results.xyxy[0]  # typical yolov5 results (x1,y1,x2,y2,conf,cls)
        except Exception:
            # fallback for other result object shapes/APIs
            try:
                detections = results.pred[0]
            except Exception:
                return None, None
        
        if detections is None or len(detections) == 0:
            return None, None
        
        # Move to CPU and convert to numpy for parsing
        try:
            dets = detections.cpu().numpy()
        except Exception:
            # If already numpy
            dets = np.array(detections)
        
        person_detections = []
        centers = []
        for row in dets:
            # row expected: [x1, y1, x2, y2, conf, cls]
            if len(row) < 6:
                continue
            x1, y1, x2, y2, conf, cls_id = row[:6]
            if int(cls_id) == self.target_class and conf >= self.confidence_threshold:
                person_detections.append([x1, y1, x2, y2, conf, cls_id])
                centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        
        if len(person_detections) == 0:
            return None, None
        
        centers = np.array(centers)
        # Distance from frame center
        distances = np.sqrt((centers[:, 0] - self.center_x)**2 + 
                            (centers[:, 1] - self.center_y)**2)
        
        # Get closest detection
        closest_idx = np.argmin(distances)
        closest_detection = np.array(person_detections[closest_idx])
        closest_center = centers[closest_idx]
        
        return closest_detection, closest_center
    
    def calculate_servo_angles(self, bbox_center):
        """
        Calculate servo angle based on detected person position
        Uses PID-like control to center the person in frame (horizontally only)
        """
        if bbox_center is None:
            return self.pan_angle
        
        # Calculate error (distance from center)
        error_x = bbox_center[0] - self.center_x  # Positive = person to the right
        
        # Convert pixel error to angle adjustment
        # Assuming ~0.1 degrees per pixel (adjust based on your camera FOV)
        pan_adjustment = error_x * 0.1 * self.kp_pan
        
        # Update angle
        new_pan = self.pan_angle + pan_adjustment
        
        # Clamp to valid range (servo limits)
        new_pan = max(self.pan_min, min(self.pan_max, new_pan))
        
        return new_pan
    
    def run(self, display=True):
        """Main loop for camera tracking"""
        print("Starting camera tracker... Press 'q' to quit")
        
        frame_count = 0
        fps_counter = deque(maxlen=30)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                start_time = time.time()
                
                # Detect persons
                bbox, bbox_center = self.detect_persons(frame)
                
                if bbox is not None and bbox_center is not None:
                    # Smooth bounding box
                    smoothed_bbox = self.get_smoothed_bbox(bbox[:4])
                    
                    # Calculate servo angle
                    pan_angle = self.calculate_servo_angles(bbox_center)
                    self.pan_angle = pan_angle
                    
                    # Send command to servo
                    self.send_servo_command(pan_angle)
                    
                    if display:
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw center point
                        cx, cy = int(bbox_center[0]), int(bbox_center[1])
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                        
                        # Draw frame center crosshair
                        cv2.line(frame, (self.center_x - 10, self.center_y), 
                                (self.center_x + 10, self.center_y), (255, 0, 0), 2)
                        cv2.line(frame, (self.center_x, self.center_y - 10), 
                                (self.center_x, self.center_y + 10), (255, 0, 0), 2)
                        
                        # Draw info
                        cv2.putText(frame, f"Pan: {pan_angle:.1f}°", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, "Person Tracked", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    if display:
                        # Draw frame center crosshair
                        cv2.line(frame, (self.center_x - 10, self.center_y), 
                                (self.center_x + 10, self.center_y), (0, 0, 255), 2)
                        cv2.line(frame, (self.center_x, self.center_y - 10), 
                                (self.center_x, self.center_y + 10), (0, 0, 255), 2)
                        cv2.putText(frame, "No person detected", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps_counter.append(1 / elapsed if elapsed > 0 else 0)
                avg_fps = np.mean(fps_counter)
                
                if display:
                    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (self.frame_width - 150, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Servo Camera Tracker', frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        if self.servo is not None:
            self.servo.cleanup()
        print("Done")


if __name__ == "__main__":
    # Initialize servo camera tracker for Jetson with PWM control (pan only)
    tracker = ServoCamera(
        model_name='yolov5su',       # YOLOv5su for better performance on Jetson
        camera_index=0,              # Default camera (or GStreamer pipeline string)
        pan_pin=33,                  # GPIO pin for pan servo (adjust to your setup)
        target_class=0,              # Class 0 = person
        confidence_threshold=0.5,    # Detection confidence threshold
        use_gpu=True                 # Use GPU acceleration on Jetson
    )
    
    # Start tracking
    tracker.run(display=True)
