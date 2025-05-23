#--------------------------------  @RUN ----------------------------------------------------------
# If python version is 3.10 or above you will have to downgrade to 3.9 to install certain packages
#-------------------Imports and Setup-----------------------------------------------------------------------------------
import cv2                  # For webcam capture & image processing
import mediapipe as mp      # For hand landmark detection
import pyautogui            # To control mouse cursor
pyautogui.FAILSAFE = False  # Disables emergency stop when moving mouse to corner
import math                 # To calculate distance between thumb and index fingertip : Pinch detection

#------------------ Setting always on top window mode --------------------------------------------
# import pygetwindow as gw
# import win32gui
# import win32con
#
# def make_window_always_on_top(window_name):
#     try:
#         window = gw.getWindowswithTitle(window_name)[0]
#         win32gui.SetWindowPos(window._hWnd,win32con.HWND_TOPMOST,0,0,0,0,win32con.SWP_NOMOVE|win32con.SWP_NOSIZE)
#     except IndexError:
#         pass # Window not found yet (first Few frame) , Ignore
#-------------------Initialize camera and libraries --------------------------------------------------------------------
cap = cv2.VideoCapture(0)           # Starts webcam
mpHands = mp.solutions.hands        # Load MediaPipe  hand module
hands = mpHands.Hands(max_num_hands=1)             # Create hand detection object ( Only detect one hand even if multiple are there )
mpDraw = mp.solutions.drawing_utils # For drawing hand landmarks on image

# Variables for relative movement
prev_cx, prev_cy = None, None       # Store previous fingertip position for relative movement
sensitivity = 8                   # Scale factor to make finger motion control cursor speed
thumb_to_index_threshold = 20        # Enable index tracking to move the pointer

# Variables for smoothing movement
smooth_cx, smooth_cy = None,None
smooth_factor = .1 # Lower = Smoother, Slower response (eg: 0.2 - 0.5)
#Main Loop (runs every frame): infinite loop for continuous webcam capture
while True:
    #-------------------- Capture Frame & Convert -----------------------------------
    success, img = cap.read()                        # Reads a frame from webcam
    #img = cv2.flip(img, 1)                          # Mirror image <Not needed>
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # Convert image to RGB (MediaPipe requires RGB)
    #-------------------- Hand Detection ---------------------------------------------
    results = hands.process(imgRGB)                  # Detects hands & returns landmarks
    #--------------------- If Hand Detected -------------------------------------------
    if results.multi_hand_landmarks:                 # Checks if any hand landmarks are found
        #----------------- Loop through Hands & Landmarks -----------------------------
        for handLms in results.multi_hand_landmarks:        # For every detected hand
            for id, lm in enumerate(handLms.landmark):      # For each landmark (21 per hand)
                h, w, c = img.shape                         # Get image height(h), width(w), channels(c)
                #----------------------- Thumb Tracking -------------------------------
                if id == 4:
                    thumb_x = float((1 - lm.x)*w)
                    thumb_y = float(lm.y * h)
                #----------------------- Pinky Tracking --------------------------------
                if id == 20:
                    pinky_x = float((1 - lm.x) * w)
                    pinky_y = float(lm.y * h)
                #----------------------- Ring Finger Tracking --------------------------
                if id == 16:
                    ring_x = float((1- lm.x) * w)
                    ring_y = float(lm.y * h)
                #----------------------- Middle Finger Tracking ------------------------
                if id == 12 :
                    middle_x = float((1-lm.x) * w)
                    middle_y = float(lm.y * h)
                #----------------------- Index Finger Tip Tracking ---------------------------------
                if id == 8:  # Index fingertip              # Track landmark 8 (index fingertip) for cursor control
                    index_x_tip = float((1 - lm.x) * w)
                    index_y_tip = float(lm.y * h)
                    #----------- Calculate Position & Relative Movement -----------------------------------
                    flipped_lm_x = 1 - lm.x         # Landmark (lm.x) is normalized (0-1). Flipping x because image was not flipped
                    cx = float(flipped_lm_x * w)      # Convert normalized landmark to pixel coordinates (cx,cy)
                    cy = float(lm.y * h)
                    #------------Initialize smoothing position
                    if smooth_cx is None and smooth_cy is None:
                        smooth_cx, smooth_cy = cx,cy
                    #------------Exponential Moving Average for smoothing
                    smooth_cx = smooth_cx + (cx - smooth_cx) * smooth_factor
                    smooth_cy = smooth_cy + (cy - smooth_cy) * smooth_factor
                    #------------ Move Mouse Relative to Previous Position ---------------------------------
                    if prev_cx is not None and prev_cy is not None:     # Calculate difference (dx,dy) between current and previous fingertip position
                        dx = (smooth_cx - prev_cx) * sensitivity               # Multiply by sensitivity
                        dy = (smooth_cy - prev_cy) * sensitivity
                #-------------------- Index finger near knuckle tracking -------------------
                if id == 5:
                    index_x = float((1 - lm.x) * w)
                    index_y = float(lm.y * h)
                    #Only move if thumb is closer to index finger
                    distance_thumb_to_index = math.hypot(index_x - thumb_x, index_y - thumb_y)
                    if distance_thumb_to_index < thumb_to_index_threshold:
                        cv2.putText(img, "Index finger tracking on", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3)
                        pyautogui.moveRel(dx, dy)                             # Move mouse relative to its current position
                    #------------------ Update Previous Position ---------------------------------------------
                    prev_cx, prev_cy = smooth_cx, smooth_cy                   # Store current coordinates for next frame

            #============================== MOUSE BUTTONS LOGIC ========================================================

            # -------------------- Left Clicking Mechanism --------------------------------------------
            distance_left_click = math.hypot(index_x_tip - thumb_x, index_y_tip - thumb_y)
            left_click_threshold = 20
            if distance_left_click < left_click_threshold:
                pyautogui.click()
                cv2.putText(img, "Left Click", (250,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3)
            # -------------------- Right Clicking Mechanism ------------------------------------------
            distance_right_click = math.hypot(middle_x - thumb_x, middle_y - thumb_y)
            right_click_threshold = 20
            if distance_right_click < right_click_threshold:
                pyautogui.rightClick()
                cv2.putText(img, "Right Click",(250,250),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            #---------------------- button 3 Mechanism %% ADD CLICK EVENT HERE %%%%%%%%%%%%%%%%%%%%%%%%%------------------------------------------
            distance_button3_click = math.hypot(middle_x - thumb_x, middle_y - thumb_y)
            button3_click_threshold = 20
            if distance_button3_click < button3_click_threshold:
                # pyautogui.leftClick()
                cv2.putText(img,"left_click",(250,250),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,225),3)
            #---------------------- Draw Hand Landmarks (For Visualization) -----------------------------------
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)     # Draw lines connecting hand landmarks on image for feedback
    #----------------- If No Hand Detected , Reset Previous Position -----------------------------------------
    else:
        prev_cx, prev_cy = None, None       # Reset tracking when hand disappears to avoid cursor jumps
        smooth_cx,smooth_cy = None,None     # Reset Smoothing

    #------------------------ Display camera feed ---------------------------------------------------------------
    custom_width,custom_height = 1024,600
    resized_img = cv2.resize(img,(custom_width,custom_height))
    cv2.imshow("Gesture Mouse", resized_img)        # Show webcam window with hand landmarks drawn
    # make_window_always_on_top("Gesture Mouse")
    if cv2.waitKey(1) & 0xFF == ord('q'):           # Press 'q' key to exit the loop
        break
#--------- Cleanup ----------------------------------------------------------------------------------------------
cap.release()                   # Releases webcam
cv2.destroyAllWindows()         # Close windows
