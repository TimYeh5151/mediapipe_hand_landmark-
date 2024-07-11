# 2024/07/06 created by tim yeh

# that uses the MediaPipe library to perform hand tracking and landmark detection from a webcam feed.
# Import necessary libraries : The script imports the required libraries, including OpenCV (cv2) for video capture and image processing, 
# MediaPipe (mediapipe) for hand tracking and landmark detection, NumPy (numpy) for numerical operations, and a custom module
# hand_measurement  for saving data to an Excel file.


import cv2
import mediapipe as mp
import numpy as np
import math
import time
from hand_measurement import save_to_excel
import os



# fps
prev_time = 0

# initial the last capture time
last_capture_time = 0
photo_count = 0  # Initialize the photo count
max_photos = 5  # Maximum number of photos to capture


# Initialize MediaPipe Hands and Drawing utilities : The script initializes the MediaPipe Hands and Drawing utilities, 
# which are used for hand detection and visualization, respectively. 

mpHand = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHand.Hands(min_detection_confidence=0.8)
hands = mpHand.Hands(max_num_hands=1)
#hands = mpHand.Hands(model_complexity=2)



# Initialize the webcam : The script initializes the webcam by capturing the first frame from the webcam,
# which is used as the initial frame for the video output.

cap = cv2.VideoCapture(0)
ws, hs = 1920, 1080

cap.set(3, ws)
cap.set(4, hs)

# Prompt the user for input name and mice model name
input_name = input("Enter input name: ")
mice_model_name = input("Enter mice model name: ")

# setup the snaps direction
direction = "sideview"


# calculate the angle for handlandmarks

def calculate_angle(lmList, p1, p2, p3):
    """
    Calculate the angle between three landmarks in degrees.
    """
    p1_coords = [lmList[p1][0], lmList[p1][1]]
    p2_coords = [lmList[p2][0], lmList[p2][1]]
    p3_coords = [lmList[p3][0], lmList[p3][1]]

    v1 = [p1_coords[0] - p2_coords[0], p1_coords[1] - p2_coords[1]]  # Vector from p2 to p1
    v2 = [p3_coords[0] - p2_coords[0], p3_coords[1] - p2_coords[1]]  # Vector from p2 to p3

    # Calculate the dot product
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]

    # Calculate the cross product
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]

    # Calculate the angle in radians
    angle_rad = math.atan2(cross_product, dot_product)

    # Convert to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg




while cap.isOpened() and photo_count < max_photos:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    multiHandDetection = results.multi_hand_landmarks #Hand Detection
    lmList = []
  
    
    if multiHandDetection:
        #Hand Visualization
        for id, lm in enumerate(multiHandDetection):
            mpDraw.draw_landmarks(img, lm, mpHand.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(color=(0, 255,255), thickness=3, circle_radius=3),
                                  mpDraw.DrawingSpec(color=(0, 0, 0), thickness = 3))

        #Hand Tracking
        singleHandDetection = multiHandDetection[0]
        for lm in singleHandDetection.landmark:
            h, w, c = img.shape
            lm_x, lm_y = int(lm.x*w), int(lm.y*h)
            lmList.append([lm_x, lm_y])

         
          # Check if lmList has enough elements
        if len(lmList) >= 21:
            # Calculate the angle between the index finger's PIP and DIP joints
            index_finger_pip = lmList[8]
            index_finger_dip = lmList[7]

            # Check if lmList has at least 6 elements for index_finger_mcp
            if len(lmList) >= 6:
                index_finger_mcp = lmList[5]
                index_angle = calculate_angle(lmList, 5, 6, 7)
                print(f"Index finger angle: {index_angle:.2f} degrees")
            else:
                print("Hand landmarks not detected correctly for index finger MCP joint.")
        else:
            print("Hand landmarks not detected correctly.")



        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        current_time = time.time()
        
        if current_time - last_capture_time >= 3:
            photo_name = f"{input_name}_{mice_model_name}_{int(current_time)}_{direction}.jpg"
    
            cv2.imwrite(photo_name, img)
            last_capture_time = current_time
            photo_count += 1
        
        cv2.imshow('frame', img)
        print(f"Photo saved: {photo_name}")
        photo_path = os.path.join('snapshots', photo_name)
        
        # Get coordinates for different landmarks
        wrist = lmList[0]
        #thumb
        thumb_cmc = lmList[1]
        thumb_mcp = lmList[2]
        thumb_ip = lmList[3]
        thumb_tip = lmList[4]
        #index_finger
        index_finger_mcp = lmList[5]
        index_finger_pip = lmList[6]
        index_finger_dip = lmList[7]
        index_tip = lmList[8]
        #middle_finger
        middle_finger_mcp = lmList[9]
        middle_finger_pip = lmList[10]
        middle_finger_dip = lmList[11]
        middle_tip = lmList[12]
        #ring_finger
        ring_finger_mcp = lmList[13]
        ring_finger_pip = lmList[14]
        ring_finger_dip = lmList[15]
        ring_tip = lmList[16]
        #pinky        
        pinky_mcp = lmList[17]
        pinky_pip = lmList[18]
        pinky_dip = lmList[19]
        pinky_tip = lmList[20]

        print(lmList)

        # draw point
        myLP_wrist = wrist
        # thumb
        myLP_thumb_cmc = thumb_cmc
        myLP_thumb_mcp = thumb_mcp
        myLP_thumb_ip = thumb_ip
        myLP = thumb_tip
                
        # index finger
        myLP_index_mcp = index_finger_mcp
        myLP_index_pip = index_finger_pip
        myLP_index_dip = index_finger_dip
        myLP_index_tip = index_tip

        # middle_finger
        myLP_middle_mcp = middle_finger_mcp
        myLP_middle_pip = middle_finger_pip
        myLP_middle_dip = middle_finger_dip
        myLP_middle_tip = middle_tip

        # ring finger
        myLP_ring_mcp = ring_finger_mcp
        myLP_ring_pip = ring_finger_pip
        myLP_ring_dip = ring_finger_dip
        myLP_ring_tip = ring_tip

        # pinky
        myLP_pinky_mcp = pinky_mcp
        myLP_pinky_pip = pinky_pip
        myLP_pinky_dip = pinky_dip
        myLP_pinky_tip = pinky_tip

        # joint coordinates
        px_wrist, py_wrist = myLP_wrist[0], myLP_wrist[1] #wrist_coordinates
        px_1, py_1 = myLP_thumb_cmc[0], myLP_thumb_cmc[1] #thumb_cmc_coordinates
        px_2, py_2 = myLP_thumb_mcp[0], myLP_thumb_mcp[1] #thumb_mcp_coordinates
        px_3, py_3 = myLP_thumb_ip[0], myLP_thumb_ip[1] #thumb_ip_coordinates
        px_4, py_4 = myLP[0], myLP[1] #thumb_tip_coordinates
        px_5, py_5 = myLP_index_mcp[0], myLP_index_mcp[1] #index_mcp_coordinates
        px_6, py_6 = myLP_index_pip[0], myLP_index_pip[1] #index_pip_coordinates
        px_7, py_7 = myLP_index_dip[0], myLP_index_dip[1] #index_dip_coordinates
        px_8, py_8 = myLP_index_tip[0], myLP_index_tip[1] #index_tip_coordinates
        px_9, py_9 = myLP_middle_mcp[0], myLP_middle_mcp[1] #middle_mcp_coordinates
        px_10, py_10 = myLP_middle_pip[0], myLP_middle_pip[1] #middle_pip_coordinates
        px_11, py_11 = myLP_middle_dip[0], myLP_middle_dip[1] #middle_dip_coordinates
        px_12, py_12 = myLP_middle_tip[0], myLP_middle_tip[1] #middle_tip_coordinates
        px_13, py_13 = myLP_ring_mcp[0], myLP_ring_mcp[1] #ring_mcp_coordinates
        px_14, py_14 = myLP_ring_pip[0], myLP_ring_pip[1] #ring_pip_coordinates
        px_15, py_15 = myLP_ring_dip[0], myLP_ring_dip[1] #ring_dip_coordinates
        px_16, py_16 = myLP_ring_tip[0], myLP_ring_tip[1] #ring_tip_coordinates
        px_17, py_17 = myLP_pinky_mcp[0], myLP_pinky_mcp[1] #pinky_mcp_coordinates
        px_18, py_18 = myLP_pinky_pip[0], myLP_pinky_pip[1] #pinky_pip_coordinates
        px_19, py_19 = myLP_pinky_dip[0], myLP_pinky_dip[1] #pinky_dip_coordinates
        px_20, py_20 = myLP_pinky_tip[0], myLP_pinky_tip[1] #pinky_tip_coordinates

        px, py = myLP[0], myLP[1]
 

        #wrist coordinates
        cv2.circle(img, (px_wrist, py_wrist), 10, (255, 0, 255), cv2.FILLED)
        cv2.putText(img, str((px_wrist, py_wrist)), (px_wrist + 10, py_wrist - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
      
        #thumb_cmc coordinates
        cv2.circle(img, (px_1, py_1), 10, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, str((px_1, py_1)), (px_1 + 10, py_1 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #thumb_mcp coordinates
        cv2.circle(img, (px_2, py_2), 10, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, str((px_2, py_2)), (px_2 + 10, py_2 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #thumb_ip coordinates
        cv2.circle(img, (px_3, py_3), 10, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, str((px_3, py_3)), (px_3 + 10, py_3 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #thumb_tip coordinates
        cv2.circle(img, (px_4, py_4), 10, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, str((px_4, py_4)), (px_4 + 10, py_4 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #index_mcp coordinates
        cv2.circle(img, (px_5, py_5), 10, (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_5, py_5)), (px_5 + 10, py_5 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #index_pip coordinates
        cv2.circle(img, (px_6, py_6), 10, (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_6, py_6)), (px_6 + 10, py_6 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #index_dip coordinates
        cv2.circle(img, (px_7, py_7), 10, (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_7, py_7)), (px_7 + 10, py_7 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #index_tip coordinates
        cv2.circle(img, (px_8, py_8), 10, (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_8, py_8)), (px_8 + 10, py_8 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #middle_mcp coordinates
        cv2.circle(img, (px_9, py_9), 10, (255, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_9, py_9)), (px_9 + 10, py_9 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #middle_pip coordinates
        cv2.circle(img, (px_10, py_10), 10, (255, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_10, py_10)), (px_10 + 10, py_10 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #middle_dip coordinates
        cv2.circle(img, (px_11, py_11), 10, (255, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_11, py_11)), (px_11 + 10, py_11 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #middle_tip coordinates
        cv2.circle(img, (px_12, py_12), 10, (255, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_12, py_12)), (px_12 + 10, py_12 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #ring_mcp coordinates
        cv2.circle(img, (px_13, py_13), 10, (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str((px_13, py_13)), (px_13 + 10, py_13 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #ring_pip coordinates
        cv2.circle(img, (px_14, py_14), 10, (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str((px_14, py_14)), (px_14 + 10, py_14 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #ring_dip coordinates
        cv2.circle(img, (px_15, py_15), 10, (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str((px_15, py_15)), (px_15 + 10, py_15 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #ring_tip coordinates
        cv2.circle(img, (px_16, py_16), 10, (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str((px_16, py_16)), (px_16 + 10, py_16 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #pinky_mcp coordinates
        cv2.circle(img, (px_17, py_17), 10, (0, 255, 255), cv2.FILLED)
        cv2.putText(img, str((px_17, py_17)), (px_17 + 10, py_17 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #pinky_pip coordinates
        cv2.circle(img, (px_18, py_18), 10, (0, 255, 255), cv2.FILLED)
        cv2.putText(img, str((px_18, py_18)), (px_18 + 10, py_18 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #pinky_dip coordinates
        cv2.circle(img, (px_19, py_19), 10, (0, 255, 255), cv2.FILLED)
        cv2.putText(img, str((px_19, py_19)), (px_19 + 10, py_19 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        #pinky_tip coordinates
        cv2.circle(img, (px_20, py_20), 10, (0, 255, 255), cv2.FILLED)
        cv2.putText(img, str((px_20, py_20)), (px_20 + 10, py_20 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        

        
        # joints distance
        # ws, hs = 1280, 720
        distance_wt_cmc =  round(math.sqrt((px_wrist - px_1)**2 + (py_wrist - py_1)**2) / 96 * 2.54 *10,2)
        distance_thumb_cmc_mcp =  round(math.sqrt((px_2 - px_1)**2 + (py_2 - py_1)**2) / 96 * 2.54*10,2)
        distance_thumb_mcp_ip =  round(math.sqrt((px_3 - px_2)**2 + (py_3 - py_2)**2) / 96 * 2.54*10,2)
        distance_thumb_ip_tip = round(math.sqrt((px_4 - px_3)**2 + (py_4 - py_3)**2) / 96 * 2.54*10,2)
        # index finger
        distance_wt_if_mcp = round(math.sqrt((px_wrist - px_5)**2 + (py_wrist - py_5)**2) / 96 * 2.54*10,2)
        distance_if_mcp_pip = round(math.sqrt((px_6 - px_5)**2 + (py_6 - py_5)**2) / 96 * 2.54*10, 2)
        distance_if_pip_dip = round(math.sqrt((px_7 - px_6)**2 + (py_7 - py_6)**2) / 96 * 2.54*10,2)
        distance_if_dip_tip = round(math.sqrt((px_8 - px_7)**2 + (py_8 - py_7)**2) / 96 * 2.54*10,2)
        # middle finger
        distance_mf_mcp_pip = round(math.sqrt((px_10 - px_9)**2 + (py_10 - py_9)**2) / 96 * 2.54*10,2)
        distance_mf_pip_dip = round(math.sqrt((px_11 - px_10)**2 + (py_11 - py_10)**2) / 96 * 2.54*10,2)
        distance_mf_dip_tip = round(math.sqrt((px_12 - px_11)**2 + (py_12 - py_11)**2) / 96 * 2.54*10,2)
        # ring finger
        distance_rf_mcp_pip = round(math.sqrt((px_14 - px_13)**2 + (py_14 - py_13)**2) / 96 * 2.54*10,2)
        distance_rf_pip_dip = round(math.sqrt((px_15 - px_14)**2 + (py_15 - py_14)**2) / 96 * 2.54*10,2)
        distance_rf_dip_tip = round(math.sqrt((px_16 - px_15)**2 + (py_16 - py_15)**2) / 96 * 2.54*10,2)
        # pinky finger
        distance_pf_mcp_pip = round(math.sqrt((px_18 - px_17)**2 + (py_18 - py_17)**2) / 96 * 2.54*10,2)
        distance_pf_pip_dip = round(math.sqrt((px_19 - px_18)**2 + (py_19 - py_18)**2) / 96 * 2.54*10,2)
        distance_pf_dip_tip = round(math.sqrt((px_20 - px_19)**2 + (py_20 - py_19)**2) / 96 * 2.54*10,2)


        # hand distance

        distance_wt_mf_tip =  round(math.sqrt((px_wrist - px_12)**2 + (py_wrist - py_12)**2) / 96 * 2.54*10,2)

        # hand width
        
        distance_thumb_pf_tip =  round(math.sqrt((px_20 - px_4)**2 + (py_20 - py_4)**2) / 96 * 2.54*10,2)

        # print distance        

        print(f'Wrist to thumb_cmc: {distance_wt_cmc:.2f} cm')
        #print(f'thumb_cmc to thumb_mcp: {distance_thumb_cmc_mcp:.2f} cm')
        print(f'thumb_cmc to thumb_mcp: {int(distance_thumb_cmc_mcp)} cm')
        
        # display distance between different joints on display
        
        # display distance between wrist and thumb 
        cv2.putText(img,f'WT_TB_CMC: {str(distance_wt_cmc)} mm', (10,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        cv2.putText(img,f'TB_CMC_MCP: {str(distance_thumb_cmc_mcp)} mm', (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        cv2.putText(img, f'TB_MCP_IP: {str(distance_thumb_mcp_ip)} mm', (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        cv2.putText(img, f'TB_IP_TIP: {str(distance_thumb_ip_tip)} mm', (10, 120), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

        # display distance between wrist to index finger
        cv2.putText(img, f'WT_IF_MCP: {str(distance_wt_if_mcp)} mm', (10, 150), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.putText(img, f'IF_MCP_PIP: {str(distance_if_mcp_pip)} mm', (10, 180), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.putText(img, f'IF_PIP_DIP: {str(distance_if_pip_dip)} mm', (10, 210), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.putText(img, f'IF_DIP_TIP: {str(distance_if_dip_tip)} mm', (10, 240), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        # dispaly distance between middle fingers
        cv2.putText(img, f'MF_MCP_PIP: {str(distance_mf_mcp_pip)} mm', (10, 270), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
        cv2.putText(img, f'MF_PIP_DIP: {str(distance_mf_pip_dip)} mm', (10, 300), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
        cv2.putText(img, f'MF_DIP_TIP: {str(distance_mf_dip_tip)} mm', (10, 330), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)

        # display distance between ring fingers
        cv2.putText(img, f'RF_MCP_PIP: {str(distance_rf_mcp_pip)} mm', (10, 370), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
        cv2.putText(img, f'RF_PIP_DIP: {str(distance_rf_pip_dip)} mm', (10, 400), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
        cv2.putText(img, f'RF_DIP_TIP: {str(distance_rf_dip_tip)} mm', (10, 430), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

        # display distance between pinky fingers
        cv2.putText(img, f'PF_MCP_PIP: {str(distance_pf_mcp_pip)} mm', (10, 470), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
        cv2.putText(img, f'PF_PIP_DIP: {str(distance_pf_pip_dip)} mm', (10, 500), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
        cv2.putText(img, f'PF_DIP_TIP: {str(distance_pf_dip_tip)} mm', (10, 530), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
      
        # display distance of hand width and length
        cv2.putText(img, f'HAND LENGHT: {str(distance_wt_mf_tip)} mm', (10, 570), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
        cv2.putText(img, f'HAND WIDTH: {str(distance_thumb_pf_tip)} mm', (10, 600), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

        # display fps
        cv2.putText(img, f'FPS: {int(fps)}', (10, 630), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

        # convert position to degree value
   
        cv2.putText(img, f'HX MDE RESEARCH - BY TIM YEH', (800, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

        
        # collect_hand_data()
        save_to_excel(f"{input_name}_{mice_model_name}_{direction}.xlsx", 
                      distance_thumb_cmc_mcp, 
                      distance_thumb_mcp_ip, 
                      distance_thumb_ip_tip, 
                      distance_wt_if_mcp, 
                      distance_if_mcp_pip, 
                      distance_if_pip_dip, 
                      distance_if_dip_tip, 
                      distance_mf_mcp_pip, 
                      distance_mf_pip_dip, 
                      distance_mf_dip_tip, 
                      distance_rf_mcp_pip, 
                      distance_rf_pip_dip, 
                      distance_rf_dip_tip, 
                      distance_pf_mcp_pip, 
                      distance_pf_pip_dip, 
                      distance_pf_dip_tip,
                      distance_wt_mf_tip,
                      distance_thumb_pf_tip)




        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cv2.destroyAllWindows()