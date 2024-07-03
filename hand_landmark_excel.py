import cv2
import mediapipe as mp
import numpy as np
#import pyfirmata
import math
from hand_measurement import save_to_excel


mpHand = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHand.Hands(min_detection_confidence=0.8)


cap = cv2.VideoCapture(0)
ws, hs = 1920, 1080
cap.set(3, ws)
cap.set(4, hs)

#port = "COM3"
#board = pyfirmata.Arduino(port)
#servo_pinX = board.get_pin('d:9:s') #pin 9 Arduino
#servo_pinY = board.get_pin('d:10:s') #pin 10 Arduino


# let user to input his name and the mice model

# Prompt the user for their name
name = input("Please enter your name: ")

# Prompt the user for the model of their mouse
mouse_model = input("Please enter the model of your mouse: ")


while cap.isOpened():
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
                                  mpDraw.DrawingSpec(color=(0, 255,255), thickness=4, circle_radius=7),
                                  mpDraw.DrawingSpec(color=(0, 0, 0), thickness = 4))

        #Hand Tracking
        singleHandDetection = multiHandDetection[0]
        for lm in singleHandDetection.landmark:
            h, w, c = img.shape
            lm_x, lm_y = int(lm.x*w), int(lm.y*h)
            lmList.append([lm_x, lm_y])

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
        cv2.circle(img, (px_wrist, py_wrist), 15, (255, 0, 255), cv2.FILLED)
        cv2.putText(img, str((px_wrist, py_wrist)), (px_wrist + 10, py_wrist - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
      
        #thumb_cmc coordinates
        cv2.circle(img, (px_1, py_1), 15, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, str((px_1, py_1)), (px_1 + 10, py_1 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #thumb_mcp coordinates
        cv2.circle(img, (px_2, py_2), 15, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, str((px_2, py_2)), (px_2 + 10, py_2 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #thumb_ip coordinates
        cv2.circle(img, (px_3, py_3), 15, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, str((px_3, py_3)), (px_3 + 10, py_3 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #thumb_tip coordinates
        cv2.circle(img, (px_4, py_4), 15, (0, 0, 255), cv2.FILLED)
        cv2.putText(img, str((px_4, py_4)), (px_4 + 10, py_4 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #index_mcp coordinates
        cv2.circle(img, (px_5, py_5), 15, (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_5, py_5)), (px_5 + 10, py_5 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #index_pip coordinates
        cv2.circle(img, (px_6, py_6), 15, (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_6, py_6)), (px_6 + 10, py_6 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #index_dip coordinates
        cv2.circle(img, (px_7, py_7), 15, (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_7, py_7)), (px_7 + 10, py_7 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #index_tip coordinates
        cv2.circle(img, (px_8, py_8), 15, (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_8, py_8)), (px_8 + 10, py_8 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #middle_mcp coordinates
        cv2.circle(img, (px_9, py_9), 15, (255, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_9, py_9)), (px_9 + 10, py_9 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #middle_pip coordinates
        cv2.circle(img, (px_10, py_10), 15, (255, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_10, py_10)), (px_10 + 10, py_10 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #middle_dip coordinates
        cv2.circle(img, (px_11, py_11), 15, (255, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_11, py_11)), (px_11 + 10, py_11 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #middle_tip coordinates
        cv2.circle(img, (px_12, py_12), 15, (255, 255, 0), cv2.FILLED)
        cv2.putText(img, str((px_12, py_12)), (px_12 + 10, py_12 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #ring_mcp coordinates
        cv2.circle(img, (px_13, py_13), 15, (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str((px_13, py_13)), (px_13 + 10, py_13 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #ring_pip coordinates
        cv2.circle(img, (px_14, py_14), 15, (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str((px_14, py_14)), (px_14 + 10, py_14 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #ring_dip coordinates
        cv2.circle(img, (px_15, py_15), 15, (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str((px_15, py_15)), (px_15 + 10, py_15 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #ring_tip coordinates
        cv2.circle(img, (px_16, py_16), 15, (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str((px_16, py_16)), (px_16 + 10, py_16 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #pinky_mcp coordinates
        cv2.circle(img, (px_17, py_17), 15, (0, 255, 255), cv2.FILLED)
        cv2.putText(img, str((px_17, py_17)), (px_17 + 10, py_17 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #pinky_pip coordinates
        cv2.circle(img, (px_18, py_18), 15, (0, 255, 255), cv2.FILLED)
        cv2.putText(img, str((px_18, py_18)), (px_18 + 10, py_18 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #pinky_dip coordinates
        cv2.circle(img, (px_19, py_19), 15, (0, 255, 255), cv2.FILLED)
        cv2.putText(img, str((px_19, py_19)), (px_19 + 10, py_19 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        #pinky_tip coordinates
        cv2.circle(img, (px_20, py_20), 15, (0, 255, 255), cv2.FILLED)
        cv2.putText(img, str((px_20, py_20)), (px_20 + 10, py_20 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        

        
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
        cv2.putText(img,f'WT_TB_CMC: {str(distance_wt_cmc)} mm', (10,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        cv2.putText(img,f'TB_CMC_MCP: {str(distance_thumb_cmc_mcp)} mm', (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        cv2.putText(img, f'TB_MCP_IP: {str(distance_thumb_mcp_ip)} mm', (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)
        cv2.putText(img, f'TB_IP_TIP: {str(distance_thumb_ip_tip)} mm', (10, 120), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 3)

        # display distance between wrist to index finger
        cv2.putText(img, f'WT_IF_MCP: {str(distance_wt_if_mcp)} mm', (10, 150), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 3)
        cv2.putText(img, f'IF_MCP_PIP: {str(distance_if_mcp_pip)} mm', (10, 180), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 3)
        cv2.putText(img, f'IF_PIP_DIP: {str(distance_if_pip_dip)} mm', (10, 210), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 3)
        cv2.putText(img, f'IF_DIP_TIP: {str(distance_if_dip_tip)} mm', (10, 240), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 3)

        # dispaly distance between middle fingers
        cv2.putText(img, f'MF_MCP_PIP: {str(distance_mf_mcp_pip)} mm', (10, 270), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 3)
        cv2.putText(img, f'MF_PIP_DIP: {str(distance_mf_pip_dip)} mm', (10, 300), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 3)
        cv2.putText(img, f'MF_DIP_TIP: {str(distance_mf_dip_tip)} mm', (10, 330), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 3)

        # display distance between ring fingers
        cv2.putText(img, f'RF_MCP_PIP: {str(distance_rf_mcp_pip)} mm', (10, 370), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 3)
        cv2.putText(img, f'RF_PIP_DIP: {str(distance_rf_pip_dip)} mm', (10, 400), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 3)
        cv2.putText(img, f'RF_DIP_TIP: {str(distance_rf_dip_tip)} mm', (10, 430), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 3)

        # display distance between pinky fingers
        cv2.putText(img, f'PF_MCP_PIP: {str(distance_pf_mcp_pip)} mm', (10, 470), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 3)
        cv2.putText(img, f'PF_PIP_DIP: {str(distance_pf_pip_dip)} mm', (10, 500), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 3)
        cv2.putText(img, f'PF_DIP_TIP: {str(distance_pf_dip_tip)} mm', (10, 530), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 3)
      
        # display distance of hand width and length
        cv2.putText(img, f'HAND LENGHT: {str(distance_wt_mf_tip)} mm', (10, 570), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 3)
        cv2.putText(img, f'HAND WIDTH: {str(distance_thumb_pf_tip)} mm', (10, 600), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 3)

        # convert position to degree value
        servoX = int(np.interp(px, [0, ws], [180, 0]))
        servoY = int(np.interp(py, [0, hs], [0, 180]))
        #cv2.rectangle(img, (40, 50), (350, 10), (0, 255, 255), cv2.FILLED)
        #cv2.putText(img, f'Servo X: {servoX} deg', (50, 850), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
        #cv2.putText(img, f'Servo Y: {servoY} deg', (50, 900), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
        cv2.putText(img, f'HX MDE RESEARCH - BY TIM YEH', (1500, 1000), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

        #servo_pinX.write(servoX)
        #servo_pinY.write(servoY)

        print(f'Hand Position x: {px} y: {py}')
        #print(f'Servo Value x: {servoX} y: {servoY}')

        # Print the user's name and mouse model
        print(f"Hello, {name}! You have a {mouse_model} mouse.")

        
        # collect_hand_data()
        save_to_excel('hand_measurements.xlsx', 
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