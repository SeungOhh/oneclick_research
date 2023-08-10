import ultralytics
from ultralytics import YOLO
import numpy as np
import cv2
import logging
import datetime
import os
from collections import defaultdict
from util import send_sms

#%%
#### 모델 불러오기
model = YOLO('./runs/detect/train/weights/best.pt') 



#%% 
#### video 사용
video_path = "./video//test.mp4"
cap = cv2.VideoCapture(video_path)

#### webcam 사용
# cap = cv2.VideoCapture(0)


#%%
# log 파일 저장할 폴더 생성
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
path_save = './/runs//detect//predict_' + now + '//'
os.mkdir(path_save)
f = open(path_save + "log.txt","w")


#%% 
# 얼마나 자주 저장할지. 
# freq = 10: 10개 프레임마다 이미지 저장
img_num = 0
freq = 5


# danger가 있는걸 여러번 확인하면 문자를 보냄
# threshold_danger_counter = 10: 저장된 사진중 10장에서 danger가 있으면 문자를 보냄 
danger_counter = 0
threshold_danger_counter = 10


# 사람수 counter 
# threshold_people_counter = 10: 저장된 사진중 10장에서 박스가 6개 넘으면 문자를 보냄
people_counter = 0
results_people_num = 0
threshold_people_counter = 10


# 프로그램 시작하고 최소한 n초는 지나야 문자를 보냄
# threshold_time = 10: 최소한 프로그램 시작하고 10초는 지나야함
t1 = datetime.datetime.now()
threshold_time = 10 # 


# status msg sent
status_msg_sent = False # 메시지 보냈는지 판단 여부 


# 메세지 보낼 핸드폰 번호 및 메시지
phone_number_to = "+821045015852"
messages = "[동양미래대학교 팀] 현재 위성 데이터 분석결과 인구밀도가 높은 '위험' 지역을 발견했습니다. 조속히 대피 안내를 해주시기 바랍니다. "





#%%
# Loop through the video frames

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    

    if success:      
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # change names
        results[0].names = {0: '사람', 1:'살려', 2:'주세요'}
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()



        # cv2 현재 시각 
        text1=datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        org1=(0,40)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_frame,text1,org1,font,0.6,(255,0,0),2)
    

        # cv2 메시지 보냈는지 여부 확인
        text2='Msg_sent:   ' + str(status_msg_sent)
        org2=(0,60)        
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_frame,text2,org2,font,0.6,(255,0,0),2)        


        # cv2 danger count
        text3='msg_count(danger):   ' + str(danger_counter)
        org3=(0,80)        
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_frame,text3,org3,font,0.6,(255,0,0),2)   

        # cv2 danger count
        text4='meg_count(ppl):   ' + str(people_counter)
        org4=(0,100)        
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_frame,text4,org4,font,0.6,(255,0,0),2)   
        # Display the annotated frame
        
        # cv2 danger count
        text5='people count:   ' + str(results_people_num)
        org5=(0,140)        
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_frame,text5,org5,font,0.6,(255,0,0),2)   
        
        cv2.imshow("YOLOv8 Inference", annotated_frame)





        # 파일 및 로그 저장
        if img_num % freq == 0:
            # save displayed image
            file_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-4] + '.jpg'
            cv2.imwrite(path_save + file_name, annotated_frame)
            
            # 클래스만 뽑기
            results_cls = results[0].boxes.cls 
            results_cls = [results[0].names[k] for k in np.array(results_cls)]
            
            # 사람 수
            results_people_num = len(results_cls)
            
            # 각자 클래스의 confidence 뽑기
            results_conf = [ k for k in np.array(results[0].boxes.conf)]
            
            
            # danger class index
            danger_index = [i for i, x in enumerate(results_cls) if x in ['danger']]
            danger_conf = [results_conf[i] for i in danger_index]
            
            # write logs
            f.write(datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-4])
            f.write('\t cls: ' +  str(results_cls))
            f.write('\t conf: '+  str(results_conf))
            f.write('\n')
            
            
            
            
            
            #### danger count
            #### 이거 쓸려면 'danger' 클래스가 있어야 함 
            # try:
            #     # 최소한 danger_max가 threshold보다 높아야만 count로 인정
            #     danger_max = max(danger_conf)
            #     if danger_max > 0.7:
            #         t2 = datetime.datetime.now()            # current time
            #         diff = t2 - t1                          # current time - last msg sent
            #         seconds_diff = diff.total_seconds()     # diff in seconds
            #         danger_counter += 1
                
            #         print()
            #         print()
            #         print()
            #         print()
            #         print('seconds diff:', seconds_diff)
            #         print('danger count:', danger_counter)
            #         print('Danger!!:', str(round(danger_max,2)))                    
                    
                    
                    
            #         # 시간조건 충족하고, danger count 조건 충족시 문자 보냄 
            #         if (seconds_diff > threshold_time) & (danger_counter > threshold_danger_counter) & (status_msg_sent == False):
            #             print()
            #             print()
            #             print()
            #             print()
            #             print('**********')
            #             print('**********')
            #             print('Mesg Sent!!:', str(round(danger_max,2)))  
                        
            #             status_msg_sent = True
            #             send_sms(phone_number_to, messages)  # 문자 보냄
                        
            
            
            
            
            
            
                        
            #### 사람 수 counter
            try:                        
                if results_people_num > 6:
                    t2 = datetime.datetime.now()            # current time
                    diff = t2 - t1                          # current time - last msg sent
                    seconds_diff = diff.total_seconds()     # diff in seconds
                    people_counter += 1
                
                    print()
                    print()
                    print()
                    print()
                    print('seconds diff:', seconds_diff)
                    print('people_counter:', people_counter)
    

                    # 시간조건 충족하고, danger count 조건 충족시 문자 보냄 
                    if (seconds_diff > threshold_time) & (people_counter > threshold_people_counter) & (status_msg_sent == False):
                        print()
                        print()
                        print()
                        print()
                        print('**********')
                        print('**********')
                        
                        status_msg_sent = True
                        send_sms(phone_number_to, messages)  # 문자 보냄

              

        
            except:
                pass








        # 저장된 이미지 갯수 카운트
        img_num += 1


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    
    else:
        # Break the loop if the end of the video is reached
        break



#%%
# close everything
f.close()
cap.release()
k = cv2.waitKey(1)
cv2.destroyAllWindows()