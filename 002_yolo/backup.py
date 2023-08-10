import ultralytics
from ultralytics import YOLO
import cv2


  
  
#%%
# 모델 불러오기
model = YOLO('./runs/detect/train/weights/best.pt') 


# 모델 학습
# model.train(data='data.yaml', epochs=30, patience=30, batch=32)




#%% image로 테스트 
# results = model.predict(source='.//test//images//', save=True)
# print(type(results), len(results))



#%% webcam으로 테스트
results = model.predict(source="0", show=True, save=True)






#%% video
video_path = "./video//test.mp4"
cap = cv2.VideoCapture(video_path)





# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    print(success)
    print(frame.shape)

    if success:
        print('test')
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)





      
        # try:
        #     results_cls = results[0].boxes.cls 
        #     results_conf = results[0].boxes.conf
            
        #     print('results_cls', results_cls)
        #     print()

        #     print('results_conf', results_conf)
        #     print()
        # except:
        #     pass





        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    
    else:
        # Break the loop if the end of the video is reached
        break


cap.release()
k = cv2.waitKey(1)
cv2.destroyAllWindows()