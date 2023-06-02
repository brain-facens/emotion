import cv2
import time
import argparse

import torch

import rospy
from std_msgs.msg import Header
from emotion.msg import Emotion


def main():
    camip = '0'

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-ci",
        "--camip",
        help=" Set camera acces. Default : 0",
    )

    args = parser.parse_args()

    if args.camip:
        camip = args.camip
        
    model = torch.hub.load('.', 'custom', path='weights_pt/best.pt', source='local') 

    classses = ['neutral','sad','fear','anger','happy']

    # define a video capture object
    cap = cv2.VideoCapture(camip if camip != '0' else int(camip))
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            return
        det = model(frame).xyxy[0].cpu().numpy().tolist()
        tam = len(det)
        try:                    
            qtd_faces = len(det)
            bb = [int(det[0][0]),int(det[0][1]),int(det[0][2]),int(det[0][3])]
            emotion = classses[int(det[0][-1])]
            
            pubROS(emotion,int(qtd_faces),bb)
        except:
            pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    if rospy.is_shutdown():
        cap.release()     

        
    # Destroy all the windows

def pubROS(face, quant, bb):
    pub = rospy.Publisher('/vision/emotion', Emotion, queue_size = 1)

    rospy.init_node('vision_emotion', anonymous = True)
    rate = rospy.Rate(10)
    
    msg                 = Emotion()
    msg.header          = Header(stamp = rospy.Time.now(), frame_id = 'odom')
    msg.qtd_faces       = int(quant)
    msg.emotion         = str(face)
    msg.bounding_box    = str(bb)
    
    
    pub.publish(msg)

def handlle_crash(func):
    while True:
        print("Restartiing ...")
        time.sleep(2)
        func()
    

if __name__ == "__main__":
    try:
        main()

    except rospy.ROSInterruptException:
        print("Shutting down")
        