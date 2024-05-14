import numpy as np
import cv2
import keras
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
from moviepy.editor import *
from moviepy.video.tools.segmenting import findObjects
from moviepy.video.tools.subtitles import SubtitlesClip
import pyttsx3 
import moviepy.video.fx.all as vfx

class TexttoSpeech:
    def __init__(self):
        # Initialize the engine
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')       #getting details of current voice
        self.engine.setProperty('voice', voices[0].id)

    def text_to_speech(self,message):
        self.engine.say(message) 
        self.engine.runAndWait()
    
    def text_to_mp3(self,message,mp3file):
        self.engine.save_to_file(message, mp3file)
        self.engine.runAndWait()

        
ts = TexttoSpeech()

def high_pass_filter(image):
    # 將影格轉換為 float32 格式
    image_float = tf.cast(image, tf.float32)

    # 將影格轉換為灰度
    grayscale = tf.image.rgb_to_grayscale(image_float)

    # 套用高通濾波器
    kernel = tf.constant([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=tf.float32)
    filtered = tf.nn.conv2d(tf.expand_dims(grayscale, axis=0), 
                             tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1), 
                             strides=[1, 1, 1, 1], 
                             padding='SAME')

    # 將影格的範圍限制在 [0, 255] 之間
    filtered = tf.clip_by_value(filtered, 0.0, 255.0)

    return tf.squeeze(filtered, axis=0)

cap1 = cv2.VideoCapture("D:\大學\大三\機器視覺\HW1_source.mp4")
cap2 = cv2.VideoCapture("D:\大學\大三\機器視覺\HW1_source2.mp4")
width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
width2 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
fps1=int(cap1.get(cv2.CAP_PROP_FPS))
fps2=int(cap2.get(cv2.CAP_PROP_FPS))
outputs1_1 = cv2.VideoWriter('outputs1_1.mp4', fourcc, fps1, (width1, height1))
outputs1_2 = cv2.VideoWriter('outputs1_2.mp4', fourcc, fps1, (width1, height1))
outputs1_3 = cv2.VideoWriter('outputs1_3.mp4', fourcc, fps1, (width1, height1))
outputs1_4 = cv2.VideoWriter('outputs1_4.mp4', fourcc, fps1, (width1, height1))
outputs1_5 = cv2.VideoWriter('outputs1_5.mp4', fourcc, fps1, (width1, height1))

outputs2_1 = cv2.VideoWriter('outputs2_1.mp4', fourcc, fps1, (width2, height2))
outputs2_2 = cv2.VideoWriter('outputs2_2.mp4', fourcc, fps1, (width2, height2))
outputs2_3 = cv2.VideoWriter('outputs2_3.mp4', fourcc, fps1, (width2, height2))
outputs2_4 = cv2.VideoWriter('outputs2_4.mp4', fourcc, fps1, (width2, height2))
outputs2_5 = cv2.VideoWriter('outputs2_5.mp4', fourcc, fps1, (width2, height2))

if not cap1.isOpened() or not cap2.isOpened():
    print('Fail')
    exit()

while(1):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if (not ret1) or (not ret2):
        print('did not get frame')
        break
    
    #source1
    grayscale1 = tf.image.rgb_to_grayscale(frame1).numpy()#灰階
    grayscale1 = cv2.putText(grayscale1, 'apply grayscale', (10,50), cv2.FONT_HERSHEY_SIMPLEX,  2, (255,255,255), 2, cv2.LINE_AA) 

    image1 = tf.cast(frame1, tf.float32)#hsv
    normal1 = image1 / 255.
    hsv1 = tf.image.rgb_to_hsv(normal1).numpy()
    hsv1 = hsv1 * 255.0
    hsv1 = cv2.putText(hsv1, 'convert rgb to hsv', (10,50), cv2.FONT_HERSHEY_SIMPLEX,  2, (255,255,255), 2, cv2.LINE_AA) 

    filtered_frame1 = high_pass_filter(frame1)#high pass filter 
    filtered_frame_np1 = filtered_frame1.numpy()
    filtered_frame_np1 = cv2.putText(filtered_frame_np1, 'apply high pass filter', (10,50), cv2.FONT_HERSHEY_SIMPLEX,  2, (255,255,255), 2, cv2.LINE_AA) 

    img_yuv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
    img_yuv1[:,:,0] = cv2.equalizeHist(img_yuv1[:,:,0])
    equalized_bgr1 = cv2.cvtColor(img_yuv1, cv2.COLOR_YUV2BGR)
    equalized_bgr1 = cv2.putText(equalized_bgr1, 'apply histogram equalization', (10,50), cv2.FONT_HERSHEY_SIMPLEX,  2, (255,255,255), 2, cv2.LINE_AA) 

    org1=frame1
    org1 = cv2.putText(org1, 'source1', (10,50), cv2.FONT_HERSHEY_SIMPLEX,  2, (255,255,255), 2, cv2.LINE_AA)

    #source2   
    grayscale2 = tf.image.rgb_to_grayscale(frame2).numpy()
    grayscale2 = cv2.putText(grayscale2, 'apply grayscale', (10,50), cv2.FONT_HERSHEY_SIMPLEX,  2, (255,255,255), 2, cv2.LINE_AA) 

    image2 = tf.cast(frame2, tf.float32)
    normal2 = image2 / 255.
    hsv2 = tf.image.rgb_to_hsv(normal2).numpy()
    hsv2 = hsv2 * 255.0
    hsv2 = cv2.putText(hsv2, 'convert rgb to hsv', (10,50), cv2.FONT_HERSHEY_SIMPLEX,  2, (255,255,255), 2, cv2.LINE_AA) 

    filtered_frame2 = high_pass_filter(frame2)
    filtered_frame_np2 = filtered_frame2.numpy()
    filtered_frame_np2 = cv2.putText(filtered_frame_np2, 'apply high pass filter', (10,50), cv2.FONT_HERSHEY_SIMPLEX,  2, (255,255,255), 2, cv2.LINE_AA) 

    img_yuv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)
    img_yuv2[:,:,0] = cv2.equalizeHist(img_yuv2[:,:,0])
    equalized_bgr2 = cv2.cvtColor(img_yuv2, cv2.COLOR_YUV2BGR)
    equalized_bgr2 = cv2.putText(equalized_bgr2, 'apply histogram equalization', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA) 

    org2=frame2
    org2 = cv2.putText(org2, 'source2', (10,50), cv2.FONT_HERSHEY_SIMPLEX,  2, (255,255,255), 2, cv2.LINE_AA)

    outputs1_1.write(org1)
    outputs1_2.write(grayscale1)
    outputs1_3.write(cv2.cvtColor(hsv1.astype(np.uint8), cv2.COLOR_HSV2BGR))
    outputs1_4.write(filtered_frame_np1.astype('uint8'))
    outputs1_5.write(equalized_bgr1)

    outputs2_1.write(org2)
    outputs2_2.write(grayscale2)
    outputs2_3.write(cv2.cvtColor(hsv2.astype(np.uint8), cv2.COLOR_HSV2BGR))
    outputs2_4.write(filtered_frame_np2.astype('uint8'))
    outputs2_5.write(equalized_bgr2)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap1.release()
cap2.release()
outputs1_1.release()
outputs1_2.release()
outputs1_3.release()
outputs1_4.release()
outputs1_5.release()

outputs2_1.release()
outputs2_2.release()
outputs2_3.release()
outputs2_4.release()
outputs2_5.release()

cv2.destroyAllWindows()

os1_1 = VideoFileClip("outputs1_1.mp4") 
os1_2 = VideoFileClip("outputs1_2.mp4")                    # 開啟第一段影片
os1_3 = VideoFileClip("outputs1_3.mp4")  
os1_4 = VideoFileClip("outputs1_4.mp4")  
os1_5 = VideoFileClip("outputs1_5.mp4")  
vs1_1 = os1_1.resize((960,720)).margin(10)    # 改變尺寸，增加邊界
vs1_2 = os1_2.resize((960,720)).margin(10)    # 改變尺寸，增加邊界
vs1_3 = os1_3.resize((960,720)).margin(10)    # 改變尺寸，增加邊界
vs1_4 = os1_4.resize((960,720)).margin(10)    # 改變尺寸，增加邊界
vs1_5 = os1_5.resize((960,720)).margin(10)    # 改變尺寸，增加邊界

os2_1 = VideoFileClip("outputs2_1.mp4") 
os2_2 = VideoFileClip("outputs2_2.mp4")                    # 開啟第一段影片
os2_3 = VideoFileClip("outputs2_3.mp4")  
os2_4 = VideoFileClip("outputs2_4.mp4")  
os2_5 = VideoFileClip("outputs2_5.mp4")  
vs2_1 = os2_1.resize((960,720)).margin(10)    # 改變尺寸，增加邊界
vs2_2 = os2_2.resize((960,720)).margin(10)    # 改變尺寸，增加邊界
vs2_3 = os2_3.resize((960,720)).margin(10)    # 改變尺寸，增加邊界
vs2_4 = os2_4.resize((960,720)).margin(10)    # 改變尺寸，增加邊界
vs2_5 = os2_5.resize((960,720)).margin(10)    # 改變尺寸，增加邊界

output = clips_array([[vs1_1,vs1_2,vs1_3,vs1_4,vs1_5],[vs2_1,vs2_2,vs2_3,vs2_4,vs2_5]])
output.write_videofile("final.mp4",temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
print('ok')

subtitles = '''這個作業的兩個原始影片分別為"聖稜-雪山的脊樑©"和"《看見台灣III》預告片"，
排列方式由左至右為原始影片 -> 灰階 -> hsv -> high pass filter -> histogram equalization。
除了histogram equalization全部都用OpenCV，其他三種變化都是透過OpenCV讀影片，tensorflow改變色彩坐標系，再透過OpenCV輸出影片。'''
source_video_filename     = "final.mp4" 
background_music_filename = "calm-background-for-video-121519.mp3" # google this mp3 and download it by yourself
target_video_with_subtitle= "homework_1_final.mp4"
lines = [msg for msg in subtitles.split('\n') if len(msg)>0]
speech= []

#念每一句旁白。
for i,msg in enumerate(lines):
    ts.text_to_mp3(msg,'subtitle-voiceover-{:04d}.mp3'.format(i))    
    speech.append(AudioFileClip('subtitle-voiceover-{:04d}.mp3'.format(i)))
    
#計算每一句旁白開始與結束時間，假設開始時間為0。    
duration       = np.array([0]+[s.duration for s in speech])   
cumduration    = np.cumsum(duration)
total_duration = int(cumduration[-1])+4    

print(total_duration)  

#產生旁白字幕，注意msjh.ttc字型檔要在這個程式所在目錄。
generator = lambda txt: TextClip(txt, font='msjh.ttc', fontsize=64, color='white')
subtitles = SubtitlesClip([((cumduration[i],cumduration[i+1]),s) for i,s in enumerate(lines)], generator)

#調整目標視訊播放速度，使得目標視訊播放時間比念完全部旁白長一點。
clip = VideoFileClip(source_video_filename)
clip = clip.fx(vfx.speedx,clip.duration/total_duration)

#產生有字幕的目標視訊。
final_clip = CompositeVideoClip([clip, subtitles.set_pos(('center','bottom'))])

#背景音樂，只擷取目標視訊長度片段，並將音量調小。
background_music = AudioFileClip(background_music_filename)
baudio1 = background_music.subclip(background_music.duration-total_duration).volumex(0.2)

#將目標視訊的音訊設為混合來源視訊音訊、背景音樂、旁白音訊的音訊。
final_clip = final_clip.set_audio(CompositeAudioClip([baudio1,concatenate_audioclips(speech)]))

#輸出至目標視訊檔。
final_clip.write_videofile(target_video_with_subtitle)