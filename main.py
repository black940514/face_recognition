import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from yolov5facedetector.face_detector import YoloDetector

from model.model import ImageTransformerClassifier  

hyper_params = {
    'num_epochs': 20,
    'lr': 0.0001,
    'image_size': 224,
    'train_batch_size': 32,
    'val_batch_size': 16,
    'print_preq': 0.1 
}

weight_path = 'weight_pth/best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ImageTransformerClassifier(num_images=5, num_classes=8)
model.load_state_dict(torch.load(weight_path, map_location='cpu'))
model.to(device)
model.eval()

face_model = YoloDetector(target_size=512, gpu=0, min_face=48)

class_list = ["angry", "contempt", "dislike", "fear", "happy", "neutral", "sad", "surprise"]

val_transform = A.Compose([
    A.LongestMaxSize(max_size=hyper_params['image_size'], always_apply=True),
    A.PadIfNeeded(min_height=hyper_params['image_size'],
                  min_width=hyper_params['image_size'],
                  always_apply=True,
                  value=0,
                  border_mode=0),
    A.Normalize(p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  
    ToTensorV2()
])

st.title("김태연의 감정 인식 웹 서비스")
st.write("이미지를 업로드하면 얼굴을 탐지하여 감정을 예측합니다.")

uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    resized_image = image.resize((150, int(150 * image.height / image.width))) 

    image_array = np.array(image)

    bboxes, confs, points = face_model.predict(image_array)

    if len(bboxes) == 0:
        st.write("얼굴을 찾을 수 없습니다.")
    else:
        face_frame_list = []
        for box in bboxes[0]: 
            cropped_face = Image.fromarray(image_array).crop(box)
            face_frame_list.append(cropped_face)

        for face_idx, face in enumerate(face_frame_list):
            face_frame_list[face_idx] = val_transform(image=np.array(face))['image']

        face_tensor = torch.stack(face_frame_list).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face_tensor)
            preds = F.softmax(output, dim=-1).cpu().numpy() 

        cols = st.columns(3)  
        with cols[0]:
            st.markdown("**업로드된 이미지**")
            st.image(resized_image, caption="업로드된 이미지", width=150)

        with cols[1]:
            st.markdown("**탐지된 얼굴**")
            st.image(cropped_face, caption="탐지된 얼굴", width=100)

        with cols[2]:
            st.markdown("**감정 예측 결과**")
            for idx, face_preds in enumerate(preds):
                max_idx = np.argmax(face_preds)  
                
                st.success(f"최고 감정: {class_list[max_idx]} ({face_preds[max_idx] * 100:.2f}%)")
                
                for pred, cls in zip(face_preds, class_list):
                    st.write(f"- {cls}: {pred * 100:.2f}%")