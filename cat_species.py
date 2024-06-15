import os  # 운영 체제 관련 기능 제어 모듈
import numpy as np  # 수치 계산 모듈
import tensorflow as tf  # 딥러닝 모듈
from PIL import Image  # 이미지 처리 모듈

# 고양이 종 분류 함수 정의
def predict_cat_breed(model, img_path, class_labels):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(200, 200))  # 이미지 로드 및 크기 조정
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # 이미지를 배열로 변환
    
    # 투명한 팔레트 이미지를 RGBA로 변환
    if img.mode == 'P' and 'transparency' in img.info:
        img = img.convert('RGBA')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # 배열 확장 및 스케일 조정

    prediction = model.predict(img_array)  # 모델 예측
    class_idx = np.argmax(prediction, axis=1)[0]  # 가장 높은 예측 값의 인덱스
    return class_labels[class_idx]  # 예측된 클래스 반환

# 모델 경로 및 테스트 이미지 디렉토리 경로 설정
model_save_path = r"C:\Users\happy\Desktop\학교\고등학교\2학년\고양이 종 구별 CNN\model\cat_breed_classifier_model_7.h5"
test_dir = r"C:\Users\happy\Desktop\학교\고등학교\2학년\고양이 종 구별 CNN\img\test"

# 모델 로드
model = tf.keras.models.load_model(model_save_path)  # 저장된 모델 로드

# 클래스 라벨 설정
class_labels = class_labels = [
    'American_Shorthair',
    'Bengal',
    'British_Shorthair',
    'Korean_Shorthair',
    'Maine_Coon',
    'Persian',
    'Ragdoll',
    'Russian_Blue',
    'Scottish_Fold',
    'Siamese'
]

# 테스트 이미지 예측 및 출력
for i, filename in enumerate(os.listdir(test_dir)):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 이미지 파일 확장자 확인
        img_path = os.path.join(test_dir, filename)
        predicted_breed = predict_cat_breed(model, img_path, class_labels)  # 고양이 종 예측
        print(f"Image {i + 1}: {filename} - Predicted breed: {predicted_breed}")  # 예측 결과 출력
