import os  # 운영 체제 관련 기능 제어 모듈
import numpy as np  # 수치 계산 모듈
import matplotlib.pyplot as plt  # 그래프 생성 모듈
import tensorflow as tf  # 딥러닝 모듈
from PIL import Image  # 이미지 처리 모듈



# 학습 및 검증 데이터 디렉토리 경로 설정
train_dir = r"C:\Users\happy\Desktop\학교\고등학교\2학년\고양이 종 구별 CNN\img\train"
validation_dir = r"C:\Users\happy\Desktop\학교\고등학교\2학년\고양이 종 구별 CNN\img\validation"


# 손상 이미지 제거 함수
def verify_images(directory):
    for root, _, files in os.walk(directory):  # 디렉토리 순회
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):  # 이미지 파일 확장자 확인
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)  # 이미지 파일 열기
                    img.verify()  # 이미지 파일 검증
                    img = Image.open(file_path)  # 이미지 파일 다시 열기
                    if img.mode == 'P' and 'transparency' in img.info:  # 투명한 팔레트 이미지를 RGBA 형식으로 변환
                        img = img.convert('RGBA')
                        img.save(file_path)
                except (IOError, SyntaxError, OSError) as e:  # 에러 발생시 파일 제거
                    print(f"Removing corrupted image: {file_path}")
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error encountered while processing {file_path}: {str(e)}")

# 손상된 이미지 제거
verify_images(train_dir)
verify_images(validation_dir)


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 손상된 이미지도 로드하도록 설정


# 데이터 증식 설정
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,  # 이미지 스케일 조정
    rotation_range=60,  # 이미지 회전 범위
    width_shift_range=0.2,  # 너비 이동 범위
    height_shift_range=0.2,  # 높이 이동 범위
    shear_range=0.2,  # 전단 변환 범위
    zoom_range=0.3,  # 확대(줌) 범위
    horizontal_flip=True,  # 수평 뒤집기 활성화
    vertical_flip=True,  # 수직 뒤집기 활성화
    brightness_range=(0.8, 1.2),  # 밝기 조정 범위
    fill_mode='nearest'  # 변환 시 빈 공간 채우기
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)  # 검증 데이터 스케일 조정


# 학습 데이터 생성기 설정
train_generator = train_datagen.flow_from_directory(
    train_dir,  # 학습 데이터 디렉토리 경로
    target_size=(200, 200),  # 이미지 크기 조정
    batch_size=64,  # 배치 크기
    class_mode='categorical'  # 분류 방식
)

# 검증 데이터 디렉토리 확인 및 생성기 설정
if not os.path.exists(validation_dir): # 검증데이터 존재 하지 않을 시
    print("Validation directory not found. Skipping validation.")
    validation_generator = None  # 검증 뛰어넘기
else:
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,  # 검증 데이터 디렉토리 경로
        target_size=(200, 200),  # 이미지 크기 조정
        batch_size=64,  # 배치 크기
        class_mode='categorical'  # 분류 방식
    )


# 클래스 수 확인
num_classes = len(train_generator.class_indices)


# 심층 합성곱 신경망 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),  # 합성곱 층
    tf.keras.layers.BatchNormalization(),  # 배치 정규화
    tf.keras.layers.MaxPooling2D(2, 2),  # 최대 풀링
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # 합성곱 층
    tf.keras.layers.BatchNormalization(),  # 배치 정규화
    tf.keras.layers.MaxPooling2D(2, 2),  # 최대 풀링
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # 합성곱 층
    tf.keras.layers.BatchNormalization(),  # 배치 정규화
    tf.keras.layers.MaxPooling2D(2, 2),  # 최대 풀링
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),  # 합성곱 층
    tf.keras.layers.BatchNormalization(),  # 배치 정규화
    tf.keras.layers.MaxPooling2D(2, 2),  # 최대 풀링
    tf.keras.layers.GlobalAveragePooling2D(),  # 전역 평균 풀링
    tf.keras.layers.Dense(512, activation='relu'),  # 완전 연결 층
    tf.keras.layers.Dropout(0.5),  # 드롭아웃
    tf.keras.layers.Dense(num_classes, activation='softmax')  # 출력 층
])


# 모델 컴파일
model.compile(
    loss='categorical_crossentropy',  # 손실 함수
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # 옵티마이저
    metrics=['accuracy']  # 평가 지표
)


# 조기 종료 콜백 설정
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True)

# 학습률 감소 콜백 설정
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)


# train_generator와 validation_generator를 tf.data.Dataset으로 변환
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,  # 생성기 함수
    output_signature=(
        tf.TensorSpec(shape=(None, 200, 200, 3), dtype=tf.float32),  # 입력 데이터 형태
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)  # 출력 데이터 형태
    )
).repeat()  # 데이터 반복

if validation_generator:
    validation_dataset = tf.data.Dataset.from_generator(
        lambda: validation_generator,  # 생성기 함수
        output_signature=(
            tf.TensorSpec(shape=(None, 200, 200, 3), dtype=tf.float32),  # 입력 데이터 형태
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)  # 출력 데이터 형태
        )
    ).repeat()  # 데이터 반복
else:
    validation_dataset = None  # 검증 데이터 없음


# 모델 학습
if validation_generator: # 검증 데이터 존재 시
    history = model.fit(
        train_dataset,  # 학습 데이터셋
        steps_per_epoch=train_generator.samples // train_generator.batch_size,  # 에포크 당 스텝 수
        epochs=50,  # 에포크 수
        validation_data=validation_dataset,  # 검증 데이터셋
        validation_steps=validation_generator.samples // validation_generator.batch_size,  # 검증 데이터 스텝 수
        callbacks=[early_stopping, reduce_lr]  # 콜백 리스트
    )
else: # 검증 데이터 미존재 시
    history = model.fit(
        train_dataset,  # 학습 데이터셋
        steps_per_epoch=train_generator.samples // train_generator.batch_size,  # 에포크 당 스텝 수
        epochs=50,  # 에포크 수
        callbacks=[early_stopping, reduce_lr]  # 콜백 리스트
    )


# 학습 결과 시각화
acc = history.history['accuracy']  # 학습 정확도
loss = history.history['loss']  # 학습 손실
epochs = range(len(acc))  # 에포크 범위


# 그래프 저장 경로 설정
save_dir = r"C:\Users\happy\Desktop\학교\고등학교\2학년\고양이 종 구별 CNN\res\6"

# 디렉토리가 없으면 생성
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# 정확도 그래프
plt.figure()
plt.plot(epochs, acc, 'bo', label='Training accuracy')  # 학습 정확도 그래프
if 'val_accuracy' in history.history:
    val_acc = history.history['val_accuracy']  # 검증 정확도
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')  # 검증 정확도 그래프
plt.title('Training and validation accuracy')  # 그래프 제목
plt.legend()
plt.savefig(os.path.join(save_dir, 'accuracy.png'))  # 그래프 저장
plt.close()  # 그래프 닫기

# 손실 그래프
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')  # 학습 손실 그래프
if 'val_loss' in history.history:
    val_loss = history.history['val_loss']  # 검증 손실
    plt.plot(epochs, val_loss, 'b', label='Validation loss')  # 검증 손실 그래프
plt.title('Training and validation loss')  # 그래프 제목
plt.legend()
plt.savefig(os.path.join(save_dir, 'loss.png'))  # 그래프 저장
plt.close()  # 그래프 닫기


# 고양이 종 분류 함수
def predict_cat_breed(model, img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(200, 200))  # 이미지 로드 및 크기 조정
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # 이미지를 배열로 변환
    
    # 투명한 팔레트 이미지를 RGBA로 변환
    if img.mode == 'P' and 'transparency' in img.info:
        img = img.convert('RGBA')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # 배열 확장 및 스케일 조정

    prediction = model.predict(img_array)  # 모델 예측
    class_idx = np.argmax(prediction, axis=1)[0]  # 가장 높은 예측 값의 인덱스
    class_labels = list(train_generator.class_indices.keys())  # 클래스 라벨
    return class_labels[class_idx]  # 예측된 클래스(종) 반환


# 테스트 이미지 디렉토리 경로 설정
test_dir = r"C:\Users\happy\Desktop\학교\고등학교\2학년\고양이 종 구별 CNN\img\test"

# 테스트 이미지 예측 및 출력
for i, filename in enumerate(os.listdir(test_dir)):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 이미지 파일 확장자 확인
        img_path = os.path.join(test_dir, filename)
        predicted_breed = predict_cat_breed(model, img_path)  # 고양이 종 예측
        print(f"Image {i + 1}: {filename} - Predicted breed: {predicted_breed}")  # 예측 결과 출력
