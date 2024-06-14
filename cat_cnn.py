import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

test_num = 4

# 손상된 이미지 파일을 제거하는 함수
def verify_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img.verify()  # 이미지 파일 검증
                    img = Image.open(file_path)  # Reopen to convert if necessary
                    if img.mode == 'P' and 'transparency' in img.info:
                        img = img.convert('RGBA')
                        img.save(file_path)
                except (IOError, SyntaxError, OSError) as e:
                    print(f"Removing corrupted image: {file_path}")
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error encountered while processing {file_path}: {str(e)}")

# 학습 및 검증 데이터 디렉토리 설정
train_dir = r"C:\Users\happy\Desktop\학교\고등학교\2학년\고양이 종 구별 CNN\img\train"
validation_dir = r"C:\Users\happy\Desktop\학교\고등학교\2학년\고양이 종 구별 CNN\img\validation"

# 손상된 이미지 파일 제거
verify_images(train_dir)
verify_images(validation_dir)

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 데이터 증식을 위한 설정
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=60,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest'
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),
    batch_size=64,
    class_mode='categorical'
)

# Validation data check
if not os.path.exists(validation_dir):
    print("Validation directory not found. Skipping validation.")
    validation_generator = None  # Set to None to skip validation
else:
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(200, 200),
        batch_size=64,
        class_mode='categorical'
    )

# 클래스 수 확인
num_classes = len(train_generator.class_indices)

# 심층 합성곱 신경망 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # 학습률 조정
    metrics=['accuracy']
)

# 조기 종료 콜백 설정
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# train_generator와 validation_generator를 tf.data.Dataset으로 변환
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 200, 200, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
    )
).repeat()

if validation_generator:
    validation_dataset = tf.data.Dataset.from_generator(
        lambda: validation_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 200, 200, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )
    ).repeat()
else:
    validation_dataset = None

# 모델 학습
if validation_generator:
    history = model.fit(
        train_dataset,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=65,  # 에포크 수 조정
        validation_data=validation_dataset,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[early_stopping]  # 조기 종료 콜백 추가
    )
else:
    history = model.fit(
        train_dataset,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=65,  # 에포크 수 조정
        callbacks=[early_stopping]  # 조기 종료 콜백 추가
    )

# 학습 결과 시각화
acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(len(acc))

# 그래프 저장 경로 설정
save_dir = r"C:\Users\happy\Desktop\학교\고등학교\2학년\고양이 종 구별 CNN\graphs\4"

# 디렉토리가 없으면 생성
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 정확도 그래프
plt.figure()
plt.plot(epochs, acc, 'bo', label='Training accuracy')
if 'val_accuracy' in history.history:
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(os.path.join(save_dir, 'accuracy.png'))  # 그래프 저장
plt.close()  # 그래프 닫기

# 손실 그래프
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
if 'val_loss' in history.history:
    val_loss = history.history['val_loss']
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(os.path.join(save_dir, 'loss.png'))  # 그래프 저장
plt.close()  # 그래프 닫기

# 고양이 종 분류 함수 정의
def predict_cat_breed(model, img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(200, 200))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Convert palette images with transparency to RGBA
    if img.mode == 'P' and 'transparency' in img.info:
        img = img.convert('RGBA')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    class_labels = list(train_generator.class_indices.keys())
    return class_labels[class_idx]


fi = open(r"C:\Users\happy\Desktop\학교\고등학교\2학년\고양이 종 구별 CNN\res\4" , 'a')
test_dir = r"C:\Users\happy\Desktop\학교\고등학교\2학년\고양이 종 구별 CNN\img\test"

for i, filename in enumerate(os.listdir(test_dir)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(test_dir, filename)
        predicted_breed = predict_cat_breed(model, img_path)
        print(f"Image {i + 1}: {filename} - Predicted breed: {predicted_breed}")
        fi.write(f"Image {i + 1}: {filename} - Predicted breed: {predicted_breed}")
fi.close()
