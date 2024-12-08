import cv2  
def detect_faces(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )  # 얼굴 탐지
    return faces

def preprocess_face(img, box, mean_values):
    x, y, w, h = box
    face = img[y:y+h, x:x+w].copy()  # 얼굴 영역 추출
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), mean_values, swapRB=False)
    return blob

def load_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return age_net, gender_net

def predict_age_gender(blob, age_net, gender_net, age_list, gender_list):
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds.argmax()]  # 성별 예측

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds.argmax()]  # 나이 예측

    return f"{gender} {age}"

def video_detector(cam, cascade, age_net, gender_net, mean_values, age_list, gender_list):
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        faces = detect_faces(frame, cascade)  # 얼굴 탐지
        for box in faces:
            blob = preprocess_face(frame, box, mean_values)  # 얼굴 전처리
            info = predict_age_gender(blob, age_net, gender_net, age_list, gender_list)  # 성별/나이 예측

            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), thickness=2)
            cv2.putText(frame, info, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)

        cv2.imshow('Video Face Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
            break

def image_detector(img_path, cascade, age_net, gender_net, mean_values, age_list, gender_list):
    img = cv2.imread(img_path)
    faces = detect_faces(img, cascade)  # 얼굴 탐지

    for box in faces:
        blob = preprocess_face(img, box, mean_values)  # 얼굴 전처리
        info = predict_age_gender(blob, age_net, gender_net, age_list, gender_list)  # 성별/나이 예측

        x, y, w, h = box
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), thickness=2)
        cv2.putText(img, info,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX ,0.5,(0 ,255 ,0),thickness=1)

    cv2.imshow('Image Face Detector', img)
    cv2.waitKey(10000)

if __name__ == "__main__":
    cascade_file = 'haarcascade_frontalface_alt.xml'
    cascade = cv2.CascadeClassifier(cascade_file)

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    
    age_categories = ['(0 ~ 2)', '(4 ~ 6)', '(8 ~ 12)', '(15 ~ 20)',
                      '(25 ~ 32)', '(38 ~ 43)', '(48 ~ 53)', '(60 ~ 100)']
    gender_categories = ['Male', 'Female']

    age_model, gender_model = load_models()

    # 영상 파일 처리
    video_file = 'sample.mp4'
    cam = cv2.VideoCapture(video_file)
    video_detector(cam,cascade,
                   age_model,
                   gender_model,
                   MODEL_MEAN_VALUES,
                   age_categories,
                   gender_categories)

    # 이미지 파일 처리
    image_file = 'sample.jpg'
    image_detector(image_file,
                   cascade,
                   age_model,
                   gender_model,
                   MODEL_MEAN_VALUES,
                   age_categories,
                   gender_categories)
