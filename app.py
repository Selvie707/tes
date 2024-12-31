from flask import Flask, jsonify, Response
import cv2
from ultralytics import YOLO
import supervision as sv
import logging
from flask_cors import CORS  # Mengimpor Flask-CORS

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Mengizinkan semua origin untuk mengakses API ini
CORS(app)  # Mengaktifkan CORS untuk semua domain dan semua route

logging.basicConfig(level=logging.DEBUG)

# Load YOLO model
model = YOLO('C:\\Users\\USER\\Downloads\\tes\\besty.pt')  # Ganti dengan path model Anda
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Video capture
cap = cv2.VideoCapture(0)  # Kamera default

# Global variable to store the latest frame
latest_frame = None

def generate_frames():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Simpan frame terbaru
        latest_frame = frame.copy()

        # Deteksi objek dengan YOLO
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter detections dengan confidence > 50%
        filtered_detections = sv.Detections(
            xyxy=detections.xyxy[detections.confidence > 0.5],
            confidence=detections.confidence[detections.confidence > 0.5],
            class_id=detections.class_id[detections.confidence > 0.5]
        )

        # Annotasi hasil deteksi pada frame
        annotated_image = bounding_box_annotator.annotate(scene=frame, detections=filtered_detections)

        # Dapatkan nama kelas berdasarkan class_id yang telah difilter
        filtered_class_names = [results.names[class_id] for class_id in filtered_detections.class_id]

        # Menambahkan label untuk setiap deteksi
        for i, class_name in enumerate(filtered_class_names):
            label = f"{class_name}: {filtered_detections.confidence[i]*100:.2f}%"
            cv2.putText(annotated_image, label, 
                        (int(filtered_detections.xyxy[i][0]), int(filtered_detections.xyxy[i][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Konversi frame menjadi format JPEG dan kirimkan ke frontend
        ret, buffer = cv2.imencode('.jpg', annotated_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/')
def index():
    return "Hello, World!"

@app.route('/detections')
def detections():
    # Ambil satu frame dari kamera
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Unable to capture frame"}), 500

    # Deteksi objek dengan YOLO
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Jika tidak ada deteksi, kembalikan error
    if not detections or len(detections.confidence) == 0:
        return jsonify([])  # Kosongkan data jika tidak ada deteksi

    # Siapkan data deteksi (hanya class_name dan confidence)
    detections_data = []
    for class_id, confidence in zip(detections.class_id, detections.confidence):
        detections_data.append({
            "class": results.names[class_id],  # Ambil nama kelas berdasarkan ID
            "confidence": round(float(confidence), 2)  # Bulatkan confidence
        })

    return jsonify(detections_data)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
