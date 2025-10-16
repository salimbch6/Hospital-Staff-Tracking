from flask import Flask, request, send_file, render_template
from ultralytics import YOLO
import os
import uuid
import cv2
import subprocess
import threading
import time

app = Flask(__name__)

# Folder setup
UPLOAD_FOLDER = 'static/videos/input'
OUTPUT_FOLDER = 'static/videos/output'
WATCH_FOLDER = UPLOAD_FOLDER
TEMP_FOLDER = 'static/videos/temp'

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Load YOLO models
det_model = YOLO("yolov8n.pt")
cls_model = YOLO("yolov8n-cls.pt")
det_model.overrides['tracker'] = 'botsort.yaml'

# Check if file is ready
def is_file_ready(file_path):
    initial_size = os.path.getsize(file_path)
    time.sleep(1)
    return initial_size == os.path.getsize(file_path)

# Convert video to safe format
def reencode_video_safe(input_path):
    safe_path = os.path.join(TEMP_FOLDER, os.path.basename(input_path).replace('.mp4', '_safe.mp4'))
    subprocess.run([
        'ffmpeg', '-y', '-i', input_path,
        '-vcodec', 'libx264', '-acodec', 'aac',
        '-preset', 'ultrafast',
        '-movflags', '+faststart', safe_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return safe_path

# Convert to browser-compatible format
def convert_to_browser_compatible(input_path, output_path):
    subprocess.run([
        'ffmpeg', '-y', '-i', input_path,
        '-vcodec', 'libx264', '-acodec', 'aac',
        '-movflags', '+faststart', output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Process video
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {input_path}")
        return

    width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    id_to_class, staff_ids, non_staff_ids = {}, set(), set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        results = det_model.track(frame.copy(), persist=True, classes=[0])
        boxes = results[0].boxes

        if boxes.id is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                conf = boxes.conf[i].item()
                track_id = int(boxes.id[i].item())

                if track_id not in id_to_class:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (224, 224))
                    pred_class = cls_model([crop])[0].probs.top1
                    class_name = cls_model.names[pred_class]
                    id_to_class[track_id] = class_name
                    (staff_ids if class_name == "staff" else non_staff_ids).add(track_id)

                color = (0, 255, 0) if id_to_class[track_id] == "staff" else (0, 0, 255)
                label = f"{track_id}: {id_to_class[track_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"STAFF: {len(staff_ids)} | NON-STAFF: {len(non_staff_ids)}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        out.write(frame)

    cap.release()
    out.release()
    convert_to_browser_compatible(temp_output, output_path)
    os.remove(temp_output)

# Background folder watcher
def background_folder_watcher():
    print(f"🔎 Watching folder: {WATCH_FOLDER}")
    processed = set()
    while True:
        for file in os.listdir(WATCH_FOLDER):
            if not file.lower().endswith(('.mp4', '.mov')):
                continue
            path = os.path.join(WATCH_FOLDER, file)
            if path not in processed and is_file_ready(path):
                print(f"🚀 New video detected and ready: {file}")
                processed.add(path)
                try:
                    safe_input = reencode_video_safe(path)
                    output_path = os.path.join(OUTPUT_FOLDER, f"{uuid.uuid4()}_output.mp4")
                    process_video(safe_input, output_path)
                    os.remove(path)
                    os.remove(safe_input)
                    print(f"✅ Done and cleaned: {file}")
                except Exception as e:
                    print(f"❌ Error: {e}")
        time.sleep(5)

# Unified index and upload route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template("index.html", error="No video uploaded.", videos=os.listdir(OUTPUT_FOLDER))

        file = request.files['video']
        video_id = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_FOLDER, f"{video_id}.mp4")
        output_path = os.path.join(OUTPUT_FOLDER, f"{video_id}_output.mp4")
        file.save(input_path)

        try:
            safe_path = reencode_video_safe(input_path)
            process_video(safe_path, output_path)
            os.remove(input_path)
            os.remove(safe_path)
        except Exception as e:
            return render_template("index.html", error=str(e), videos=os.listdir(OUTPUT_FOLDER))

        return render_template("index.html", message="Video processed!", videos=os.listdir(OUTPUT_FOLDER))

    return render_template("index.html", videos=os.listdir(OUTPUT_FOLDER))

@app.route('/download/<video_id>')
def download_video(video_id):
    path = os.path.join(OUTPUT_FOLDER, f"{video_id}_output.mp4")
    return send_file(path, as_attachment=True)

@app.after_request
def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

if __name__ == '__main__':
    threading.Thread(target=background_folder_watcher, daemon=True).start()
    app.run(debug=True, use_reloader=False)
