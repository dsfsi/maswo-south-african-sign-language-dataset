import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mediapipe as mp
import os

def take_video_snapshot_at_timestamp(video_path, timestamp, output_path):
    """
    Take a snapshot from the video at a specific timestamp.

    Args:
        video_path (str): Path to the input video file.
        timestamp (float): Timestamp in seconds to capture.
        output_path (str): Where to save the snapshot.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    
    if success:
        cv2.imwrite(output_path, frame)
        print(f"Snapshot saved: {output_path}")
    else:
        raise ValueError(f"Failed to capture frame at {timestamp}s")
    
    cap.release()

def detect_interpreter_face_mediapipe(video_path, show_results=True, save_results=True, output_dir="output", confidence_threshold=0.5):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    timestamps = [35, 65]
    snapshot_dir = "snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    count = 1
    all_faces = []
    all_landmarks = []

    for timestamp in timestamps:
        snapshot_filename = os.path.join(snapshot_dir, f"snapshot_{count}.jpg")
        take_video_snapshot_at_timestamp(video_path, timestamp, snapshot_filename)

        image = cv2.imread(snapshot_filename)
        if image is None:
            raise ValueError(f"Could not load image from {snapshot_filename}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        bottom_start = height // 2
        right_start = width // 2
        interpreter_region = image_rgb[bottom_start:, right_start:]
        interpreter_region_height, interpreter_region_width = interpreter_region.shape[:2]

        faces = []
        face_landmarks = []

        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=confidence_threshold) as face_detection:
            results = face_detection.process(interpreter_region)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * interpreter_region_width)
                    y = int(bbox.ymin * interpreter_region_height)
                    w = int(bbox.width * interpreter_region_width)
                    h = int(bbox.height * interpreter_region_height)
                    adjusted_x = x + right_start
                    adjusted_y = y + bottom_start

                    faces.append((adjusted_x, adjusted_y, w, h))

                    confidence = detection.score[0]
                    keypoints = []
                    if detection.location_data.relative_keypoints:
                        for keypoint in detection.location_data.relative_keypoints:
                            kp_x = int(keypoint.x * interpreter_region_width) + right_start
                            kp_y = int(keypoint.y * interpreter_region_height) + bottom_start
                            keypoints.append((kp_x, kp_y))

                    face_landmarks.append({
                        'confidence': confidence,
                        'keypoints': keypoints,
                        'bbox': (adjusted_x, adjusted_y, w, h)
                    })

        all_faces.extend(faces)
        all_landmarks.extend(face_landmarks)

        if show_results:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image_rgb)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(interpreter_region)
            axes[1].set_title('Interpreter Region (Bottom Right)')
            axes[1].axis('off')

            for i, (x, y, w, h) in enumerate(faces):
                region_x = x - right_start
                region_y = y - bottom_start
                rect = Rectangle((region_x, region_y), w, h, linewidth=2, edgecolor='red', facecolor='none')
                axes[1].add_patch(rect)
                confidence = face_landmarks[i]['confidence']
                axes[1].text(region_x, region_y - 5, f'Conf: {confidence:.2f}', color='red', fontsize=8, weight='bold')

            axes[2].imshow(image_rgb)
            axes[2].set_title(f'Detected Faces: {len(faces)}')
            axes[2].axis('off')

            for i, (x, y, w, h) in enumerate(faces):
                rect = Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none')
                axes[2].add_patch(rect)
                for kp_x, kp_y in face_landmarks[i]['keypoints']:
                    axes[2].plot(kp_x, kp_y, 'go', markersize=4)
                confidence = face_landmarks[i]['confidence']
                axes[2].text(x, y - 10, f'Interpreter (Conf: {confidence:.2f})', color='red', fontsize=10, weight='bold')

            plt.tight_layout()
            if save_results:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                plt.savefig(os.path.join(output_dir, f"{base_name}_mediapipe_analysis_{count}.png"),
                            dpi=300, bbox_inches='tight')
                print(f"MediaPipe analysis saved to: {output_dir}/{base_name}_mediapipe_analysis_{count}.png")
            plt.show()

        if save_results:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            image_with_faces = image_rgb.copy()
            image_bgr_with_faces = cv2.cvtColor(image_with_faces, cv2.COLOR_RGB2BGR)

            for i, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(image_bgr_with_faces, (x, y), (x + w, y + h), (0, 0, 255), 3)
                for kp_x, kp_y in face_landmarks[i]['keypoints']:
                    cv2.circle(image_bgr_with_faces, (kp_x, kp_y), 3, (0, 255, 0), -1)
                confidence = face_landmarks[i]['confidence']
                label = f'Interpreter ({confidence:.2f})'
                cv2.putText(image_bgr_with_faces, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            output_path = os.path.join(output_dir, f"{base_name}_mediapipe_faces_{count}.jpg")
            cv2.imwrite(output_path, image_bgr_with_faces)
            print(f"Image with faces saved to: {output_path}")

            interpreter_with_faces = cv2.cvtColor(interpreter_region, cv2.COLOR_RGB2BGR)
            for i, (x, y, w, h) in enumerate(faces):
                region_x = x - right_start
                region_y = y - bottom_start
                cv2.rectangle(interpreter_with_faces, (region_x, region_y),
                              (region_x + w, region_y + h), (0, 0, 255), 2)
                for kp_x, kp_y in face_landmarks[i]['keypoints']:
                    region_kp_x = kp_x - right_start
                    region_kp_y = kp_y - bottom_start
                    cv2.circle(interpreter_with_faces, (region_kp_x, region_kp_y), 2, (0, 255, 0), -1)
                confidence = face_landmarks[i]['confidence']
                cv2.putText(interpreter_with_faces, f'Face ({confidence:.2f})',
                            (region_x, region_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            interpreter_output_path = os.path.join(output_dir, f"{base_name}_interpreter_region_{count}.jpg")
            cv2.imwrite(interpreter_output_path, interpreter_with_faces)
            print(f"Interpreter region saved to: {interpreter_output_path}")

            for i, (x, y, w, h) in enumerate(faces):
                padding_x = int(w * 1.8)
                padding_y_top = int(h * 0.5)
                padding_y_bottom = int(h * 3.5)

                x_start = max(0, x - padding_x)
                x_end = min(image_rgb.shape[1], x + w + padding_x)
                y_start = max(0, y - padding_y_top)
                y_end = min(image_rgb.shape[0], y + h + padding_y_bottom)

                interpreter_crop = image_rgb[y_start:y_end, x_start:x_end]
                interpreter_crop_bgr = cv2.cvtColor(interpreter_crop, cv2.COLOR_RGB2BGR)

                output_path = os.path.join(output_dir, f"{base_name}_interpreter_crop_{count}_{i+1}.jpg")
                cv2.imwrite(output_path, interpreter_crop_bgr)
                print(f"Interpreter crop {i+1} saved to: {output_path}")

        count += 1

    return len(all_faces), image_rgb, all_faces, all_landmarks

# === USAGE ===
if __name__ == "__main__":
    video_path = "videoplayback.webm"
    try:
        print("=== MediaPipe Face Detection ===")
        num_faces, processed_image, face_coords, face_data = detect_interpreter_face_mediapipe(
            video_path,
            show_results=True,
            save_results=True,
            output_dir="mediapipe_results",
            confidence_threshold=0.3
        )
        print(f"Detected {num_faces} face(s) in the interpreter region")
        print(f"Processed coordinates: {face_coords}")

        for i, face_info in enumerate(face_data):
            x, y, w, h = face_info['bbox']
            confidence = face_info['confidence']
            print(f"Face {i+1}: x={x}, y={y}, width={w}, height={h}, confidence={confidence:.3f}")
            print(f"  Keypoints detected: {len(face_info['keypoints'])}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure MediaPipe is installed: pip install mediapipe opencv-python matplotlib")