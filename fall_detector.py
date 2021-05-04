from pose_estimation import PoseEstimation
import cv2
from args import get_args, show_args
import numpy as np
import pickle
from collections import deque
import time
import telegram_send
import threading


class FallDetector:

    def __init__(self, args):
        # Get args
        self.frame_rate = args.frame_rate
        self.telegram_alert = args.telegram_alert
        self.chunk_seconds = args.chunk_seconds
        self.video_name = "temp/state"

        show_args(args)

        # Load human state classifier model
        self.model = pickle.load(open(args.fall_model_path, 'rb'))
        with open(args.path_model_header) as f:
            self.model_header = f.readline().split(",")

    def estimate_pose(self, camera_url, camera_name="Unknow"):
        pe = PoseEstimation()
        camera = cv2.VideoCapture(camera_url)
        video_rate = int(np.round(camera.get(cv2.CAP_PROP_FPS) / self.frame_rate))
        total_frames = self.frame_rate * self.chunk_seconds
        print("input frames per seconds", camera.get(cv2.CAP_PROP_FPS))
        state = "Nothing"
        # I use deque because I want to delete de first element when other one is added
        video_keypoints = deque(maxlen=total_frames)
        video_to_send = deque(maxlen=total_frames)

        times = []
        frame_number = -1
        while True:
            ret, frame = camera.read()
            if not ret:
                print("No image could be obtained from the camera")
                break
            frame_number = (frame_number + 1) % video_rate

            if frame_number == 0:

                # Resize img
                frame = self.make_square(frame, pe.expected_pixels)

                # Save the frame
                video_to_send.append(np.copy(frame))

                # Get body parts
                start = time.time()
                pose = pe.get_pose_estimation(frame)
                end = time.time()

                # Measuring time
                total_time = end - start
                times.append(total_time)

                # Normalizing position
                # average_x = np.mean(pose[:, 0])
                # average_y = np.mean(pose[:, 1])
                # pose[:, 0] = pose[:, 0] - average_x
                # pose[:, 1] = pose[:, 1] - average_y

                # Normalizing scale
                max_val = np.abs(np.max(pose))
                pose[:] = pose[:] / max_val

                video_keypoints.append(np.reshape(pose, -1))

                if len(video_keypoints) == total_frames:
                    # video_to_send.release()
                    state = self.report_state(np.reshape(video_keypoints, (1, -1)), np.copy(video_to_send),
                                              camera_name, pe.expected_pixels)

                    # Clear queue
                    video_keypoints.clear()
                    video_to_send.clear()

                self.show_results(frame, camera_name, state)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        mean = np.mean(times)
        camera.release()
        cv2.destroyAllWindows()

    def report_state(self, video_keypoints, video_to_send, camera_name, video_size):
        state = str(self.model.predict(video_keypoints)[0])
        if (state == "Fall" or state == "Recover") and self.telegram_alert:
            telegram_t = threading.Thread(target=self.report,
                                          args=(video_to_send, video_size, state, camera_name))
            telegram_t.start()
        return state

    def report(self, video_to_send, video_size, state, camera_name=None):
        file_video_name = "{}{}.mp4".format(self.video_name, camera_name)
        video = cv2.VideoWriter(file_video_name, cv2.VideoWriter_fourcc(*'mp4v'),
                                self.frame_rate, (video_size, video_size))
        for frame in video_to_send:
            video.write(frame)
        video.release()

        alert_icon = "⚠"
        if state == "Recover":
            alert_icon = "👍🏼"

        message = "{}{} detected{}: {}".format(alert_icon, state, camera_name, alert_icon)
        telegram_send.send(messages=[message])
        telegram_send.send(captions=[message], videos=[open(file_video_name, 'rb')])

    def make_square(self, img, expected_pixels):
        max_side = max(img.shape[0:2])

        # Background square
        square_img = np.zeros((max_side, max_side, 3), np.uint8)

        # Getting the centering position
        ax, ay = (max_side - img.shape[1]) // 2, (max_side - img.shape[0]) // 2

        # Pasting the image in a centering position
        square_img[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img

        return cv2.resize(square_img, (expected_pixels, expected_pixels))

    def show_results(self, img, camera, state):
        org = (40, 40)
        color = (255, 255, 0)
        thickness = 1
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.imshow(camera, cv2.putText(img, state, org, font, font_scale, color, thickness, cv2.LINE_AA))


if __name__ == "__main__":
    # Get args
    parser = get_args()
    args = parser.parse_args()
    # Start the program
    fall_detector = FallDetector(args)
    path_cameras = open(args.path_cameras, 'r')
    for line in path_cameras.readlines():
        if line.startswith("#") or line == '':
            continue

        parts = line.strip().split(", ")
        print("Starting camera {}".format(parts[1]))
        threading.Thread(target=fall_detector.estimate_pose,
                         args=(parts[0], parts[1],)).start()
