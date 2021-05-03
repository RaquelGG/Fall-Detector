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

        show_args(args)

        # Load human state classifier model
        self.model = pickle.load(open(args.fall_model_path, 'rb'))
        with open(args.path_model_header) as f:
            self.model_header = f.readline().split(",")

    def estimate_pose(self, camera):
        pe = PoseEstimation()
        camera = cv2.VideoCapture(camera)
        video_rate = int(np.round(camera.get(cv2.CAP_PROP_FPS) / self.frame_rate))
        total_frames = self.frame_rate * self.chunk_seconds
        print("input frames per seconds", camera.get(cv2.CAP_PROP_FPS))
        state = "Nothing"
        # I use deque because I want to delete de first element when other one is added
        video_keypoints = deque(maxlen=total_frames)
        video_to_send = cv2.VideoWriter("state.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
                                        self.frame_rate, (pe.expected_pixels, pe.expected_pixels))
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
                video_to_send.write(frame)

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
                    video_to_send.release()
                    state = self.check_state(np.reshape(video_keypoints, (1, -1)), "state.mp4")

                    # Clear queue
                    video_keypoints.clear()

                    video_to_send = cv2.VideoWriter("state.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
                                                    self.frame_rate, (pe.expected_pixels, pe.expected_pixels))

                self.show_results(frame, state)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #
        mean = np.mean(times)
        camera.release()
        video_to_send.release()
        cv2.destroyAllWindows()

    def check_state(self, video_keypoints, video_to_send):
        state = str(self.model.predict(video_keypoints)[0])
        if state == "Fall":
            self.report(message="⚠ Fall detected ⚠", caption="The video of the fall", video=video_to_send)
        elif state == "Recover":
            self.report(message="👍🏼Recover detected!👍🏼", caption="The video of the recover", video=video_to_send)
        return state

    def report(self, message=None, caption=None, video=None):
        telegram_send.send(messages=[message])
        telegram_send.send(captions=[caption], videos=[open(video, 'rb')])

    def make_square(self, img, expected_pixels):
        max_side = max(img.shape[0:2])

        # Background square
        square_img = np.zeros((max_side, max_side, 3), np.uint8)

        # Getting the centering position
        ax, ay = (max_side - img.shape[1]) // 2, (max_side - img.shape[0]) // 2

        # Pasting the image in a centering position
        square_img[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img

        return cv2.resize(square_img, (expected_pixels, expected_pixels))

    def show_results(self, img, state):
        org = (40, 40)
        color = (255, 255, 0)
        thickness = 1
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.imshow("frame", cv2.putText(img, state, org, font, font_scale, color, thickness, cv2.LINE_AA))


if __name__ == "__main__":
    # Get args
    parser = get_args()
    args = parser.parse_args()
    # Start the program
    fall_detector = FallDetector(args)

    fall_detector.estimate_pose(r"D:\Universidad\TFG\videos\2-4.mp4")
