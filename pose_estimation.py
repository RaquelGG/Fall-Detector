import tensorflow as tf  # pip install tensorflow
import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
from os import path


class PoseEstimation:
    model_path = "../Google Dataset (posenet)/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
    body_parts = ["nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder", "rightShoulder",
                  "leftElbow", "rightElbow", "leftWrist", "rightWrist", "leftHip", "rightHip", "leftKnee",
                  "rightKnee", "leftAnkle", "rightAnkle"]

    def __init__(self, model_path=model_path):
        # Load the TFLite model and allocate tensors.
        if path.isfile(model_path):
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
        else:
            self.interpreter = tf.lite.Interpreter(model_path="../{}".format(model_path))
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.expected_pixels = self.input_details[0]['shape'][1]

    def get_pose_estimation(self, img):
        # Resize input image

        #img = make_square(img, expected_pixels)

        # Convert image to a 1D numpy array
        input_data = np.expand_dims(img.copy(), axis=0)

        # check the type of the input tensor
        floating_model = self.input_details[0]['dtype'] == np.float32
        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        # Setting the value of the input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run the computation
        self.interpreter.invoke()

        # Extract output data from the interpreter
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        offset_data = self.interpreter.get_tensor(self.output_details[1]['index'])

        # The output consist of 2 parts:
        # - heatmaps (9,9,17) - corresponds to the probability of appearance of
        # each keypoint in the particular part of the image (9,9)(without applying sigmoid
        # function). Is used to locate the approximate position of the joint
        # - offset vectors (9,9,34) is called offset vectors. Is used for more exact
        #  calculation of the keypoint's position. First 17 of the third dimension correspond
        # to the x coordinates and the second 17 of them correspond to the y coordinates
        heatmaps = np.squeeze(output_data)
        offsets = np.squeeze(offset_data)

        # show = np.squeeze((input_data.copy() * 127.5 + 127.5) / 255.0)
        # show = np.array(show * 255, np.uint8)
        # cv2.imshow('image', )
        # cv2.imwrite("../Images/puntos.jpg", draw_kps(img, keyparts))

        pose = get_keypoints_positions(heatmaps, offsets)
        # Show image with pose
        #cv2.imshow("frame", cv2.resize(self.draw_kps(img, pose), (500, 500)))
        # return keyparts
        return pose

    def draw_kps(self, img, kps, ratio=None):
        for i in range(5, kps.shape[0]):
            org = (kps[i, 1], kps[i, 0])
            color = (255, 255, 0)
            thickness = 1
            radius = 2
            font_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX

            if isinstance(ratio, tuple):
                cv2.circle(img, (int(round(kps[i, 1] * ratio[1])), int(round(kps[i, 0] * ratio[0]))), radius,
                           color, round(int(1 * ratio[1])))
                continue

            cv2.circle(img, org, radius, color, -1)
            cv2.putText(img, str(i), org, font, font_scale, color, thickness, cv2.LINE_AA)

        return img


def get_keypoints_positions(heatmap, offsets):
    """
  Input:
    heatmap - heatmaps for an image. Three dimension array
    offset - offset vectors for an image. Three dimension array

  Output:
    array with coordinates of the keypoints
  """
    # Number of keypoints detected by PoseNet
    keypoints_detected = heatmap.shape[-1]
    keypoints_positions = np.zeros((keypoints_detected, 2), np.float64)

    for i in range(keypoints_detected):
        # get body data from heatmap
        layer = heatmap[..., i]
        # get x and y index in the heatmap with the highest score
        approx_position = np.squeeze(np.unravel_index(layer.argmax(), layer.shape))
        # get the x and y from the offsets corresponding to the x and y index in the heatmap for that part
        relocation = np.array(approx_position / 8 * 256)
        keypoints_positions[i, 0] = int(relocation[0] + offsets[approx_position[0],
                                                                   approx_position[1], i])
        keypoints_positions[i, 1] = int(relocation[1] + offsets[approx_position[0],
                                                                   approx_position[1],
                                                                   i + keypoints_detected])
        # max_prob = np.max(layer)
        # if max_prob > threshold:
        #    if keypoints_positions[i, 0] < 257 and keypoints_positions[i, 1] < 257:
        #        keypoints_positions[i, 2] = 1

        # if max_prob <= threshold:
        #    keypoints_positions[i, 0] = 555
        #    keypoints_positions[i, 1] = 555

    return keypoints_positions
