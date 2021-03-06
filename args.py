import argparse


def get_args():
    frame_rate = 5
    telegram_alert = True
    fall_model_path = "human_state_classifier/model/HSC.sav"
    posenet_model_path = "../Google Dataset (posenet)/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
    chunk_seconds = 3
    path_model_header = "human_state_classifier/model/header.txt"
    path_cameras = "cameras.conf"
    display_video = False


    parser = argparse.ArgumentParser(description="Real-Time Fall Detector",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--chunk_seconds",
                        help="Seconds for each chunk.",
                        type=int, default=chunk_seconds)

    parser.add_argument("-f", "--frame_rate",
                        help="Frames per second.",
                        type=int, default=frame_rate)

    parser.add_argument("-t", "--telegram_alert",
                        help="Sends an alert when a fall is detected.",
                        type=bool, default=telegram_alert)

    parser.add_argument("-p", "--posenet_model_path",
                        help="Path to the PoseNet model.",
                        type=str, default=posenet_model_path)

    parser.add_argument("-m", "--fall_model_path",
                        help="Path to the human state classifier model.",
                        type=str, default=fall_model_path)

    parser.add_argument("-x", "--path_model_header",
                        help="Path to the human state classifier model header.",
                        type=str, default=path_model_header)

    parser.add_argument("-c", "--path_cameras",
                        help="Path to the configuration file containing the camera urls",
                        type=str, default=path_cameras)

    parser.add_argument("-d", "--display_video",
                        help="Display videos from cameras",
                        type=bool, default=display_video)

    return parser


def show_args(args):
    print("Seconds per chunk:", args.chunk_seconds, "s")
    print("Frames per second:", args.frame_rate, "frames")
    print("Sends an alert when a fall is detected:", args.telegram_alert)
    print("Display videos from cameras:", args.display_video)
    print("Path to human state classifier model: '", args.fall_model_path, "'")
    print("Path to PoseNEt model: '", args.posenet_model_path, "'")
    print("Path to the configuration file containing the camera urls: '", args.path_cameras, "'")


