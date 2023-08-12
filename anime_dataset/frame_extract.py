import os
import json
import cv2
import argparse

VALID_VIDEO_EXTENSIONS = ['.mp4', '.mkv']


def crop_max_square(image, pos):
    global x
    h, w, c = image.shape

    if pos == 0: # left
        x = 0
    elif pos == 1: # center
        x = (w - h) // 2
    elif pos == 2: # right
        x = w - h
    else:
        raise ValueError("Invalid position value for cropping.")

    x = int(x)
    return image[:, x:x+h]


def extract_frames(args):
    # Load the JSON file
    with open(args.json_path, 'r') as f:
        movie_data = json.load(f)

    for movie_name, frame_nums in movie_data.items():
        """
        Check if the video file exists
        """
        movie_file = None
        for ext in VALID_VIDEO_EXTENSIONS:
            movie_file = os.path.join(args.data_dir, f"{movie_name}{ext}")
            if os.path.exists(movie_file):
                break
        
        if not movie_file or not os.path.isfile(movie_file):
            print(f"'{movie_name}' not found in '{args.data_dir}'.")
            continue

        print(f"'{movie_name}' start extracting...")

        # Create folder
        save_dir = os.path.join(args.save_path, "Anime Scene" if args.crop else movie_name)
        os.makedirs(save_dir, exist_ok=True)

        # Read the video
        cap = cv2.VideoCapture(movie_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Extract frames from the specified frame numbers
        for frame_num in frame_nums:
            if 0 <= frame_num < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if ret:
                    target_frames = [frame] if not args.crop else [crop_max_square(frame, i) for i in range(3)]

                    for i, target_frame in enumerate(target_frames):
                        name = os.path.join(save_dir, f"frame_{frame_num}_{i}.png")
                        cv2.imwrite(name, target_frame)
                        
                else:
                    print(f"Error reading {frame_num} from '{movie_name}'.")

            else:
                print(f"Invalid {frame_num} for '{movie_name}'.")
            
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("json_path", metavar="FILE", type=str, help="json file path")
    parser.add_argument("data_dir", metavar="FILE", type=str, help="path to movie folder")
    parser.add_argument("save_path", metavar="FILE", type=str, help="path to save extracted data")
    parser.add_argument('--crop', action='store_true', help='if specified, then crop to 1080x1080')
    args = parser.parse_args()

    extract_frames(args)
    # python frame_extract.py frameList.json F:\raw_video F:\dataset