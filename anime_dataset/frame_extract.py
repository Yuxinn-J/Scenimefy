import os
import json
import cv2
import argparse

VALID_VIDEO_EXTENSIONS = ['.mp4', '.mkv']


def extract_frames(args):
    # Load the JSON file
    with open(args.json_path, 'r') as f:
        movie_data = json.load(f)

    # Loop through each movie
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
            print(f"Movie'{movie_name}' not found in '{args.data_dir}'.")
            continue

        print(f"Movie '{movie_name}' start extracting...")

        # Create a folder for the current movie's frames
        save_dir = os.path.join(args.save_path, movie_name)
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
                    frame_file = os.path.join(save_dir, f"frame_{frame_num}.png")
                    cv2.imwrite(frame_file, frame)
                else:
                    print(f"Error reading frame {frame_num} from movie '{movie_name}'.")

            else:
                print(f"Invalid frame number {frame_num} for movie '{movie_name}'.")
            
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("json_path", metavar="FILE", type=str, help="json file path")
    parser.add_argument("data_dir", metavar="FILE", type=str, help="path to movie folder")
    parser.add_argument("save_path", metavar="FILE", type=str, help="path to save extracted data")
    args = parser.parse_args()

    extract_frames(args)
    # python frame_extract.py frameList.json F:\raw_video F:\dataset