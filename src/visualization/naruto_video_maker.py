"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm

def adjust_string_length(desired_length: int, input_str: str) -> str:
    """
    Adjusts the length of a string to a specified length by padding with spaces or cutting the string.

    Args:
        desired_length (int): The desired length of the string.
        input_str (str)     : The input string to adjust.

    Returns:
        str: The adjusted string with the specified length.
    """
    # If the input string is shorter than the desired length, pad it with spaces
    if len(input_str) < desired_length:
        return input_str.ljust(desired_length)
    # If the input string is longer than the desired length, cut it to fit
    else:
        return input_str[:desired_length]

if __name__ == "__main__":
    ##################################################
    ### argument parsing
    ##################################################
    def argument_parsing() -> argparse.Namespace:
        """parse arguments

        Returns:
            args: arguments
            
        """
        parser = argparse.ArgumentParser(
                description="Arguments to make video for NARUTO."
            )
        parser.add_argument("--scene", type=str, default="", 
                    help="scene name")
        parser.add_argument("--base_dir", type=str, default="", 
                    help="visualization data directory")
        parser.add_argument("--out_video", type=str, default="output.mp4", 
                    help="output video path")
        parser.add_argument("--pb_speed", type=int, default=1, 
                            help="playback speed")
        args = parser.parse_args()
        return args

    args = argument_parsing()
    ##################################################
    ### Video properties
    ##################################################
    frame_width = 1920 * 3
    frame_height = 1080 * 3
    caption_frame_width = frame_width
    caption_frame_height = 250
    frame_rate = 30
    video_codec = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = args.out_video

    # Calculate individual image size (assuming 4 images per row)
    img_width = frame_width // 4
    img_height = frame_height // 2

    ##################################################
    ### Directory containing image folders
    ##################################################
    base_dir = args.base_dir
    folders = ['rgbd',
            f'traj_vis_at_{args.scene}_view1', 
            f'rendered_color_mesh_at_{args.scene}_view1', 
            f'rendered_uncert_mesh_at_{args.scene}_view1', 
            'rgbd',
            f'traj_vis_at_{args.scene}_view2',
            f'rendered_color_mesh_at_{args.scene}_view2', 
            f'rendered_uncert_mesh_at_{args.scene}_view2', 
            ]

    ### initialize video writer ###
    # Assuming all folders contain the same number of images
    num_images = len(os.listdir(os.path.join(base_dir, folders[0])))

    # Create a VideoWriter object
    video_writer = cv2.VideoWriter(output_video_path, video_codec, frame_rate, (frame_width, frame_height+caption_frame_height))

    ##################################################
    ### main
    ##################################################
    for i in tqdm(range(num_images)):
        ### playback speed ###
        if i % args.pb_speed != 0:
            continue 

        ### Create a blank frame ###
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        ##################################################
        ### load image
        ##################################################
        for idx, folder in enumerate(folders):
            ### Calculate position ###
            row = idx // 4
            col = idx % 4

            ### Read and resize image ###
            if folder in [f"traj_vis_at_{args.scene}_view1", f"traj_vis_at_{args.scene}_view2"]:
                img_path = os.path.join(base_dir, folder, f'{i//5*5:04}.png')  # Adjust file naming as needed
            elif "rendered" in folder:
                img_path = os.path.join(base_dir, folder, f'{i//5*5:04}.png')  # Adjust file naming as needed
            else:
                img_path = os.path.join(base_dir, folder, f'{i:04}.png')  # Adjust file naming as needed
            img = cv2.imread(img_path)
            
            ### RGB from RGB-D ###
            if idx == 0:
                h, w, _ = img.shape
                img = img[:, :w//2]
            elif idx == 4:
                img = img[:, w//2:]
            else:
                img = img
            
            img = cv2.resize(img, (img_width, img_height))

            ### put caption ###
            if folder == f"traj_vis_at_{args.scene}_view1":
                line = "View 1"
                cv2.putText(img, line, (100, 100), cv2.FONT_HERSHEY_SIMPLEX , 3, (0, 255, 0), 10, cv2.LINE_AA)
            if folder == f"traj_vis_at_{args.scene}_view2":
                line = "View 2"
                cv2.putText(img, line, (100, 100), cv2.FONT_HERSHEY_SIMPLEX , 3, (0, 255, 0), 10, cv2.LINE_AA)

            ### Place image in the frame ###
            start_y = img_height * row
            end_y = start_y + img_height
            start_x = img_width * col
            end_x = start_x + img_width
            frame[start_y:end_y, start_x:end_x, :] = img

        ##################################################
        ### initialize caption properties
        ##################################################
        ### caption properties ###
        caption_frames = []
        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # org 
        org1 = (100, 100)
        org2 = (100, 200) 
        # fontScale 
        fontScale = 3
        # Blue color in BGR 
        color = (0, 255, 0) 
        # Line thickness of 2 px 
        thickness = 10

        ##################################################
        ### put caption
        ##################################################
        for j in range(4):
            caption_frame = np.ones((caption_frame_height, caption_frame_width//4, 3), dtype=np.uint8) * 0
            caption_frame[:125] += 122
            caption_frame[125:] += 0

            ### first line ###
            if j == 0:
                with open(os.path.join(base_dir, "state", f"{i:04}.txt"), 'r') as f:
                    state = f.readlines()[0]
                state_str = adjust_string_length(25, f"State: {state}")
                line = f"{state_str}"
                cv2.putText(caption_frame, line, org1, font, fontScale, color, thickness, cv2.LINE_AA)
            if j == 1:
                line = "  Final Mesh (2cm voxel)"
                cv2.putText(caption_frame, line, org1, font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)
            if j == 2:
                line = "   ---: Planned Path"
                cv2.putText(caption_frame, line, org1, font, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)
            if j == 3:
                line = "  --- : Uncertain Targets"
                cv2.putText(caption_frame, line, org1, font, fontScale, (255,255,255), thickness, cv2.LINE_AA)

            ### second line ###
            if j == 0:
                line = f"Step-{i:04}"
            elif j == 1:
                line = "        Trajectory"
            elif j == 2:
                line = "Textured Mesh (5cm voxel)"
            elif j == 3:
                line = "   Normalized Uncertainty"
            else:
                raise NotImplementedError
            
            ### put text ###
            cv2.putText(caption_frame, line, org2, font, fontScale, color, thickness, cv2.LINE_AA)
            
            caption_frames.append(caption_frame)
        
        ### concatenate caption frames ###
        caption_frames = np.concatenate(caption_frames, axis=1)
        
        ### concatenate caption frames with main frame ###
        frame = np.concatenate([frame, caption_frames], axis=0)

        ### Write the frame to the video ###
        video_writer.write(frame)

    ### Release the VideoWriter ###
    video_writer.release()
    print("Video has been created successfully.")