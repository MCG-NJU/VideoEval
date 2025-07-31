import os
import shutil
from moviepy.editor import VideoFileClip

def get_video_info(video_path):
    """
    Retrieve video resolution and frame rate information
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        tuple: (resolution, fps) where resolution is (width, height)
               Returns (None, None) if error occurs
    """
    try:
        with VideoFileClip(video_path) as clip:
            # Get resolution as (width, height)
            resolution = clip.size
            # Get frame rate
            fps = clip.fps
            return resolution, fps
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None, None

def filter_videos(input_dir, output_dir):
    """
    Filter videos with low resolution (below 240p) or low frame rate (below 15fps)
    
    Args:
        input_dir (str): Directory containing videos to process
        output_dir (str): Directory to save filtered videos
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through all files in input directory
    for filename in os.listdir(input_dir):
        # Check if file has a video extension
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
            input_path = os.path.join(input_dir, filename)
            resolution, fps = get_video_info(input_path)
            
            if resolution is None or fps is None:
                print(f"Could not process video: {filename}")
                continue
            
            width, height = resolution
            # Check if resolution is below 240p (using height as reference)
            is_low_res = height < 240
            # Check if frame rate is below 15fps
            is_low_fps = fps < 15
            
            if is_low_res or is_low_fps:
                print(f"Filtered out: {filename} - Resolution: {width}x{height}, Frame rate: {fps:.2f}fps")
            else:
                # Copy acceptable videos to output directory
                output_path = os.path.join(output_dir, filename)
                shutil.copy2(input_path, output_path)
                print(f"Kept: {filename} - Resolution: {width}x{height}, Frame rate: {fps:.2f}fps")

if __name__ == "__main__":
    # Input and output folder paths
    input_folder = "path/to/your/video/folder"  # Replace with your video folder path
    output_folder = "filtered_videos"  # Path to save filtered videos
    
    # Run the filtering process
    filter_videos(input_folder, output_folder)
    print("Video filtering completed!")