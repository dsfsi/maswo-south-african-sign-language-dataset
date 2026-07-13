import json
import os
import requests
import yt_dlp
import cv2
import re
from collections import Counter
from moviepy import VideoFileClip
import subprocess
import argparse
import pandas as pd
from yt_dlp import YoutubeDL

def findChannelId(language):
    if language == "afrikaans":
        return "UCPde3flccFt8JwCQwmOCWsg"
    elif language == "tshivenda":
        return "UC3liuiQ-WmUe1vAEOQxErZQ"
    elif language == "isixhosa":
        return "UCl8gMaUWyG6v9gxpq6iWbmA"
    elif language == "isizulu":
        return "UCSO8qij295DSxMiofTLk9QQ"
    elif language == "tsonga":
        return "UC5fWoGVzmD7p1PFeatPPvyw"
    elif language == "sesotho":
        return "UCfJ6oO7RhiemsjzDuakMJKQ"
    elif language == "sepedi":
        return "UCfJ6oO7RhiemsjzDuakMJKQ"
    elif language == "setswana":
        return "UCfJ6oO7RhiemsjzDuakMJKQ"
    else:
        return "UC8yH-uI81UUtEMDsowQyx1g"
    
# save filename    
def safe_filename(filename):
    keep_chars = r'[^a-zA-Z0-9\s.]'
    safe_name = re.sub(keep_chars, '', filename)
    return safe_name.replace(' ', '_') 

# This function returns True if s is a number else False
def isNumber(s):
    for char in s:
        if not char.isdigit():
            return False
    return True

def append_to_csv(filename, row, columns, unique_col=None):
    """
    Append a new row to a CSV file using pandas, avoiding duplicates if `unique_col` is provided.

    Parameters:
    - filename (str): Path to the CSV file.
    - row (dict): A dictionary representing the new row (keys should match columns).
    - columns (list): List of column names for the CSV file.
    - unique_col (str, optional): Column name to check for duplicates.
    """
    print(f"Appending to CSV: {filename}")
    # Create an empty DataFrame if the file doesn't exist
    if not os.path.exists(filename):
          df = pd.DataFrame(columns=columns)
    else:
          # Load existing data
          df = pd.read_csv(filename)

    # Check if the value in row[unique_col] exists in the 'Video_Url' column
    if not df.empty:
        if str(unique_col) in df['Video_Url'].astype(str).tolist():
            print(f"Value '{row['Video_Url']}' already exists in Video_Url column. Skipping.")
            return 
    
    # Append new row and save
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.head()
    df.to_csv(filename, index=False)


# list of cookies
def get_cookie_files(cookie_dir="./cookies/"):
    """
    Get a list of all cookie files in the specified directory.
    
    Parameters:
    - cookie_dir (str): Directory containing cookie files
    
    Returns:
    - list: List of paths to cookie files
    """
    if not os.path.exists(cookie_dir):
        print(f"Cookie directory {cookie_dir} does not exist. Using default cookie.")
        return ["./cookies.txt"]
        
    cookie_files = [os.path.join(cookie_dir, f) for f in os.listdir(cookie_dir) 
                  if f.endswith('.txt') and os.path.isfile(os.path.join(cookie_dir, f))]
    
    if not cookie_files:
        print("No cookie files found. Using default cookie.")
        return ["./cookies.txt"]
    
    print(f"Found {len(cookie_files)} cookie files: {cookie_files}")
    return cookie_files

class CookieManager:
    def __init__(self, initial_cookie_path="./cookies.txt", cookie_dir="./cookies/"):
        """
        Initialize the cookie manager with available cookie files.
        
        Parameters:
        - initial_cookie_path (str): Path to the initial cookie file
        - cookie_dir (str): Directory containing additional cookie files
        """
        self.cookie_files = get_cookie_files(cookie_dir)
        
        # If the initial cookie isn't in our list, add it
        if initial_cookie_path not in self.cookie_files and os.path.exists(initial_cookie_path):
            self.cookie_files.insert(0, initial_cookie_path)
            
        self.current_index = 0
        self.failed_cookies = set()
        
    def get_current_cookie(self):
        """Get the current cookie file path"""
        if not self.cookie_files:
            raise Exception("No cookie files available!")
        return self.cookie_files[self.current_index]
    
    def rotate_cookie(self):
        """
        Rotate to the next available cookie file.
        
        Returns:
        - str: Path to the next cookie file
        - None: If all cookies have failed
        """
        # Mark current cookie as failed
        self.failed_cookies.add(self.cookie_files[self.current_index])
        
        # Try to find a cookie that hasn't failed yet
        for _ in range(len(self.cookie_files)):
            self.current_index = (self.current_index + 1) % len(self.cookie_files)
            if self.cookie_files[self.current_index] not in self.failed_cookies:
                print(f"Rotating to cookie: {self.cookie_files[self.current_index]}")
                return self.cookie_files[self.current_index]
        
        # If all cookies have failed
        print("All cookie files have failed!")
        return None

# Find rectangles
def find_rectangles_in_video(video_path, start_time=30, end_time=60):

    print(f"Processing video: {video_path}")
    # Open the video file
    
    cap = cv2.VideoCapture(video_path)

    print(f"Video opened: {cap.isOpened()}")
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return []

    # Frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print("Error: FPS is 0, possible codec issue.")
        return []

    # Convert start and end time to frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Set the video to start at the start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Define region of interest (ROI) as the bottom-right part of the frame
    roi_bottom = int(frame_height * 0.5)  # Bottom half of the frame
    roi_top = frame_height
    roi_left = int(frame_width * 0.5)  # Right half of the bottom half
    roi_right = frame_width

    # Define the minimum and maximum width and height for the rectangles
    min_width, min_height = 290, 300
    max_width, max_height = 390, 400

    rectangle_dimensions = []
    current_frame = start_frame

    print("Processing frames...")
    while cap.isOpened() and current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame, possibly end of video.")
            break

        # Convert WebM frame to correct color format (some WebM videos might be read as grayscale)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Crop the bottom-right part of the frame
        bottom_right_part = frame[roi_bottom:roi_top, roi_left:roi_right]

        # Convert to grayscale
        gray = cv2.cvtColor(bottom_right_part, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to highlight regions
        # _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(gray, 100, 280)
        thresh = cv2.dilate(edges, None, iterations=0)


        # Find contours in the thresholded image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through contours and find rectangular regions with height > width
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # print(f"Detected rectangle at ({x}, {y}) with size ({w}x{h})")
            if (min_width <= w <= max_width) and (min_height <= h <= max_height):
                # print(f"Valid rectangle with size ({w}x{h})")
                rectangle_dimensions.append((x + roi_left, y + roi_bottom, w, h))

        # Optionally, display the frame for debugging
        for rect in rectangle_dimensions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        current_frame += 1

    print(f"Total rectangles detected: {len(rectangle_dimensions)}")
    cap.release()
    cv2.destroyAllWindows()

    return rectangle_dimensions

def create_cropped_video(video_path, out_path, rectangles, output_filename, cropped):
    
    print("Creating cropped video...")
    # Count repeated dimensions of rectangles and get smallest dimension
    dimension_counter = Counter((rect[2], rect[3]) for rect in rectangles)
    smallest_dimensions = min(dimension_counter)
    most_common_dim = smallest_dimensions

    print("Dimension counter:", dimension_counter)
    print("Smallest dimension:", smallest_dimensions)

    if(smallest_dimensions[0] < 323 or smallest_dimensions[0] > 345):
        smallest_dimensions = (334, smallest_dimensions[1])
    if(smallest_dimensions[1] < 373 or smallest_dimensions[1] > 395):
        smallest_dimensions = (smallest_dimensions[0], 383)

    crop_rectangles = [rect for rect in rectangles if (rect[2], rect[3]) == most_common_dim]

    if not crop_rectangles:
        print("No rectangles with the selected dimensions. Using default crop position.")
        crop_left = 1437
        crop_top = 540
        crop_width = smallest_dimensions[0]
        crop_height = smallest_dimensions[1]
    else:
        print("Crop rectangles:", crop_rectangles)
        if(crop_rectangles[0][0] < 1426 or crop_rectangles[0][0] > 1448):
            crop_left = 1437
        else:
            crop_left = crop_rectangles[0][0]
            
        if(crop_rectangles[0][1] < 529 or crop_rectangles[0][1] > 551):
            crop_top = 540
        else:
            crop_top = crop_rectangles[0][1]
            
        crop_width = smallest_dimensions[0]
        crop_height = smallest_dimensions[1]

    # Load the video clip to get dimensions
    try:
        video_clip = VideoFileClip(video_path)
        
        # Ensure crop dimensions are within video bounds
        video_width, video_height = video_clip.size
        crop_left = max(0, min(crop_left, video_width - 1))  
        crop_top = max(0, min(crop_top, video_height - 1))  
        crop_width = min(crop_width, video_width - crop_left)  
        crop_height = min(crop_height, video_height - crop_top)
        
        # Close the clip to free resources
        video_clip.close()

        # Determine output dimensions - simpler approach
        output_width = min(1280, crop_width * 2)
        output_height = min(720, crop_height * 2)
        
        # Crop and resize the video using ffmpeg
        command = (
            f"ffmpeg -hwaccel cuda -i {video_path} -vf "
            f"crop={crop_width}:{crop_height}:{crop_left}:{crop_top},"
            f"scale={output_width}:{output_height} "
            f"-c:v libvpx-vp9 -b:v 5000k -c:a opus -crf 30 -b:a 192k "
            f"-threads 32 -y {os.path.join(out_path, output_filename)}"
        )

        print(f"Running command: {command}")
        
        # Run the command and capture the output
        process = subprocess.run(command, shell=True, capture_output=True, text=True)

        print(f"Command has been executed")
        
        if process.returncode == 0:
            print(f"Processed WebM video saved as: {output_filename}")
            if os.path.getsize(os.path.join(out_path, output_filename)) > 0:
                return True
            else:
                print("Error: Output file is empty.")
                print(f"ffmpeg stderr: {process.stderr}")
                return False
        else:
            print(f"ffmpeg command failed with code {process.returncode}")
            print(f"ffmpeg stderr: {process.stderr}")
            return False
            
    except Exception as e:
        print(f"Error in create_cropped_video: {e}")
        return False

def reencode_webm(input_path):
    """
    Re-encode a WebM file to ensure compatibility with OpenCV.

    Parameters:
    - input_path (str): Path to the input WebM file.

    Returns:
    - bool: True if re-encoding was successful, False otherwise.
    """
    temp_output_path = input_path + "_temp.webm"

    command = [
        "ffmpeg", "-i", input_path,
        "-c:v", "libvpx-vp9", "-b:v", "2M",
        "-c:a", "libopus", 
        "-crf", "30", "-b:a", "192k",
        "-threads", "32",
        "-y", temp_output_path
    ]

    print(f"Re-encoding {input_path} to {temp_output_path}...")
    process = subprocess.run(command, capture_output=True, text=True)

    if process.returncode == 0:
        print(f"Re-encoding successful. Replacing original file.")
        os.replace(temp_output_path, input_path)  # Replace the original file
        return True
    else:
        print(f"Error during re-encoding: {process.stderr}")
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)  # Clean up temp file if it exists
        return False

# Get available formats
def get_available_formats(video_url, cookiefile=None):
    ydl_opts = {'quiet': True}
    if cookiefile:
        ydl_opts['cookiefile'] = cookiefile

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        return [f['ext'] for f in info['formats']]


# Convert mp4 to webm
def convert_mp4_to_webm(input_path, output_path=None):
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".webm"

    cmd = [
        "ffmpeg", "-i", input_path,
        "-c:v", "libvpx-vp9", "-b:v", "2M",
        "-c:a", "libopus",
        output_path
    ]

    subprocess.run(cmd, check=True)
    return output_path

def check_video_exists(filename, video_id):
    """
    Check if a video ID already exists in the CSV file.
    
    Parameters:
    - filename (str): Path to the CSV file
    - video_id (str): The video ID to check
    
    Returns:
    - bool: True if video exists, False otherwise
    """
    # If file doesn't exist, the video isn't in it
    if not os.path.exists(filename):
        return False
    
    try:
        # Read the CSV file
        df = pd.read_csv(filename)
        
        # Check if the 'id' column exists and if the video_id is in it
        if 'id' in df.columns and video_id in df['id'].values:
            print(f"Video ID {video_id} already exists in {filename}")
            return True
        return False
    except Exception as e:
        print(f"Error checking for existing video: {e}")
        # If there's an error reading the file, assume video isn't in it
        return False
    
def download_store_videos(url, args, params):
    
    cookie_manager = CookieManager(args.cookies, cookie_dir="./cookies/")

    successful_downloads = 0
    target_downloads = int(params.get("maxResults", 0)) if isNumber(params.get("maxResults", 0)) else float('inf')
    next_page_token = None

    filename = args.meta_data_dir + args.language + '_meta_file.csv'
    print(f"Meta data file: {filename}")
    print
    if not os.path.exists(args.meta_data_dir):
        os.makedirs(args.meta_data_dir, exist_ok=True)

    while successful_downloads < target_downloads:

        if next_page_token:
            params['pageToken'] = next_page_token

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = json.loads(response.text)

            if 'items' not in data or len(data['items']) == 0:
                print("No more videos found.")
                break
        
            next_page_token = data.get('nextPageToken')
            
            # print(data)
            # Download each video
            for i, item in enumerate(data['items']):
                if successful_downloads >= target_downloads:
                    break

                print(f"Looking at video: {item['snippet']['title']}")
                print(f"Query: {params}")

                if check_video_exists(filename, item['id']['videoId']):
                    print(f"Video {item['id']['videoId']} already exists in {filename}. Skipping..")
                    continue

                if args.rerun == 0:
                    if params.get('q'):
                        if params['q'].lower() not in item['snippet']['title'].lower():
                            # Sepedi, Setswana, Sesotho are under the same channel
                            continue

                    print(f"Video Title: {item['snippet']['title']}")
                    video_url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                    print(f"Video URL: {video_url}")
                    
                    path_for_video = os.path.join(args.download_dir, str(item['id']['videoId']))

                    if not os.path.isdir(path_for_video):
                        os.makedirs(path_for_video)

                    current_cookie = cookie_manager.get_current_cookie()
                    if not current_cookie:
                        print("No valid cookies available. Cannot continue downloading.")
                        break

                    format = get_available_formats(video_url, cookiefile=current_cookie)
                    print(f"Available formats: {format}")

                    video_path = ""
                    if 'webm' in format:
                        # make it webm
                        ydl_opts = {
                            'format': 'bestvideo[height<=1080]/best[ext=webm]',
                            'merge_output_format': 'webm',
                            'outtmpl': os.path.join(path_for_video, safe_filename(item['snippet']['title']) + '.webm'),
                            'cookiefile': args.cookies,
                        }
                        video_path = os.path.join(path_for_video, safe_filename(item['snippet']['title']) + '.webm')

                    else:
                        ydl_opts = {
                            'format': 'bestvideo[height<=1080]/best[ext=mp4]',
                            'merge_output_format': 'mp4',
                            'outtmpl': os.path.join(path_for_video, safe_filename(item['snippet']['title']) + '.mp4'),
                            'cookiefile': args.cookies,
                        }
                        video_path = os.path.join(path_for_video, safe_filename(item['snippet']['title']) + '.mp4')

                    try:
                        print(f"#######Now downloading video###########")
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([video_url])
                            print(f"Video downloaded: {item['snippet']['title']}")

                            if os.path.exists(video_path):
                                print(f"Video file found at: {video_path}")

                                # Re-encode the video to ensure compatibility with OpenCV
                                # if not reencode_webm(video_path):
                                #     print("Error: Could not re-encode video. Skipping this video.")
                
                                rectangles = find_rectangles_in_video(video_path, start_time=30, end_time=40)
                                
                                print(f"Total Rectangles Detected: {len(rectangles)}")

                                cropped_done = False            
                                if len(rectangles) > 0:
                                    cropping = create_cropped_video(video_path, path_for_video, rectangles, "cropped_"+safe_filename(item['snippet']['title']) + ".webm", cropped_done)
                                    cropped_done = cropping
                                    continue
                                else:
                                    # cropping = create_cropped_video(video_path, path_for_video, rectangles, "h22_cropped_"+safe_filename(item['snippet']['title']) + ".webm", cropped_done)
                                    cropped_done = cropped_done
                            
                            else:
                                print(f"Error: Video file not found at {video_path}")
                                next_cookie = cookie_manager.rotate_cookie()
                                if not next_cookie:
                                    print("-------No more cookies available. Exiting.-------")
                                    break
                                cropped_done = False

                    except Exception as e:
                        print(f"Error downloading video: {e}")
                        
                        if "429" in str(e) or "quota" in str(e).lower() or "exceeded" in str(e).lower() or "permission" in str(e).lower():
                            print("Detected rate limiting or permissions error. Rotating cookie...")
                            next_cookie = cookie_manager.rotate_cookie()
                            if not next_cookie:
                                break
                        else:
                            # For other errors, just skip this video
                            print(f"Skipping video due to non-cookie related error: {e}")
                            break

                        cropped_done = False

                    print(f"#######Now downloading audio###########")
                    mp3_stored = False
                    #Download audio
                    ydl_music_opts = {
                        'outtmpl': os.path.join(path_for_video, safe_filename(item['snippet']['title']) + '_mp3'),
                        'format': 'bestaudio/best',
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                            'preferredquality': '0',
                        }],
                        'cookiefile': current_cookie,
                    }

                    try:
                        with yt_dlp.YoutubeDL(ydl_music_opts) as ydl:
                            ydl.download([video_url])
                            print(f"Audio downloaded: {safe_filename(item['snippet']['title'])}.mp3")
                        mp3_stored = True   
                    
                    except Exception as e:
                        print(f"Error downloading audio: {e}")

                    # save record metadata to csv
                    # columns  = ["id", "Video_Url", "URL", "Video_title", "language", "path", "Extracted_Crop", "Extracted_mp3", "Mp3_path", ]
                    filename = args.meta_data_dir + args.language + '_meta_file.csv'

                    # create directory if not exists
                    if(not os.path.exists(args.meta_data_dir)):
                        os.makedirs(args.meta_data_dir, exist_ok=True)

                    row = {"id":item['id'], "Video_Url": item['id']['videoId'] ,"URL":video_url, "Video_title": item['snippet']['title'], "language":args.language, "path":video_path, 
                    "Extracted_Crop":cropped_done, "Extracted_mp3":mp3_stored,"Mp3_path":os.path.join(path_for_video, safe_filename(item['snippet']['title']) + ".mp3")} 
                
                    unique_id = item['id']['videoId']
                    columns = list(row.keys())
                    append_to_csv(filename, row, columns, unique_id)   
                    successful_downloads += 1

                    if cropped_done == False:
                        os.rmdir(path_for_video)

                    if next_page_token is None:
                        print("No more pages to process.")
                        break

        except Exception as e:
            print(f"Error processing videos: {e}")    
            break #need to confirm if this is correct

def main (args):
        args = args
        # add api_key
        api_key = args.api_key

        # get channel
        channel = findChannelId(args.language)

        # number of videos returned
        maxVideos = args.maxResults

        # create outputdir
        print('Creating output directory')
        language = args.language
        if args.language =="":
                language = 'English'

        download_dir = os.path.join(args.download_dir, language.capitalize())
        # download_dir = os.path.join('./Youtube', language.capitalize())
        if not os.path.isdir(download_dir):
                os.makedirs(download_dir, exist_ok=True)

        args.download_dir = download_dir

        #static params 
        url = f"https://www.googleapis.com/youtube/v3/search"
        params = {
            "key": args.api_key,
            "channelId": channel,
            "order": "date",
            "part": "snippet",
            "type": "video",
        }

        # define query
        query = ""
        if args.language == "sesotho":
            query = "Sesotho"
        elif args.language == "sepedi":
            query = "Sepedi"
        elif args.language == "setswana":
            query = "Setswana"
        elif args.language == "isizulu":
             query = "IsiZulu"
        elif args.language == "isiXhosa":
             query = "IsiXhosa"
        elif args.language == "isiSwati":
             query = "IsiSwati" 
        elif args.language == "isiNdebele":
             query = "IsiNdebele"     
        elif args.language == "tshivenda":
             query = "Tshivenda"
        elif args.language == "xitsonga":
             query = "Xitsonga" 
        elif args.language == "afrikaans":
             query == "Afrikaans"                             
        else:
            query = ""
        # print(query)

        # update params
        if isNumber(maxVideos):
            # print(maxVideos)
            if (maxVideos and (not query)):
                params["maxResults"] = maxVideos
            if query:
                params["q"] = query
                if maxVideos:
                    params["maxResults"] = maxVideos
            if args.publishedAfter:
                 params["publishedAfter"] = args.publishedAfter
            if args.publishedBefore:
                params["publishedBefore"] = args.publishedBefore        
        else:
               if maxVideos == "all":
                     params["maxResults"] = ""
                     if args.publishedAfter:
                         params["publishedAfter"] = args.publishedAfter
                     if args.publishedBefore:
                          params["publishedBefore"] = args.publishedBefore 
               if query:
                    params["q"] = query
                    if maxVideos == "all":
                        params["maxResults"] = ""  
                    else:
                         params["maxResults"] =  maxVideos * 1       
        
        # download videos
        try:
             #download
             download_store_videos(url, args, params)
        except:
             print("Something went wrong with the download")     
         
        
if __name__ == "__main__":
        
        cookies_dir = "./cookies/"
        if not os.path.exists(cookies_dir):
            os.makedirs(cookies_dir, exist_ok=True)
            print(f"Created cookies directory: {cookies_dir}")
            print("Please place additional cookie files in this directory")

        parser = argparse.ArgumentParser()
        parser.add_argument("--language", type=str, default="sepedi", help="language being crawled")
        parser.add_argument("--api_key", type=str, default="AIzaSyBFK_8MQOKhaJ7rmBxMaNSAnDjEPFpImhY", help="API key to get videos from Youtube")
        parser.add_argument("--maxResults",  default="1", help="number of requested videos")
        parser.add_argument("--download_dir", type=str, default="", help="root path for outputs")
        parser.add_argument("--publishedBefore", type=str, default="", help="date of publish before")
        parser.add_argument("--publishedAfter", type=str, default="", help="date of publish after")
        parser.add_argument("--rerun", type=int, default=0, help="0 if starting the run and 1 if code terminated as ran again")
        parser.add_argument("--meta_data_dir", type=str, default="./meta_outputs/", help="root path for metadata outputs")
        parser.add_argument("--cookies", type=str, default="./cookies.txt", help="root path for cookies file")

        args = parser.parse_args() 
        # print(args.cookies)
        main(args)      