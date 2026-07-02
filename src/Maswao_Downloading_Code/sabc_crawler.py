import json
import os
import requests
import yt_dlp
import cv2
import re
from collections import Counter
from moviepy import VideoFileClip
import subprocess

def get_sabc_videos(language, publishedBefore, publishedAfter, maxResults):
    api_key = "AIzaSyBFK_8MQOKhaJ7rmBxMaNSAnDjEPFpImhY"

    channel_id = findChannelId(language)
    originalMax = maxResults
    
    query = ""
    if language == "sesotho":
        query = "Sesotho"
    elif language == "sepedi":
        query = "Sepedi"
    elif language == "setswana":
        query = "Setswana"
    else:
        query = ""
    
    download_dir = os.path.join(os.getcwd(), 'youtube_downloads')
    os.makedirs(download_dir, exist_ok=True)

    url = f"https://www.googleapis.com/youtube/v3/search"
    params = {
        "key": api_key,
        "channelId": channel_id,
        "order": "date",
        "part": "snippet",
        "type": "video",
    }

    if (maxResults and (not query)):
        params["maxResults"] = maxResults * 3
    if query:
        params["q"] = query
        if maxResults:
            params["maxResults"] = maxResults * 3
    if publishedAfter:
        params["publishedAfter"] = publishedAfter
    if publishedBefore:
        params["publishedBefore"] = publishedBefore

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = json.loads(response.text)

        # Download each video
        for item in data['items']:
            if query:
                if query.lower() not in item['snippet']['title'].lower():
                    continue

            video_url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            
            # Download video
            ydl_opts = {
                'format': 'best',
                'outtmpl': os.path.join(download_dir, safe_filename(item['snippet']['title']) + '.%(ext)s'),
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                    print(f"Video downloaded: {item['snippet']['title']}")
                    video_path = "youtube_downloads/" + safe_filename(item['snippet']['title']) + ".mp4"
                    rectangles = find_rectangles_in_video(video_path, start_time=30, end_time=40)
                    # print("Detected Rectangles (x, y, width, height):")
                    # for rect in rectangles:
                    #     print(rect)
                    # print(f"Total Rectangles Detected: {len(rectangles)}")
                    
                    if len(rectangles) <= 40:
                        os.remove(video_path)
                        continue
                    else:
                        create_cropped_video(video_path, rectangles, "cropped_"+safe_filename(item['snippet']['title']) + ".mp4")

            except Exception as e:
                print(f"Error downloading video: {e}")

            # Download audio
            ydl_music_opts = {
                'format': 'bestaudio/best',
                'extractaudio': True,
                'audioformat': 'mp3',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(download_dir, safe_filename(item['snippet']['title']) + '.%(ext)s'),
            }

            try:
                with yt_dlp.YoutubeDL(ydl_music_opts) as ydl:
                    ydl.download([video_url])
                    print(f"Audio downloaded: {safe_filename(item['snippet']['title'])}.mp3")
                    if originalMax > 0:
                        originalMax -= 1
                        if originalMax == 0:
                            break

            except Exception as e:
                print(f"Error downloading audio: {e}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error accessing YouTube API: {e}")
    except Exception as e:
        print(f"Error processing videos: {e}")

def safe_filename(filename):
    keep_chars = r'[^a-zA-Z0-9\s.]'
    safe_name = re.sub(keep_chars, '', filename)
    return safe_name.replace(' ', '_')

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

def find_rectangles_in_video(video_path, start_time=30, end_time=40):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Frame rate (frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)

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
    min_width, min_height = 90, 90
    max_width, max_height = 135, 135

    rectangle_dimensions = []

    current_frame = start_frame
    while cap.isOpened() and current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the bottom-right part of the frame
        bottom_right_part = frame[roi_bottom:roi_top, roi_left:roi_right]

        # Convert to grayscale
        gray = cv2.cvtColor(bottom_right_part, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to highlight regions (adjust threshold value if needed)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through contours and find rectangular regions with height > width
        for contour in contours:
            # Get bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)

            # Check if the rectangle's width and height are within the specified range
            if (min_width <= w <= max_width) and (min_height <= h <= max_height):
                # Adjust the detected rectangle's position based on the cropping
                rectangle_dimensions.append((x + roi_left, y + roi_bottom, w, h))  # Adjust x and y by the ROI

        # Optionally, you can display the frame with detected rectangles for debugging
        for rect in rectangle_dimensions:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Break on a specific key press if desired
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        current_frame += 1

    # Release the video capture object and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    return rectangle_dimensions

def create_cropped_video(video_path, rectangles, output_filename):
    # Count repeated dimensions of rectangles
    dimension_counter = Counter((rect[2], rect[3]) for rect in rectangles)
    repeated_dimensions = [dim for dim, count in dimension_counter.items() if count > 15]
    
    if repeated_dimensions:
        most_common_dim = repeated_dimensions[0]
        crop_rectangles = [rect for rect in rectangles if (rect[2], rect[3]) == most_common_dim]
        crop_left = crop_rectangles[0][0]
        crop_top = crop_rectangles[0][1]
        crop_width = crop_rectangles[0][2]
        crop_height = crop_rectangles[0][3]
        
        # Load the video clip
        video_clip = VideoFileClip(video_path)
        
        # Ensure that the crop is within bounds
        video_width, video_height = video_clip.size
        crop_left = max(0, min(crop_left, video_width - 1))  
        crop_top = max(0, min(crop_top, video_height - 1))  
        crop_width = min(crop_width, video_width - crop_left)  
        crop_height = min(crop_height, video_height - crop_top)  
        
        # Crop the video using crop method
        cropped_clip = video_clip.cropped(
            x1=crop_left,
            y1=crop_top,
            x2=crop_left + crop_width,
            y2=crop_top + crop_height
        )
        
        # Calculate target dimensions while maintaining aspect ratio
        aspect_ratio = cropped_clip.w / cropped_clip.h
        target_width = min(cropped_clip.w * 2, 1280)  
        target_height = int(target_width / aspect_ratio)
        
        # Ensure target dimensions are within bounds
        target_width = min(target_width, video_width) 
        target_height = min(target_height, video_height)  
        
        # Resize the cropped video
        resized_clip = cropped_clip.resized(
            (target_width, target_height)
        )
        
        # Write the final video with optimized settings
        resized_clip.write_videofile(
            "./youtube_downloads/" + output_filename,
            codec="libx264",
            audio_codec="aac",
            fps=video_clip.fps,  
            preset="slower",  
            bitrate="5000k",
            threads=4  
        )
        
        print(f"Processed video saved as: {output_filename}")


get_sabc_videos("setswana", "", "", 1)