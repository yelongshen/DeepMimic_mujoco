import numpy as np
import cv2
import os
import time

class VideoSaver():
    """Video saver for recording environment episodes"""
    
    def __init__(self, video_path=None, width=640, height=480, fps=30, show_type="dump"):
        """
        Initialize video saver
        
        Args:
            video_path: Full path to save video file (if None, auto-generates name)
            width: Video width
            height: Video height  
            fps: Frames per second
            show_type: "dump" to save to file, "play" to display
        """
        assert show_type in ["dump", "play"]
        
        self.show_type = show_type
        self.fps = fps
        self.width = width
        self.height = height
        self.video_fd = None
        self.frame_count = 0
        
        if self.show_type == "dump":
            if video_path is None:
                # Auto-generate path
                dump_dir = "./render"
                if not os.path.isdir(dump_dir):
                    os.makedirs(dump_dir, exist_ok=True)
                video_path = os.path.join(
                    dump_dir,
                    time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '.avi'
                )
            else:
                # Ensure directory exists
                dump_dir = os.path.dirname(video_path)
                if dump_dir and not os.path.isdir(dump_dir):
                    os.makedirs(dump_dir, exist_ok=True)
            
            self.video_path = video_path
            
            # Remove existing file
            if os.path.isfile(video_path):
                os.remove(video_path)
            
            # Setup codec
            (major, _, _) = cv2.__version__.split(".")
            if major == '2':
                fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            
            print(f"VideoSaver: Opening {video_path}")
            self.video_fd = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            if not self.video_fd.isOpened():
                raise RuntimeError(f"Failed to open video writer for {video_path}")
        
        print("VideoSaver: Initialized successfully")
    
    def add_frame(self, img):
        """Add a frame to the video (alias for addFrame)"""
        self.addFrame(img)
    
    def addFrame(self, img):
        """
        Add a frame to the video
        
        Args:
            img: RGB image as numpy array [H, W, 3]
        """
        if img is None:
            return
        
        # Convert RGB to BGR for OpenCV
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        
        # Resize if needed
        if img_bgr.shape[0] != self.height or img_bgr.shape[1] != self.width:
            img_bgr = cv2.resize(img_bgr, (self.width, self.height))
        
        if self.show_type == "dump":
            if self.video_fd is not None:
                self.video_fd.write(img_bgr)
                self.frame_count += 1
        else:
            # Display mode
            cv2.imshow('Video', img_bgr)
            cv2.waitKey(1)
    
    def save(self):
        """Save and close the video file"""
        self.close()
    
    def close(self):
        """Close the video writer"""
        if self.video_fd is not None:
            self.video_fd.release()
            print(f"VideoSaver: Saved {self.frame_count} frames to {self.video_path}")
            self.video_fd = None
        
        if self.show_type == "play":
            cv2.destroyAllWindows()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()


# Legacy compatibility
class VideoSaver_Old():
    """Old VideoSaver class for backward compatibility"""
    gShowType = "dump"
    gDumpDir = "./render"
    gDumpFd = None
    gFps = 15

    def __init__(self, showType="dump", dumpDir="./render", width=640, height=480, fps=15):
        assert showType in ["dump", "play"]

        self.gShowType = showType
        self.gDumpDir = dumpDir
        self.gFps = fps
        if self.gShowType == "dump":
            if not os.path.isdir(self.gDumpDir):
                os.mkdir(self.gDumpDir)
            video_name = self.gDumpDir + "/" + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '.avi'
            if os.path.isfile(video_name):
                os.remove(video_name)

            (major, _, _) = cv2.__version__.split(".")
            if major == '2':
                fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')

            print("demoShow.init openning %s " %(video_name))
            self.gDumpFd = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
        print("demoShow.init success")

    def close(self):
        assert self.gDumpFd != None
        self.gDumpFd.release()

    def addFrame(self, img):
        if self.gShowType == "dump":
            assert self.gDumpFd != None
            self.gDumpFd.write(img)
        else:
            sleepTime = 1
            cv2.imshow('image', img)
            cv2.waitKey(sleepTime)