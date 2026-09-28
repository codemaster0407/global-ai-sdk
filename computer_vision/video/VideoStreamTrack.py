import asyncio
import cv2
import os
import signal
import time
from aiortc import MediaStreamTrack


class VideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, camera_id=0, width=3840, height=2160, fps=30, max_retries=5):
        """
        Initialize the VideoStreamTrack with the specified camera settings.

        Args:
            camera_id (int): ID of the camera to use.
            width (int): Desired width of the video resolution.
            height (int): Desired height of the video resolution.
            fps (int): Desired frames per second.
            max_retries (int): Number of retries if camera fails to open.
        """
        super().__init__()
        self.camera_id = camera_id
        self.fps = fps
        self.retry_count = 0
        self.max_retries = max_retries  # Maximum allowed retries before shutting down
        self.cap = self.initialize_camera(camera_id, width, height, fps)

    def initialize_camera(self, camera_id, width, height, fps):
        """
        Try initializing the camera with retries.

        Args:
            camera_id (int): Camera ID.
            width (int): Desired width.
            height (int): Desired height.
            fps (int): Desired FPS.

        Returns:
            cv2.VideoCapture: OpenCV VideoCapture object.
        """
        while self.retry_count < self.max_retries:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                self.configure_camera(cap, width, height, fps)
                print("Camera successfully initialized.")
                return cap
            else:
                print(f"️Failed to open camera {camera_id}. Retrying... ({self.retry_count + 1}/{self.max_retries})")
                self.retry_count += 1
                time.sleep(2)

        print("Camera initialization failed after retries. Shutting down FastAPI server.")
        self.shutdown_server()
        return None

    async def recv(self):
        """
        Read the next frame from the camera asynchronously, ensuring FPS control.

        Returns:
            frame (numpy array): The video frame, or None if reading fails.
        """
        # Check if camera is still open before trying to read
        if not self.cap or not self.cap.isOpened():
            print("Camera not open in recv. Attempting to reinitialize...")
            await self.handle_camera_failure()
            return None
            
        # Add a timeout mechanism to prevent hanging
        try:
            # Use asyncio.wait_for to add a timeout
            ret, frame = await asyncio.wait_for(
                asyncio.to_thread(self.cap.read),
                timeout=2.0  # 2 second timeout
            )
            
            if not ret or frame is None or frame.size == 0:
                print("Failed to read valid frame. Checking camera status...")
                await self.handle_camera_failure()
                return None
                
            return frame
            
        except asyncio.TimeoutError:
            print("Camera read operation timed out. Reinitializing camera...")
            await self.handle_camera_failure()
            return None
        except Exception as e:
            print(f"Unexpected error reading from camera: {e}")
            await self.handle_camera_failure()
            return None

    async def handle_camera_failure(self):
        """
        Handle camera failure by retrying initialization. If retries fail, shut down the FastAPI server.
        """
        self.retry_count += 1
        if self.retry_count >= self.max_retries:
            print("Maximum camera failures reached. Shutting down FastAPI server.")
            self.shutdown_server()
            return

        print(f"️Camera failure detected. Attempting to reinitialize ({self.retry_count}/{self.max_retries})...")
        await asyncio.sleep(2)
        self.cap.release()
        self.cap = self.initialize_camera(self.camera_id, 3840, 2160, self.fps)

    def configure_camera(self, cap, width=3840, height=2160, fps=30, codec="MJPG"):
        """
        Configure camera settings including resolution, FPS, and codec.

        Args:
            cap (cv2.VideoCapture): OpenCV camera object.
            width (int): Width of the video.
            height (int): Height of the video.
            fps (int): Frames per second.
            codec (str): Codec format (default: "MJPG").
        """
        fourcc = cv2.VideoWriter_fourcc(*codec)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for real-time streaming

        print("Camera initialized with the following settings:")
        self.print_camera_configuration(cap)

    def print_camera_configuration(self, cap):
        """Print the current camera configuration."""
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        codec = self.decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))

        print(f"Resolution: {int(width)}x{int(height)}, FPS: {int(fps)}, Codec: {codec}")

    def decode_fourcc(self, v):
        """
        Decode a FOURCC (Four Character Code) integer to a readable string format.

        Args:
            v (int): FOURCC integer value.

        Returns:
            str: Decoded FOURCC string.
        """
        v = int(v)
        return "".join([chr((v >> (8 * i)) & 0xFF) for i in range(4)])

    async def stop(self):
        """
        Asynchronously stop the video stream and release camera resources.
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("Camera stream stopped.")
        # await super().stop()

    def shutdown_server(self):
        """
        Gracefully shuts down the FastAPI server when the camera fails.
        """
        print("Shutting down FastAPI server due to camera failure.")
        os.kill(os.getpid(), signal.SIGTERM)