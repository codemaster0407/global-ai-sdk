import cv2
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from computer_vision.video.VideoStreamTrack import VideoStreamTrack
from collections import deque
import threading
import time
import uvicorn

STREAM_SIZE = (640, 480)


class StreamManager:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.frame_queue = deque(maxlen=5)
        self.video_track = None
        self.current_client_stop_event = None
        self.is_capturing = False
        self.capture_task = None
        self._lock = threading.Lock()
        self.is_camera_on = False  # Track if the camera is active
        self.last_frame_time = time.time()
        self.watchdog_task = None
        self.watchdog_interval = 60  # Check every minute

    async def initialize(self):
        """Initialize video track if not already initialized"""
        if self.video_track is None:
            # VideoStreamTrack opens the camera with blocking retries, keep it off the event loop
            self.video_track = await asyncio.to_thread(VideoStreamTrack, self.camera_id)
            print("Video stream initialized.")
            self.is_camera_on = True
            self.last_frame_time = time.time()  # Don't let the watchdog see stale time from a previous run
            self.start_watchdog()

    def start_watchdog(self):
        """Start the watchdog task to monitor camera health"""
        if self.watchdog_task is None:
            self.watchdog_task = asyncio.create_task(self.camera_watchdog())
            print("Camera watchdog started.")

    async def stop_watchdog(self):
        """Stop the watchdog task"""
        if self.watchdog_task and self.watchdog_task is not asyncio.current_task():
            self.watchdog_task.cancel()
            try:
                await self.watchdog_task
            except asyncio.CancelledError:
                pass
        self.watchdog_task = None

    async def restart_camera(self):
        """Restart the camera completely"""
        print("Performing camera restart...")

        # Stop everything
        await self.stop_camera()

        # Wait a moment to ensure full cleanup
        await asyncio.sleep(2)

        # Start everything again
        await self.initialize()
        await self.start_capture()

        print("Camera restarted successfully.")

    async def camera_watchdog(self):
        """Monitor camera health and restart if necessary"""
        try:
            while self.is_camera_on:
                await asyncio.sleep(self.watchdog_interval)
                if not self.is_camera_on:
                    break

                # Check if we've received frames recently
                time_since_last_frame = time.time() - self.last_frame_time
                if time_since_last_frame > self.watchdog_interval * 2:
                    print(f"WARNING: No frames received in {time_since_last_frame:.1f} seconds. Restarting camera.")
                    await self.restart_camera()
                    continue

                # Check if OpenCV still has the device open
                cap = getattr(self.video_track, 'cap', None)
                if cap is not None and not cap.isOpened():
                    print("WARNING: Camera reported as closed. Restarting camera.")
                    await self.restart_camera()
        except asyncio.CancelledError:
            print("Camera watchdog task cancelled.")
            raise
        except Exception as e:
            print(f"Error in camera watchdog: {e}")
        finally:
            if self.watchdog_task is asyncio.current_task():
                self.watchdog_task = None

    async def start_capture(self):
        """Start frame capture if not already running"""
        if not self.is_capturing and self.is_camera_on:
            self.is_capturing = True
            self.capture_task = asyncio.create_task(self.capture_frames())
            print("Frame capture started.")

    async def stop_capture(self):
        """Stop frame capture"""
        if self.is_capturing:
            self.is_capturing = False
            if self.capture_task:
                self.capture_task.cancel()
                try:
                    await self.capture_task
                except asyncio.CancelledError:
                    pass
                self.capture_task = None
            print("Frame capture stopped.")

    async def stop_camera(self):
        """Stop camera and release resources"""
        await self.stop_capture()
        self.is_camera_on = False  # Mark camera as off
        if self.video_track:
            await self.video_track.stop()
            self.video_track = None
            print("Camera stopped and released.")
        with self._lock:
            self.frame_queue.clear()

    async def capture_frames(self):
        """Continuously capture frames from video stream"""
        while self.is_capturing:
            try:
                frame = await self.video_track.recv()
                if frame is not None:
                    # Update the timestamp when we successfully get a frame
                    self.last_frame_time = time.time()

                    low_res_frame = cv2.resize(frame, STREAM_SIZE, interpolation=cv2.INTER_AREA)
                    with self._lock:
                        self.frame_queue.append((frame, low_res_frame))
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"Error capturing frame: {e}")
                await asyncio.sleep(1)

    def latest_frame(self):
        """Return the most recent (frame, low_res_frame) pair, or None"""
        with self._lock:
            return self.frame_queue[-1] if self.frame_queue else None

    async def set_current_client(self):
        """Terminate previous client and set up new one"""
        if self.current_client_stop_event:
            self.current_client_stop_event.set()
            await asyncio.sleep(0.1)
        self.current_client_stop_event = asyncio.Event()
        return self.current_client_stop_event

    async def cleanup(self):
        """Clean up resources"""
        await self.stop_watchdog()
        await self.stop_camera()


# Global stream manager instance
stream_manager = StreamManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the camera on startup and release it on shutdown"""
    await stream_manager.initialize()
    await stream_manager.start_capture()
    yield
    await stream_manager.cleanup()


app = FastAPI(lifespan=lifespan)


@app.get("/video_feed")
async def video_feed(request: Request):
    """Stream video feed to a single client at a time"""
    if not stream_manager.is_camera_on:
        await stream_manager.initialize()
        await stream_manager.start_capture()
        await asyncio.sleep(1)

    stop_event = await stream_manager.set_current_client()

    async def generate_frames():
        try:
            while not stop_event.is_set():
                if await request.is_disconnected():
                    print("Client disconnected. Ending video stream.")
                    break

                latest = stream_manager.latest_frame()

                # If camera is stopped and queue is empty, break the loop
                if not stream_manager.is_camera_on and latest is None:
                    print("Camera stopped and no frames left. Ending video stream.")
                    break

                if latest is not None:
                    _, low_res_frame = latest
                    ok, jpeg = cv2.imencode('.jpg', low_res_frame)
                    if ok:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print("Streaming stopped for client.")
            raise
        except Exception as e:
            print(f"Error generating frames: {e}")
            raise

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/camera/start")
async def start_camera():
    """Start the camera stream"""
    if not stream_manager.is_camera_on:
        await stream_manager.initialize()
        await stream_manager.start_capture()
        return {"message": "Camera started successfully"}
    return {"message": "Camera is already running"}


@app.get("/camera/stop")
async def stop_camera():
    """Stop the camera stream and release the device"""
    if stream_manager.is_camera_on:
        await stream_manager.stop_camera()
        return {"message": "Camera stopped successfully"}
    return {"message": "Camera is already off"}


def start_streaming_backend():
    uvicorn.run(app, host="0.0.0.0", port=8000)
