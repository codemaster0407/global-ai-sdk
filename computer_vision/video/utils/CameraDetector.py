import cv2
import platform
import os
import subprocess
import glob

class CameraDetector:
    """
    A utility class to detect and list available camera devices across different platforms.
    """
    
    def __init__(self):
        """
        Initialize the CameraDetector instance.

        Retrieves the platform system and prints it.
        """
        self.system = platform.system()
        print(platform.system())
        
    def get_device_list(self):
        """
        Get a list of all available camera devices.
        
        Returns:
        list: A list of dictionaries containing device information
              Each dictionary has 'id', 'name', and 'working' keys
        """
        if self.system == 'Windows':
            return self._get_windows_devices()
        elif self.system == 'Linux':
            return self._get_linux_devices()
        elif self.system == 'Darwin':  # macOS
            return self._get_macos_devices()
        else:
            return []

    def _test_camera(self, device_id):
        """
        Test if a camera device is working by attempting to open it.
        
        Args:
        device_id: Camera device identifier (integer or string)
        
        Returns:
        bool: True if camera opens successfully, False otherwise
        """
        try:
            cap = cv2.VideoCapture(device_id)
            if not cap.isOpened():
                return False
            ret, _ = cap.read()
            cap.release()
            return ret is True
        except Exception as e:
            print(f'Exception at _test_camera func: {e}')
            return False

    def _get_windows_devices(self):
        """
        Get available camera devices on Windows.
        """
        devices = []
        # Try the first 10 indices
        for i in range(10):
            if self._test_camera(i):
                devices.append({
                    'id': i,
                    'name': f'Camera {i}',
                    'working': True
                })
        return devices

    def _get_linux_devices(self):
        """
        Get available camera devices on Linux.
        """
        devices = []
        
        # Look for video devices in /dev
        video_devices = glob.glob('/dev/video*')
        
        for device_path in sorted(video_devices):
            try:
                # Extract device number
                device_num = int(device_path.replace('/dev/video', ''))
                
                # Test if device is working
                is_working = self._test_camera(device_num)
                
                if is_working:
                    # Try to get more information about the device
                    device_name = f"Camera {device_num}"
                    try:
                        cmd = ['udevadm', 'info', '--name=' + device_path, '--query=property']
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        for line in result.stdout.splitlines():
                            if 'ID_MODEL=' in line:
                                device_name = line.split('=')[1]
                                break
                    except Exception as e:
                        print(f'Exception at get_linux_devices func : {e}')
                        pass  # If we can't get detailed info, just use the basic name

                    devices.append({
                        'id': device_num,
                        'name': device_name,
                        'path': device_path,
                        'working': True
                    })
            except Exception as e:
                print(f'Exception at get_linux_devices func : {e}')
                continue
                
        return devices

    def _get_macos_devices(self):
        """
        Get available camera devices on macOS.
        """
        devices = []
        # Try the first 10 indices
        for i in range(3):
            if self._test_camera(i):
                devices.append({
                    'id': i,
                    'name': f'Camera {i}',
                    'working': True
                })
        return devices

    def get_working_devices(self):
        """
        Get a list of only the working camera devices.
        
        Returns:
        list: A list of dictionaries containing only working device information
        """
        return [device for device in self.get_device_list() if device['working']]

# Example usage
def fetch_device_id(device_type = '48MP'):
    """
    Fetch a camera device ID based on the device type.

    Args:
        device_type (str): The type of camera device to search for. Defaults to '48MP'.

    Returns:
        str: The ID of the first camera device found with the matching device type, or an empty string if no matching device is found.
    """
    detector = CameraDetector()
    devices = detector.get_device_list()
    device_ids = []
    if not devices:
        print("No camera devices found")
        return
    
    print("Found camera devices:")
    for device in devices:
        device_ids.append(device['id'])
        if device_type in  device['name']:
            return device['id']
        print(f"ID: {device['id']}")
        print(f"Name: {device['name']}")
        print(f"Working: {device['working']}")
        if 'path' in device:
            print(f"Path: {device['path']}")
        print("---")
    return device_ids[0] if len(device_ids) > 0 else ""
