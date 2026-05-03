import pydicom
import cv2
import numpy as np
import os

def convert_dicom_to_mp4(dicom_path, output_video_path):
    """
    Converts a DICOM file (multi-frame or single-frame) to an MP4 video.
    """
    try:
        ds = pydicom.dcmread(dicom_path)
        frames = ds.pixel_array

        # Handle multi-frame or single-frame
        if len(frames.shape) == 3:
            height, width = frames[0].shape
        else:
            height, width = frames.shape
            frames = [frames]

        video = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            15,
            (width, height)
        )

        # Handle overlay if present
        overlay = None
        if (0x6000, 0x3000) in ds:
            try:
                overlay = ds.overlay_array(0x6000)
            except:
                pass

        for frame in frames:
            if frame.dtype != np.uint8:
                if frame.max() > 0:
                    frame = (frame / frame.max() * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

            img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Draw overlay if present
            if overlay is not None:
                mask = overlay.astype(bool)
                if mask.shape == frame.shape:
                    img[mask] = [255, 255, 255]

            video.write(img)

        video.release()
        return True
    except Exception as e:
        print(f"Error converting DICOM: {str(e)}")
        return False
