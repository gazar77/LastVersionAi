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

        # Read Photometric Interpretation to handle inverted images
        pi = ds.PhotometricInterpretation if 'PhotometricInterpretation' in ds else 'MONOCHROME2'
        
        if pi == 'MONOCHROME1':
            frames = np.amax(frames) - frames

        is_rgb = False
        if len(frames.shape) == 2:
            # (height, width) - single frame grayscale
            height, width = frames.shape
            frames = [frames]
        elif len(frames.shape) == 3:
            if frames.shape[2] == 3:
                # (height, width, 3) - single frame RGB
                height, width, _ = frames.shape
                frames = [frames]
                is_rgb = True
            else:
                # (frames, height, width) - multi-frame grayscale
                num_frames, height, width = frames.shape
        elif len(frames.shape) == 4:
            # (frames, height, width, 3) - multi-frame RGB
            num_frames, height, width, _ = frames.shape
            is_rgb = True
        else:
            print(f"Unsupported DICOM shape: {frames.shape}")
            return False

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

            if is_rgb:
                # DICOM RGB is usually RGB, OpenCV expects BGR
                img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
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
