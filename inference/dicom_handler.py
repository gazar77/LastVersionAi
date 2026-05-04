import pydicom
import cv2
import numpy as np
import os


def _normalize_to_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert single-channel or RGB frame to uint8 for VideoWriter."""
    if frame.dtype == np.uint8:
        return frame
    f = frame.astype(np.float32)
    lo = float(np.min(f))
    hi = float(np.max(f))
    if hi <= lo:
        return np.zeros_like(frame, dtype=np.uint8)
    return ((f - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _iter_slices(arr: np.ndarray, is_rgb: bool):
    """
    Yield (frame_2d_or_hwc_rgb, frame_index) from pixel_array with shape:
    (H,W), (H,W,3), (N,H,W), (N,H,W,3).
    """
    if arr.ndim == 2:
        yield arr, 0
        return
    if arr.ndim == 3:
        if is_rgb or arr.shape[2] == 3:
            yield arr, 0
            return
        # (N, H, W) grayscale
        for i in range(arr.shape[0]):
            yield arr[i], i
        return
    if arr.ndim == 4:
        for i in range(arr.shape[0]):
            yield arr[i], i
        return
    raise ValueError(f"Unsupported DICOM pixel_array ndim={arr.ndim} shape={arr.shape}")


def convert_dicom_to_mp4(dicom_path, output_video_path):
    """
    Converts a DICOM file (multi-frame or single-frame) to an MP4 video.
    """
    video = None
    frames_written = 0
    try:
        ds = pydicom.dcmread(dicom_path)
        frames = np.asarray(ds.pixel_array)

        pi = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        if pi == "MONOCHROME1":
            frames = np.amax(frames) - frames

        is_rgb = False
        if frames.ndim == 3 and frames.shape[2] == 3:
            is_rgb = True
        elif frames.ndim == 4:
            is_rgb = True

        # Overlay: optional 2D mask or per-frame 3D stack
        overlay = None
        if (0x6000, 0x3000) in ds:
            try:
                overlay = ds.overlay_array(0x6000)
            except Exception:
                overlay = None

        slices = list(_iter_slices(frames, is_rgb))
        if not slices:
            return False

        first, _ = slices[0]
        if is_rgb:
            height, width = first.shape[0], first.shape[1]
        else:
            height, width = first.shape[0], first.shape[1]

        if width <= 1 or height <= 1:
            return False

        fps = 15.0
        video = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not video.isOpened():
            return False

        for frame, idx in slices:
            frame_u8 = _normalize_to_uint8(frame)

            if is_rgb:
                img = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
            else:
                img = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)

            if overlay is not None:
                ov = overlay
                if ov.ndim == 3:
                    if idx < ov.shape[0]:
                        m = ov[idx].astype(bool)
                    else:
                        m = None
                elif ov.ndim == 2:
                    m = ov.astype(bool)
                else:
                    m = None

                if m is not None and m.shape[:2] == img.shape[:2]:
                    img[m] = [255, 255, 255]

            video.write(img)
            frames_written += 1

        return frames_written > 0
    except Exception as e:
        print(f"Error converting DICOM: {str(e)}")
        return False
    finally:
        if video is not None:
            try:
                video.release()
            except Exception:
                pass
        # Remove corrupt / empty output if nothing valid was written
        try:
            if frames_written == 0 and output_video_path and os.path.exists(output_video_path):
                os.remove(output_video_path)
        except OSError:
            pass
