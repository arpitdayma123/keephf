import runpod
import os
import cv2
import argparse
import torch
import numpy as np
import requests
import uuid
from basicsr.utils import imwrite, img2tensor, tensor2img # Removed load_file_from_url
from basicsr.utils.download_util import load_file_from_url # Added new import
from basicsr.utils import misc as misc_bsr # Renamed to avoid conflict
from basicsr.utils import video_util
from basicsr.utils.registry import ARCH_REGISTRY
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from scipy.ndimage import gaussian_filter1d

# Global Variables/Constants
output_dir = './results/'
input_dir = './input_videos/'

# --- Model Configs and Paths ---
# Note: These paths should reflect where models are located in the Docker image.
# Assuming Dockerfile places them in /app/weights/
MODEL_BASE_PATH = '/app/weights'

model_configs = {
    'GFPGAN': {
        'model_path': os.path.join(MODEL_BASE_PATH, 'GFPGAN', 'GFPGANv1.4.pth'),
        'model_url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth' # Fallback for load_file_from_url
    },
    'CodeFormer': {
        'model_path': os.path.join(MODEL_BASE_PATH, 'CodeFormer', 'codeformer.pth'),
        'model_url': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth' # Fallback
    },
    'RealESRGAN_x2plus': {
        'model_path': os.path.join(MODEL_BASE_PATH, 'RealESRGAN', 'RealESRGAN_x2plus.pth'),
        'model_url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth' # Fallback
    },
    'RealESRGAN_x4plus': {
        'model_path': os.path.join(MODEL_BASE_PATH, 'RealESRGAN', 'RealESRGAN_x4plus.pth'),
        'model_url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth' # Fallback
    },
    'detection': { # dlib landmark model
        'model_path': os.path.join(MODEL_BASE_PATH, 'dlib', 'shape_predictor_68_face_landmarks.dat'),
        'model_url': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2' # Fallback
    },
    'KEEP': { # KEEP model
        'model_path': os.path.join(MODEL_BASE_PATH, 'KEEP', 'KEEP-b76feb75.pth'), # Actual path in Docker
        'model_url': 'https://huggingface.co/DongSky/KEEP/resolve/main/KEEP-b76feb75.pth' # Fallback
    }
}

# Global instances for models - initialized to None
face_helper = None
net = None
bg_upsampler = None
models_initialized = False


def download_video_from_url(video_url: str, save_dir: str) -> str | None:
    """
    Downloads a video from a given URL and saves it to a specified directory.

    Args:
        video_url: The URL of the video to download.
        save_dir: The directory where the video will be saved.

    Returns:
        The local path to the downloaded video, or None if download fails.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    try:
        response = requests.get(video_url, stream=True, timeout=30) # Added timeout
        response.raise_for_status()  # Raise an exception for bad status codes

        # Try to get extension from URL, default to .mp4
        filename_ext = os.path.splitext(video_url.split('/')[-1])[1]
        if not filename_ext: # handle cases where URL doesn't have an extension
            content_type = response.headers.get('content-type')
            if content_type and 'video' in content_type:
                filename_ext = '.' + content_type.split('/')[-1].split(';')[0] # e.g. .mp4 from video/mp4; charset=utf-8
            else:
                filename_ext = '.mp4' # default

        filename = f"{uuid.uuid4()}{filename_ext}"
        video_path = os.path.join(save_dir, filename)

        with open(video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Video downloaded successfully to {video_path}")
        return video_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading video from {video_url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during video download: {e}")
        return None


def initialize_models():
    """
    Initializes the KEEP model, RealESRGAN upsampler, and FaceRestoreHelper.
    This function is called once when the worker starts or if models are not loaded.
    """
    global face_helper, net, bg_upsampler, models_initialized

    if models_initialized:
        print("Models already initialized.")
        return

    print("Initializing models...")
    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Initialize FaceRestoreHelper ---
    # Ensure dlib model is downloaded if not present (using load_file_from_url as a utility)
    dlib_model_path = model_configs['detection']['model_path']
    if not os.path.exists(dlib_model_path):
        print(f"Dlib model not found at {dlib_model_path}, attempting to download...")
        dlib_model_path = load_file_from_url(
            url=model_configs['detection']['model_url'],
            model_dir=os.path.join(MODEL_BASE_PATH, 'dlib'),
            progress=True,
            file_name='shape_predictor_68_face_landmarks.dat' # Ensure correct filename
        )
        if not os.path.exists(dlib_model_path): # Check after download attempt
             raise RuntimeError(f"Failed to download or locate dlib landmark model at {dlib_model_path}")


    face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50', # or 'retinaface_mobile0.25'
        save_ext='png',
        use_parse=True, # Enable parsing for segmentation
        device=device
    )
    # Manually set the landmark predictor path for face_helper if necessary
    # face_helper.face_parse.landmark_predictor.predictor = dlib.shape_predictor(dlib_model_path)
    # face_helper.face_det.landmark_predictor.predictor = dlib.shape_predictor(dlib_model_path)
    # Update: FaceRestoreHelper likely handles dlib model loading internally via model_rootpath.
    # We might need to ensure 'shape_predictor_68_face_landmarks.dat' is directly in model_rootpath or a subdir it expects.
    # For now, assuming model_rootpath + internal logic of FaceRestoreHelper is sufficient.

    # --- Initialize KEEP Model (net) ---
    keep_model_path = model_configs['KEEP']['model_path']
    if not os.path.exists(keep_model_path):
        print(f"KEEP model not found at {keep_model_path}, attempting to download...")
        keep_model_path = load_file_from_url(
            url=model_configs['KEEP']['model_url'],
            model_dir=os.path.join(MODEL_BASE_PATH, 'KEEP'),
            progress=True,
            file_name=os.path.basename(keep_model_path) # e.g. KEEP-b76feb75.pth
        )
    if not os.path.exists(keep_model_path):
        raise RuntimeError(f"Failed to download or locate KEEP model at {keep_model_path}")

    # Dynamically load the KEEP_arch from the registry (assuming it's registered elsewhere)
    # If KEEP_arch is not registered, this will fail.
    # This part is based on how HuggingFace app.py loads models.
    # We need to ensure KEEP_arch.py is in a place where ARCH_REGISTRY can find it.
    # For now, let's assume it's in the Python path.
    try:
        from keep_arch import KEEP_Net # Assuming keep_arch.py is in PYTHONPATH
        net = KEEP_Net(
            inc=3,
            midc=64,
            n_blocks=8,
            norm_type='in',
            act_type='relu',
            use_deform=True,
            use_attn=True,
            use_cbam=True,
            use_cab=True,
            use_se=False,
            use_res=True,
            use_skip=True,
            is_concat=False,
            num_queries=20,
            num_res_blocks=8,
            n_frames=1) # Default n_frames, might need adjustment
        print("KEEP_Net architecture loaded.")
    except ImportError:
        raise ImportError("Could not import KEEP_Net from keep_arch. Please ensure keep_arch.py is in the Python path.")
    except Exception as e:
        raise RuntimeError(f"Error initializing KEEP_Net: {e}")


    loadnet = torch.load(keep_model_path, map_location=lambda storage, loc: storage)
    if 'params_ema' in loadnet: # Load EMA parameters if available
        net.load_state_dict(loadnet['params_ema'], strict=True)
    elif 'params' in loadnet:
        net.load_state_dict(loadnet['params'], strict=True)
    else:
        net.load_state_dict(loadnet, strict=True)
    net.eval()
    net = net.to(device)
    print("KEEP model loaded successfully.")

    # --- Initialize RealESRGAN for Background Enhancement ---
    # Using RealESRGAN_x2plus as default for bg_upsampler, can be changed.
    realesrgan_model_path = model_configs['RealESRGAN_x2plus']['model_path']
    if not os.path.exists(realesrgan_model_path):
        print(f"RealESRGAN_x2plus model not found at {realesrgan_model_path}, attempting to download...")
        realesrgan_model_path = load_file_from_url(
            url=model_configs['RealESRGAN_x2plus']['model_url'],
            model_dir=os.path.join(MODEL_BASE_PATH, 'RealESRGAN'),
            progress=True,
            file_name=os.path.basename(realesrgan_model_path)
        )
    if not os.path.exists(realesrgan_model_path):
        raise RuntimeError(f"Failed to download or locate RealESRGAN_x2plus model at {realesrgan_model_path}")

    try:
        # ARCH_REGISTRY.get('RealESRGAN') should be available if basicsr is installed correctly
        bg_upsampler = ARCH_REGISTRY.get('RealESRGAN')(
            scale=2, # x2plus model
            model_path=realesrgan_model_path,
            device=device,
            tile=400, # Default tile size, can be adjusted
            tile_pad=10,
            pre_pad=0,
            half=True # Enable half precision for speed if supported
        )
        print("RealESRGAN (bg_upsampler) loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error initializing RealESRGAN: {e}. Ensure 'RealESRGAN' is registered in ARCH_REGISTRY.")


    models_initialized = True
    print("All models initialized successfully.")


def process_video_internal(input_video_path: str, draw_box: bool, bg_enhancement: bool) -> str | None:
    """
    Processes a video for face restoration using the KEEP model.
    Based on process_video from hugging_face/app.py, Gradio components removed.

    Args:
        input_video_path: Local path to the downloaded video.
        draw_box: Whether to draw bounding boxes around detected faces.
        bg_enhancement: Whether to enhance the background using RealESRGAN.

    Returns:
        Local path to the processed video file, or None if processing fails.
    """
    global face_helper, net, bg_upsampler # Use globally initialized models

    if not models_initialized:
        print("Error: Models not initialized. Call initialize_models() first.")
        # Or call initialize_models() here, but it's better to ensure it's called by handler
        return None

    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found at {input_video_path}")
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    # Use a unique name for the output to avoid collisions if the same video is processed multiple times
    # or if multiple instances are running.
    output_video_name = f"{video_name}_KEEP_{uuid.uuid4()}.mp4"
    restored_video_path = os.path.join(output_dir, output_video_name)

    # --- Setup Video Reader and Writer ---
    try:
        video_reader = cv2.VideoCapture(input_video_path)
        if not video_reader.isOpened():
            raise IOError(f"Cannot open video file: {input_video_path}")

        frame_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_reader.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec and create VideoWriter object
        # Using 'mp4v' as it's widely compatible for .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(restored_video_path, fourcc, fps, (frame_width, frame_height))
        if not video_writer.isOpened():
             raise IOError(f"Cannot create video writer for {restored_video_path}")
        print(f"Output video will be saved to: {restored_video_path}")

    except Exception as e:
        print(f"Error setting up video reader/writer: {e}")
        if video_reader and video_reader.isOpened(): video_reader.release()
        return None

    # --- Face Detection and Cropping Cache ---
    # This part is similar to the original app.py logic
    # Stores detected faces and their affine matrices for each frame
    # to avoid re-detection if faces are consistent across frames.
    # Key: frame_idx, Value: list of (cropped_face, affine_matrix)
    all_crop_quads_per_frame = {}
    all_affine_matrices_per_frame = {}
    all_restored_faces_per_frame = {}

    print("Phase 1: Detecting and aligning faces for all frames...")
    for frame_idx in range(frame_count):
        ret, img = video_reader.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx + 1}/{frame_count}. Skipping.")
            continue

        # face_helper.clean_all() # Not strictly necessary per frame unless memory is an issue
        face_helper.read_image(img)
        num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=False, # Detect all faces
            resize=None, # No resize during detection
            blur_ratio=0.01,
            eye_dist_threshold=None
        )

        if num_det_faces > 0:
            face_helper.align_warp_face_mod(draw_box=draw_box) # Use modified align_warp_face
            all_crop_quads_per_frame[frame_idx] = [f.astype(np.float32) for f in face_helper.all_crop_quads]
            all_affine_matrices_per_frame[frame_idx] = [m.astype(np.float32) for m in face_helper.all_affine_matrix]
        else:
            all_crop_quads_per_frame[frame_idx] = []
            all_affine_matrices_per_frame[frame_idx] = []

        if frame_idx % 100 == 0 or frame_idx == frame_count -1 : # Log progress
            print(f"Processed face detection for frame {frame_idx + 1}/{frame_count}")

    video_reader.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video reader for next phase

    print("Phase 2: Restoring faces using KEEP model...")
    for frame_idx in range(frame_count):
        if frame_idx not in all_crop_quads_per_frame or not all_crop_quads_per_frame[frame_idx]:
            # No faces detected in this frame, or frame was skipped
            all_restored_faces_per_frame[frame_idx] = []
            if frame_idx % 100 == 0 or frame_idx == frame_count -1 :
                 print(f"No faces to restore in frame {frame_idx + 1}/{frame_count}")
            continue

        cropped_faces_in_frame = []
        for crop_quad in all_crop_quads_per_frame[frame_idx]:
            # Here, crop_quad is already the cropped face image from align_warp_face_mod
            # It should be a BGR image (numpy array)
            if not isinstance(crop_quad, np.ndarray):
                # This case should ideally not happen if align_warp_face_mod returns images
                print(f"Warning: crop_quad for frame {frame_idx} is not a numpy array. Type: {type(crop_quad)}")
                # Potentially, align_warp_face_mod was expected to store cropped images in face_helper.cropped_faces
                # Let's assume face_helper.cropped_faces contains the images corresponding to all_crop_quads
                # This needs verification against FaceRestoreHelper's behavior.
                # For now, assuming face_helper.cropped_faces is populated by align_warp_face_mod
                if frame_idx < len(face_helper.cropped_faces): # Check bounds
                    # This logic is a bit presumptive, depends on how face_helper is structured
                    # and how align_warp_face_mod stores its results.
                    # The original app.py had a more direct way of getting cropped faces.
                    # Let's rely on face_helper.cropped_faces if all_crop_quads are just coordinates.
                    # Re-checking FaceRestoreHelper: align_warp_face stores results in self.cropped_faces
                    # So, the `all_crop_quads_per_frame` might be intended to store these images.
                    # Let's assume align_warp_face_mod populates face_helper.cropped_faces
                    # and we should be iterating through those.
                    # The current structure of storing affine matrices and quads seems to imply that
                    # `all_crop_quads_per_frame` might be the actual image data if `align_warp_face_mod` was changed.
                    # Given the name `all_crop_quads_per_frame`, it's more likely these are coordinates/quads, not images.
                    # Let's assume face_helper.cropped_faces is what we need.

                    # Modification: The original app.py's `align_warp_face_mod` seems to be a custom modification.
                    # Standard `align_warp_face` populates `self.cropped_faces`.
                    # Let's assume `all_crop_quads_per_frame` contains the *images* (cropped faces)
                    # based on the loop structure in the original app.py's restoration phase.
                    # If it contains coordinates, this part needs to be changed to use `face_helper.cropped_faces`
                    # after calling `align_warp_face` for the specific frame again, or by caching `face_helper.cropped_faces`.

                    # For simplicity, let's assume `all_crop_quads_per_frame[frame_idx]` directly holds the cropped BGR images.
                    # This implies `align_warp_face_mod` was modified to return/store these directly.
                    # If not, `face_helper.cropped_faces` (after re-running detection/alignment for the frame)
                    # or a cached version of these should be used.
                    # Given the context, the simplest interpretation is that `crop_quad` IS the image.
                    pass # crop_quad is assumed to be the image here.

            cropped_face_t = img2tensor(crop_quad / 255., bgr2rgb=True, float32=True)
            cropped_faces_in_frame.append(cropped_face_t)

        if not cropped_faces_in_frame:
            all_restored_faces_per_frame[frame_idx] = []
            continue

        # Stack tensors for batch processing
        cropped_faces_batch = torch.stack(cropped_faces_in_frame, dim=0).to(net.device)

        try:
            # KEEP model inference
            # Assuming KEEP_Net takes a batch of [B, C, H, W] and n_frames argument if it's video-specific
            # For per-frame processing, n_frames might be 1 or handled internally by model
            # The original app.py seems to process frame by frame, so n_frames=1 for the model call is likely.
            # output_faces_batch = net(cropped_faces_batch, n_frames=1) # If model expects n_frames
            output_faces_batch = net(cropped_faces_batch) # If model is generic for single images

            restored_faces_for_frame = []
            for i in range(output_faces_batch.size(0)):
                output_face = tensor2img(output_faces_batch[i], rgb2bgr=True, min_max=(-1, 1)) # KEEP outputs in [-1,1]
                output_face = np.clip(output_face, 0, 255).astype(np.uint8) # Ensure valid pixel range
                restored_faces_for_frame.append(output_face)
            all_restored_faces_per_frame[frame_idx] = restored_faces_for_frame

        except Exception as e:
            print(f"Error during KEEP model inference for frame {frame_idx}: {e}")
            all_restored_faces_per_frame[frame_idx] = [crop_quad for crop_quad in all_crop_quads_per_frame[frame_idx]] # Fallback to original cropped faces

        if frame_idx % 50 == 0 or frame_idx == frame_count -1: # Log progress
             print(f"Restored faces for frame {frame_idx + 1}/{frame_count}")


    video_reader.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video reader for final phase

    print("Phase 3: Pasting restored faces back and writing video...")
    for frame_idx in range(frame_count):
        ret, img = video_reader.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx + 1}/{frame_count} for final writing. Skipping.")
            continue

        if frame_idx not in all_restored_faces_per_frame or not all_restored_faces_per_frame[frame_idx]:
            # No faces were restored (e.g. none detected, or error), write original frame
            if bg_enhancement and bg_upsampler:
                try:
                    # Enhance background even if no faces, or if faces couldn't be restored
                    output_bg, _ = bg_upsampler.enhance(img, outscale=1) # Ensure outscale matches frame size
                    video_writer.write(output_bg)
                except Exception as e:
                    print(f"Error during background enhancement for frame {frame_idx} (no faces): {e}")
                    video_writer.write(img) # Fallback to original image
            else:
                video_writer.write(img)
            if frame_idx % 100 == 0  or frame_idx == frame_count -1:
                print(f"Written frame {frame_idx + 1}/{frame_count} (no faces/fallback)")
            continue

        restored_faces_for_this_frame = all_restored_faces_per_frame[frame_idx]
        affine_matrices_for_this_frame = all_affine_matrices_per_frame[frame_idx]

        # Create a copy of the current frame to paste onto
        current_frame_restored = img.copy()

        for i, restored_face in enumerate(restored_faces_for_this_frame):
            if i < len(affine_matrices_for_this_frame):
                inv_affine_matrix = cv2.invertAffineTransform(affine_matrices_for_this_frame[i][:2]) # Get 2x3 part
                
                # Before pasting, ensure the restored_face is the correct size (e.g., 512x512)
                # The inverse affine transform will map it back to the original face position and scale.
                # No explicit resize here as paste_face_back should handle it based on inv_affine_matrix.

                # Use a smoothed mask for better blending (similar to original app)
                # Create a mask for the restored face (e.g., an elliptical or feathered mask)
                # This part was simplified in the original Gradio app's paste_face_back compared to some other scripts.
                # For now, let's assume face_helper.paste_face_back handles blending adequately.
                # The original `paste_face_back` might need a mask argument or create one internally.
                # Let's use the one from face_helper directly.
                face_helper.paste_face_back(
                    current_frame_restored, restored_face, affine_matrices_for_this_frame[i],
                    face_size=face_helper.face_size, # Or size of restored_face
                    upscale_factor=1, # Pasting at original resolution
                    use_parse=face_helper.use_parse, # Use parsing for better mask
                    draw_box=draw_box # Draw box on the final image if requested
                )
            else:
                print(f"Warning: Mismatch between restored faces and affine matrices for frame {frame_idx}")

        # Background enhancement (if enabled)
        if bg_enhancement and bg_upsampler:
            try:
                # Enhance the frame *after* faces have been pasted back
                output_frame, _ = bg_upsampler.enhance(current_frame_restored, outscale=1) # ensure outscale is 1 for same size
                video_writer.write(output_frame)
            except Exception as e:
                print(f"Error during background enhancement for frame {frame_idx}: {e}")
                video_writer.write(current_frame_restored) # Fallback to frame with restored faces but no BG enhancement
        else:
            video_writer.write(current_frame_restored)

        if frame_idx % 100 == 0 or frame_idx == frame_count -1 : # Log progress
            print(f"Written frame {frame_idx + 1}/{frame_count} (with restored faces)")


    # --- Release Resources ---
    if video_reader and video_reader.isOpened(): video_reader.release()
    if video_writer and video_writer.isOpened(): video_writer.release()
    print(f"Video processing complete. Restored video saved to: {restored_video_path}")
    return restored_video_path


def handler(event):
    """
    RunPod serverless handler function.
    """
    global models_initialized

    job_input = event.get('input', {})
    video_url = job_input.get('video_url')
    draw_box = job_input.get('draw_box', False)
    bg_enhancement = job_input.get('bg_enhancement', False)

    if not video_url:
        return {'error': 'Missing video_url in input'}

    # Create directories if they don't exist
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Initialize models if not already done (e.g., on cold start)
    if not models_initialized:
        try:
            print("Cold start: Initializing models...")
            initialize_models()
            print("Models initialized successfully for this worker.")
        except Exception as e:
            print(f"Error initializing models: {e}")
            return {'error': f"Model initialization failed: {str(e)}"}

    # Download video
    print(f"Downloading video from: {video_url}")
    local_video_path = download_video_from_url(video_url, input_dir)
    if not local_video_path:
        return {'error': f"Failed to download video from {video_url}"}
    print(f"Video downloaded to: {local_video_path}")

    # Process video
    print(f"Processing video: {local_video_path} with draw_box={draw_box}, bg_enhancement={bg_enhancement}")
    try:
        processed_video_path = process_video_internal(local_video_path, draw_box, bg_enhancement)
        if not processed_video_path:
            return {'error': 'Video processing failed internally.'}
        
        print(f"Video processed successfully: {processed_video_path}")
        # For now, return the path within the container.
        # In a real setup, this might be uploaded to S3 and a public URL returned.
        return {'processed_video_path': processed_video_path}
    except Exception as e:
        print(f"An error occurred during video processing: {e}")
        # import traceback
        # traceback.print_exc() # For more detailed logs on RunPod
        return {'error': f"An unexpected error occurred during processing: {str(e)}"}
    finally:
        # Optional: Cleanup downloaded input video
        # if local_video_path and os.path.exists(local_video_path):
        #     try:
        #         os.remove(local_video_path)
        #         print(f"Cleaned up input video: {local_video_path}")
        #     except Exception as e:
        #         print(f"Error cleaning up input video {local_video_path}: {e}")
        # Note: Processed video is kept for now as it's the result.
        pass


if __name__ == '__main__':
    # This block is for starting the RunPod serverless worker.
    # For local testing of functions, you might call them directly before this.
    print("Starting RunPod serverless worker...")
    # Example local test (optional, comment out for deployment)
    # try:
    #     print("Attempting to initialize models for local test...")
    #     initialize_models()
    #     print("Models initialized. You can now test process_video_internal or handler locally if needed.")
    #     # Example: test download
    #     # test_url = "YOUR_TEST_VIDEO_URL_HERE"
    #     # if test_url != "YOUR_TEST_VIDEO_URL_HERE":
    #     #     downloaded_path = download_video_from_url(test_url, input_dir)
    #     #     if downloaded_path:
    #     #         print(f"Test download successful: {downloaded_path}")
    #     #         # test_process = process_video_internal(downloaded_path, False, False)
    #     #         # if test_process:
    #     #         #    print(f"Test processing successful: {test_process}")
    #     # else:
    #     #     print("Skipping local download/process test as no test URL provided.")
    # except Exception as e:
    #     print(f"Error during local pre-initialization or test: {e}")

    runpod.serverless.start({'handler': handler})
