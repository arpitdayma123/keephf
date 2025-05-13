import os
import cv2
import argparse
import glob
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from scipy.ndimage import gaussian_filter1d
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.utils.video_util import VideoReader, VideoWriter
from basicsr.utils.registry import ARCH_REGISTRY
import gradio as gr
from torch.hub import download_url_to_file

title = r"""<h1 align="center">KEEP: Kalman-Inspired Feature Propagation for Video Face Super-Resolution</h1>"""

description = r"""
<b>Official Gradio demo</b> for <a href='https://github.com/jnjaby/KEEP' target='_blank'><b>Kalman-Inspired FEaturE Propagation for Video Face Super-Resolution (ECCV 2024)</b></a>.<br>
🔥 KEEP is a robust video face super-resolution algorithm.<br>
🤗 Try to drop your own face video, and get the restored results!<br>
"""

post_article = r"""
If you found KEEP helpful, please consider ⭐ the <a href='https://github.com/jnjaby/KEEP' target='_blank'>Github Repo</a>. Thanks! 
[![GitHub Stars](https://img.shields.io/github/stars/jnjaby/KEEP)](https://github.com/jnjaby/KEEP)
---
📝 **Citation**
<br>
If our work is useful for your research, please consider citing:
```bibtex
@InProceedings{feng2024keep,
      title     = {Kalman-Inspired FEaturE Propagation for Video Face Super-Resolution},
      author    = {Feng, Ruicheng and Li, Chongyi and Loy, Chen Change},
      booktitle = {European Conference on Computer Vision (ECCV)},
      year      = {2024}
}
```

📋 **License**
<br>
This project is licensed under <a rel="license" href="https://github.com/jnjaby/KEEP/blob/main/LICENSE">S-Lab License 1.0</a>. 
Redistribution and use for non-commercial purposes should follow this license.
<br><br>
📧 **Contact**
<br>
If you have any questions, please feel free to reach out via <b>ruicheng002@ntu.edu.sg</b>.
"""



def interpolate_sequence(sequence):
    interpolated_sequence = np.copy(sequence)
    missing_indices = np.isnan(sequence)
    if np.any(missing_indices):
        valid_indices = ~missing_indices
        x = np.arange(len(sequence))
        interpolated_sequence[missing_indices] = np.interp(x[missing_indices], x[valid_indices], sequence[valid_indices])
    return interpolated_sequence

def set_realesrgan():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer
    use_half = False
    if torch.cuda.is_available():
        no_half_gpu_list = ['1650', '1660']
        if not any(gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list):
            use_half = True
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(scale=2, model_path="https://github.com/jnjaby/KEEP/releases/download/v0.1.0/RealESRGAN_x2plus.pth", model=model, tile=400, tile_pad=40, pre_pad=0, half=use_half)
    if not gpu_is_available():
        import warnings
        warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA. The unoptimized RealESRGAN is slow on CPU.', category=RuntimeWarning)
    return upsampler

def process_video(input_video, draw_box, bg_enhancement):
    device = get_device()
    args = argparse.Namespace(
        input_path=input_video,
        upscale=1,
        max_length=20,
        has_aligned=False,
        only_center_face=True,
        draw_box=draw_box,
        detection_model='retinaface_resnet50',
        bg_enhancement=bg_enhancement,
        face_upsample=False,
        bg_tile=400,
        suffix=None,
        save_video_fps=None,
        model_type='KEEP'
    )

    output_dir = './results/'
    os.makedirs(output_dir, exist_ok=True)

    model_configs = {
        'KEEP': {
            'architecture': {
                'img_size': 512, 'emb_dim': 256, 'dim_embd': 512, 'n_head': 8, 'n_layers': 9,
                'codebook_size': 1024, 'cft_list': ['16', '32', '64'], 'kalman_attn_head_dim': 48,
                'num_uncertainty_layers': 3, 'cfa_list': ['16', '32'], 'cfa_nhead': 4, 'cfa_dim': 256, 'cond': 1
            },
            'checkpoint_dir': '../weights/KEEP',
            'checkpoint_url': 'https://github.com/jnjaby/KEEP/releases/download/v1.0.0/KEEP-b76feb75.pth'
        },
    }
    if args.bg_enhancement:
        bg_upsampler = set_realesrgan()
    else:
        bg_upsampler = None
    if args.face_upsample:
        face_upsampler = bg_upsampler if bg_upsampler is not None else set_realesrgan()
    else:
        face_upsampler = None

    if args.model_type not in model_configs:
        raise ValueError(f"Unknown model type: {args.model_type}. Available options: {list(model_configs.keys())}")
    config = model_configs[args.model_type]
    net = ARCH_REGISTRY.get('KEEP')(**config['architecture']).to(device)
    ckpt_path = load_file_from_url(url=config['checkpoint_url'], model_dir=config['checkpoint_dir'], progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path, weights_only=True)
    net.load_state_dict(checkpoint['params_ema'])
    net.eval()
    if not args.has_aligned:
        print(f'Face detection model: {args.detection_model}')
    if bg_upsampler is not None:
        print(f'Background upsampling: True, Face upsampling: {args.face_upsample}')
    else:
        print(f'Background upsampling: False, Face upsampling: {args.face_upsample}')
    face_helper = FaceRestoreHelper(args.upscale, face_size=512, crop_ratio=(1, 1), det_model=args.detection_model, save_ext='png', use_parse=True, device=device)

    # Reading the input video.
    input_img_list = []
    if args.input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')):
        vidreader = VideoReader(args.input_path)
        image = vidreader.get_frame()
        while image is not None:
            input_img_list.append(image)
            image = vidreader.get_frame()
        fps = vidreader.get_fps() if args.save_video_fps is None else args.save_video_fps
        vidreader.close()
        clip_name = os.path.basename(args.input_path)[:-4]
    else:
        raise TypeError(f'Unrecognized type of input video {args.input_path}.')
    if len(input_img_list) == 0:
        raise FileNotFoundError('No input image/video is found...')

    print('Detecting keypoints and smooth alignment ...')
    if not args.has_aligned:
        raw_landmarks = []
        for i, img in enumerate(input_img_list):
            face_helper.clean_all()
            face_helper.read_image(img)
            num_det_faces = face_helper.get_face_landmarks_5(only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5, only_keep_largest=True)
            if num_det_faces == 1:
                raw_landmarks.append(face_helper.all_landmarks_5[0].reshape((10,)))
            elif num_det_faces == 0:
                raw_landmarks.append(np.array([np.nan]*10))
        raw_landmarks = np.array(raw_landmarks)
        for i in range(10):
            raw_landmarks[:, i] = interpolate_sequence(raw_landmarks[:, i])
        video_length = len(input_img_list)
        avg_landmarks = gaussian_filter1d(raw_landmarks, 5, axis=0).reshape(video_length, 5, 2)
    cropped_faces = []
    for i, img in enumerate(input_img_list):
        face_helper.clean_all()
        face_helper.read_image(img)
        face_helper.all_landmarks_5 = [avg_landmarks[i]]
        face_helper.align_warp_face()
        cropped_face_t = img2tensor(face_helper.cropped_faces[0] / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_faces.append(cropped_face_t)
    cropped_faces = torch.stack(cropped_faces, dim=0).unsqueeze(0).to(device)
    print('Restoring faces ...')
    with torch.no_grad():
        video_length = cropped_faces.shape[1]
        output = []
        for start_idx in range(0, video_length, args.max_length):
            end_idx = min(start_idx + args.max_length, video_length)
            if end_idx - start_idx == 1:
                output.append(net(cropped_faces[:, [start_idx, start_idx], ...], need_upscale=False)[:, 0:1, ...])
            else:
                output.append(net(cropped_faces[:, start_idx:end_idx, ...], need_upscale=False))
        output = torch.cat(output, dim=1).squeeze(0)
        assert output.shape[0] == video_length, "Different number of frames"
        restored_faces = [tensor2img(x, rgb2bgr=True, min_max=(-1, 1)) for x in output]
        del output
        torch.cuda.empty_cache()
    print('Pasting faces back ...')

    restored_frames = []
    for i, img in enumerate(input_img_list):
        face_helper.clean_all()
        if args.has_aligned:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=10)
            if face_helper.is_gray:
                print('Grayscale input: True')
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            face_helper.all_landmarks_5 = [avg_landmarks[i]]
            face_helper.align_warp_face()
        face_helper.add_restored_face(restored_faces[i].astype('uint8'))
        if not args.has_aligned:
            if bg_upsampler is not None:
                bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            if args.face_upsample and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box, face_upsampler=face_upsampler)
            else:
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box)

        restored_frames.append(restored_img)

    # Saving the output video.
    print('Saving video ...')
    height, width = restored_frames[0].shape[:2]
    save_restore_path = os.path.join(output_dir, f'{clip_name}.mp4')
    vidwriter = VideoWriter(save_restore_path, height, width, fps)
    for f in restored_frames:
        vidwriter.write_frame(f)
    vidwriter.close()
    print(f'All results are saved in {save_restore_path}.')
    return save_restore_path

# Downloading necessary models and sample videos.
sample_videos_dir = os.path.join("/home/user/app/hugging_face/", "test_sample/")
os.makedirs(sample_videos_dir, exist_ok=True)
download_url_to_file("https://github.com/jnjaby/KEEP/releases/download/media/real_1.mp4", os.path.join(sample_videos_dir, "real_1.mp4"))
download_url_to_file("https://github.com/jnjaby/KEEP/releases/download/media/real_2.mp4", os.path.join(sample_videos_dir, "real_2.mp4"))
download_url_to_file("https://github.com/jnjaby/KEEP/releases/download/media/real_3.mp4", os.path.join(sample_videos_dir, "real_3.mp4"))
download_url_to_file("https://github.com/jnjaby/KEEP/releases/download/media/real_4.mp4", os.path.join(sample_videos_dir, "real_4.mp4"))

model_dir = os.path.join("/home/user/app/", "weights/KEEP")
_ = load_file_from_url(url='https://github.com/jnjaby/KEEP/releases/download/v1.0.0/KEEP-b76feb75.pth', model_dir=model_dir, progress=True, file_name=None)

realesrgan_dir = os.path.join("/home/user/app/", "weights/realesrgan")
_ = load_file_from_url(url='https://github.com/jnjaby/KEEP/releases/download/v1.0.0/RealESRGAN_x2plus.pth', model_dir=realesrgan_dir, progress=True, file_name=None)

# Launching the Gradio interface.
demo = gr.Interface(
    fn=process_video,
    title=title,
    description=description,
    inputs=[
        gr.Video(label="Input Video"),
        gr.Checkbox(label="Draw Box", value=False),
        gr.Checkbox(label="Background Enhancement", value=False),
    ],
    outputs=gr.Video(label="Processed Video"),
    examples=[
        [os.path.join(os.path.dirname(__file__), sample_videos_dir, "real_1.mp4"), True, False],
        [os.path.join(os.path.dirname(__file__), sample_videos_dir, "real_2.mp4"), True, False],
        [os.path.join(os.path.dirname(__file__), sample_videos_dir, "real_3.mp4"), True, False],
        [os.path.join(os.path.dirname(__file__), sample_videos_dir, "real_4.mp4"), True, False],
    ],
    article=post_article
)


demo.launch(share=True)
