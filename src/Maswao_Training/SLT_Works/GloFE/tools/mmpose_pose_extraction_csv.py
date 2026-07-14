# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv
import pandas as pd

from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from tqdm import tqdm
import numpy as np
import pickle as pkl


def main():
    """Visualize the demo video (support both single-frame and multi-frame).

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--csv-file', type=str, required=True, 
                        help='Path to CSV file containing Segment ID and video ID columns')
    parser.add_argument('--video-root', type=str, required=True,
                        help='Root directory containing video files')
    parser.add_argument('--output-root', type=str, required=True,
                        help='Root directory for output pickle files')
    parser.add_argument('--segment-col', type=str, default='Segment ID',
                        help='Name of the Segment ID column in CSV')
    parser.add_argument('--video-col', type=str, default='video ID',
                        help='Name of the video ID column in CSV')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--sid',
        type=int,
        default=0,
        help='Split ID for parallel processing')
    parser.add_argument(
        '--splits',
        type=int,
        default=1,
        help='Total number of splits for parallel processing')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.det_config is not None
    assert args.det_checkpoint is not None

    print('Initializing model...')
    # build the detection model from a config file and a checkpoint file
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    # get datasetinfo
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # Create output directory
    os.makedirs(args.output_root, exist_ok=True)

    arg_dict = {
        'det_model': det_model,
        'pose_model': pose_model,
        'dataset': dataset,
        'dataset_info': dataset_info,
        'output_root': args.output_root,
        'video_root': args.video_root,
        'args': args,
    }

    # Load video paths from CSV
    all_samples = load_videos_from_csv(
        args.csv_file, 
        args.video_root,
        args.segment_col,
        args.video_col
    )
    
    total_samples = len(all_samples)
    print(f'Total samples: {total_samples}')
    
    # Split samples for parallel processing
    chunk = (total_samples + args.splits - 1) // args.splits
    start_idx = args.sid * chunk
    end_idx = min((args.sid + 1) * chunk, total_samples)
    sample_split = all_samples[start_idx:end_idx]
    print(f'Running split:[{start_idx}:{end_idx}]')

    # Process each video
    for sample_info in tqdm(sample_split):
        video_path = sample_info['video_path']
        output_name = sample_info['output_name']
        output_file_path = os.path.join(args.output_root, f'{output_name}.pkl')
        
        if os.path.exists(output_file_path):
            print(f'Skipping {output_name} - already processed')
            continue
            
        if not os.path.exists(video_path):
            print(f'Video not found: {video_path}')
            with open(f'log-missing-videos-s{args.sid}.txt', 'a') as f:
                f.write(f'{video_path}\n')
            continue
            
        process_single_video(video_path, output_name, arg_dict)


def load_videos_from_csv(csv_path, video_root, segment_col, video_col):
    """Load video paths from CSV file.
    
    Args:
        csv_path: Path to CSV file
        video_root: Root directory for videos
        segment_col: Column name for Segment ID
        video_col: Column name for video ID
        
    Returns:
        List of dictionaries with video_path and output_name
    """
    df = pd.read_csv(csv_path)
    
    # Check if required columns exist
    if segment_col not in df.columns:
        raise ValueError(f"Column '{segment_col}' not found in CSV. Available columns: {df.columns.tolist()}")
    if video_col not in df.columns:
        raise ValueError(f"Column '{video_col}' not found in CSV. Available columns: {df.columns.tolist()}")
    
    samples = []
    for idx, row in df.iterrows():
        segment_id = str(row[segment_col])
        video_id = str(row[video_col])
        
        # Construct video path: root_path/Segment_ID/segment_in_ratio/{segment_id}_{video_id}.mp4
        video_path = os.path.join(
            video_root,
            segment_id,
            'segment_gradients_sent_ratios_1',
            f'segment_{video_id}_{segment_id}.mp4'
        )
        print(video_path)
        # Use segment_id_video_id as output name
        output_name = f'segment_{video_id}_{segment_id}'
        
        samples.append({
            'video_path': video_path,
            'output_name': output_name,
            'segment_id': segment_id,
            'video_id': video_id
        })
    
    print(f'Loaded {len(samples)} samples from {csv_path}')
    return samples


def process_single_video(video_path, output_name, arg_dict):
    """Process a single video and extract pose keypoints.
    
    Args:
        video_path: Path to video file
        output_name: Name for output file (without extension)
        arg_dict: Dictionary containing models and arguments
    """
    det_model = arg_dict['det_model']
    pose_model = arg_dict['pose_model']
    dataset = arg_dict['dataset']
    dataset_info = arg_dict['dataset_info']
    output_root = arg_dict['output_root']
    args = arg_dict['args']
    
    # Read video
    video = mmcv.VideoReader(video_path)
    if not video.opened:
        print(f'Failed to load video file {video_path}')
        with open(f'log-failed-videos-s{args.sid}.txt', 'a') as f:
            f.write(f'{video_path}\n')
        return

    # Frame index offsets for inference, used in multi-frame inference setting
    if args.use_multi_frames:
        assert 'frame_indices_test' in pose_model.cfg.data.test.data_cfg
        indices = pose_model.cfg.data.test.data_cfg['frame_indices_test']

    # Whether to return heatmap, optional
    return_heatmap = False

    # Return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    results = []
    
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        # Get the detection results of current frame
        # The resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, cur_frame)

        # Keep the person class bounding boxes
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        if args.use_multi_frames:
            frames = collect_multi_frames(video, frame_id, indices, args.online)

        # Test a single image, with a list of bboxes
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            frames if args.use_multi_frames else cur_frame,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if len(pose_results) != 0:
            pose_results = pose_results[0]
            results.append(pose_results['keypoints'])
        else:
            print(f'{output_name} Frame:{frame_id} has no person')
            with open(f'log-no-person-s{args.sid}.txt', 'a') as f:
                f.write(f'{output_name} Frame:{frame_id} has no person\n')

    results = np.array(results)
    
    # Save results
    output_file = os.path.join(output_root, f'{output_name}.pkl')
    with open(output_file, 'wb') as f:
        if len(results.shape) == 3:
            pkl.dump(results, f)
            print(f'Saved: {output_file} with shape {results.shape}')
        else:
            print(f'{output_name} Incorrect result shape: {results.shape}')
            with open(f'log-incorrect-shape-s{args.sid}.txt', 'a') as f:
                f.write(f'{output_name} Incorrect result shape: {results.shape}\n')


if __name__ == '__main__':
    main()
