import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from typing import List, Tuple

# Set modern styling
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

def get_motion_sequence(video_path):
    """Compute frame-to-frame motion magnitude using optical flow."""
    cap = cv2.VideoCapture(str(video_path))
    
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return np.array([])
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_values = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Compute motion magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        motion_score = np.mean(magnitude)
        motion_values.append(motion_score)
        
        prev_gray = gray
    
    cap.release()
    return np.array(motion_values)

def trim_trailing_zeros(values):
    """
    Trims consecutive zeros from the end of a 1D array or list.
    Keeps all leading/middle zeros; only trims the uninterrupted zeros at the end.
    """
    values = np.asarray(values)
    
    # Find index of last nonzero element
    nonzero_indices = np.nonzero(values)[0]
    
    if len(nonzero_indices) == 0:
        # All zeros — return empty array
        return np.array([])
    
    last_nonzero_idx = nonzero_indices[-1]
    return values[:last_nonzero_idx + 1]

def get_sentence_lengths(token_file):
    """
    Read sentences and return both count and character lengths.
    
    Returns:
        tuple: (sentence_count, list of sentence lengths in characters)
    """
    with open(token_file, "r", encoding="utf-8") as f:
        sentences = [l.strip() for l in f if l.strip()]
    
    sentence_lengths = [len(s) for s in sentences]
    return len(sentences), sentence_lengths

def calculate_gradient_score(motion_values, index, window_size=3):
    """
    Calculate gradient score for a potential breakpoint.
    Higher score = steeper descent before + steeper ascent after.
    
    Args:
        motion_values: Array of motion scores
        index: Candidate breakpoint index
        window_size: Number of frames to look before/after
    
    Returns:
        Gradient score (higher = better breakpoint)
    """
    start = max(0, index - window_size)
    end = min(len(motion_values), index + window_size + 1)
    
    # Calculate gradient before the point (should be negative/descending)
    if index > start:
        before_window = motion_values[start:index]
        if len(before_window) > 1:
            grad_before = np.mean(np.diff(before_window))
        else:
            grad_before = 0
    else:
        grad_before = 0
    
    # Calculate gradient after the point (should be positive/ascending)
    if index < end - 1:
        after_window = motion_values[index + 1:end]
        if len(after_window) > 1:
            grad_after = np.mean(np.diff(after_window))
        else:
            grad_after = 0
    else:
        grad_after = 0
    
    # Score: want steep descent before (negative grad_before) and steep ascent after (positive grad_after)
    score = -grad_before + grad_after
    
    return score

def find_gradient_based_breakpoints(
    motion_values, 
    sentence_lengths,
    min_spacing_seconds=0.5, 
    fps=30,
    gradient_window=3
):
    """
    Find breakpoints using gradient analysis and sentence length ratios.
    
    Args:
        motion_values: Array of motion scores per frame
        sentence_lengths: List of sentence lengths in characters
        min_spacing_seconds: Minimum seconds between breakpoints
        fps: Video frame rate
        gradient_window: Window size for gradient calculation
    
    Returns:
        Array of frame indices for breakpoints
    """
    N = len(sentence_lengths)
    total_frames = len(motion_values)
    
    if total_frames < N:
        raise ValueError(f"Not enough frames ({total_frames}) for {N} breakpoints")
    
    min_spacing_frames = int(min_spacing_seconds * fps)
    
    # CRITICAL: Reserve space for the last segment
    # The last breakpoint must leave enough frames for the final segment
    min_last_segment_frames = min_spacing_frames  # Ensure last segment has minimum duration
    max_breakpoint_position = total_frames - min_last_segment_frames
    
    print(f"🎯 Max breakpoint position: {max_breakpoint_position} (leaving {min_last_segment_frames} frames for last segment)")
    
    # Calculate sentence ratios (proportion of total content)
    total_chars = sum(sentence_lengths)
    sentence_ratios = [length / total_chars for length in sentence_lengths]
    
    print(f"📊 Sentence ratios: {[f'{r:.3f}' for r in sentence_ratios]}")
    
    # FIXED: Calculate expected frame positions based on cumulative sentence ratios
    # We need N-1 breakpoints to create N segments
    # Each breakpoint should be at the END of each sentence (except the last)
    # Scale expected positions to fit within the allowed range
    expected_positions = []
    cumulative_ratio = 0
    
    for i in range(N - 1):  # Only need N-1 breakpoints
        cumulative_ratio += sentence_ratios[i]  # Add current sentence ratio
        # Scale to max_breakpoint_position instead of total_frames
        expected_frame = int(cumulative_ratio * max_breakpoint_position)
        expected_positions.append(expected_frame)
    
    print(f"🎯 Expected breakpoint positions: {expected_positions}")
    print(f"🎯 This creates {N} segments with the last segment from frame {expected_positions[-1] if expected_positions else 0} to {total_frames}")
    
    # For each expected position, find the best valley within a search window
    search_window = int(fps * 3)  # 3 second search window around expected position
    
    breakpoints = []
    for idx, expected_pos in enumerate(expected_positions):
        # Define search range
        search_start = max(gradient_window, expected_pos - search_window)
        search_end = min(expected_pos + search_window, max_breakpoint_position - gradient_window)
        
        # CRITICAL: For the last breakpoint, ensure it stays well within bounds
        if idx == len(expected_positions) - 1:
            # Last breakpoint must leave room for final segment
            search_end = min(search_end, max_breakpoint_position - min_spacing_frames)
            print(f"🔒 Last breakpoint search range: {search_start} to {search_end} (max allowed: {max_breakpoint_position})")
        
        # Find local minima in search range
        candidates = []
        for i in range(search_start, search_end):
            # Check if it's a local minimum
            if i > 0 and i < len(motion_values) - 1:
                if motion_values[i] < motion_values[i-1] and motion_values[i] < motion_values[i+1]:
                    # Calculate combined score: low motion value + high gradient score
                    grad_score = calculate_gradient_score(motion_values, i, gradient_window)
                    
                    # Normalize motion value (invert so lower is better)
                    motion_norm = 1.0 / (motion_values[i] + 0.01)  # Avoid division by zero
                    
                    # Distance penalty (prefer points closer to expected position)
                    distance = abs(i - expected_pos)
                    distance_penalty = 1.0 / (1.0 + distance / search_window)
                    
                    # Combined score
                    combined_score = grad_score * 0.4 + motion_norm * 0.3 + distance_penalty * 0.3
                    
                    candidates.append((combined_score, grad_score, motion_values[i], i))
        
        if candidates:
            # Sort by combined score (highest first)
            candidates.sort(reverse=True, key=lambda x: x[0])
            
            # Select best candidate that respects minimum spacing
            for combined_score, grad_score, motion_val, frame_idx in candidates:
                # Check spacing from previous breakpoints
                if all(abs(frame_idx - bp) >= min_spacing_frames for bp in breakpoints):
                    # CRITICAL: Ensure breakpoint doesn't exceed max position
                    if frame_idx <= max_breakpoint_position:
                        breakpoints.append(frame_idx)
                        frames_remaining = total_frames - frame_idx
                        print(f"✅ Breakpoint {len(breakpoints)}: frame {frame_idx} "
                              f"(expected: {expected_pos}, offset: {frame_idx-expected_pos}, "
                              f"grad_score: {grad_score:.2f}, motion: {motion_val:.2f}, "
                              f"leaves {frames_remaining} frames for remaining segments)")
                        break
            else:
                print(f"⚠️ No valid candidate found for expected position {expected_pos}")
        else:
            print(f"⚠️ No local minima found near expected position {expected_pos}")
    
    if len(breakpoints) < N - 1:
        print(f"⚠️ Only found {len(breakpoints)} breakpoints out of {N-1} needed")
    
    return np.sort(np.array(breakpoints))

def split_video_with_ratios(video_path, break_indices, output_dir, video_id, sentence_lengths):
    """
    Split video into segments and report actual vs expected durations.
    FIXED: Ensures last segment is properly written by reopening video for each segment.
    """
    print(f"🔪 Splitting video at frames: {break_indices.tolist()}")
    
    segment_dir = os.path.join(output_dir, "segment_gradients_sent_ratios_1")
    os.makedirs(segment_dir, exist_ok=True)
    
    # Get video properties first
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Calculate expected durations
    total_chars = sum(sentence_lengths)
    total_duration = total_frames / fps
    expected_durations = [(length / total_chars) * total_duration for length in sentence_lengths]
    
    boundaries = [0] + list(break_indices) + [total_frames]
    segment_paths = []
    
    print(f"\n📹 Segment Analysis:")
    print(f"{'Seg':<4} {'Frames':<12} {'Duration':<10} {'Expected':<10} {'Ratio':<8} {'Chars':<6}")
    print("-" * 60)
    
    # CRITICAL FIX: Process each segment independently by reopening the video
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        segment_file = os.path.join(segment_dir, f"segment_{i+1}_{video_id}.mp4")
        
        # Reopen video for each segment to avoid seek issues
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Failed to open video for segment {i+1}")
            continue
        
        # Create video writer
        out = cv2.VideoWriter(str(segment_file), fourcc, fps, (width, height))
        
        # Set position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        frames_written = 0
        expected_frames = end - start
        
        # Read and write frames for this segment
        for frame_idx in range(start, end):
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ Warning: Could not read frame {frame_idx} for segment {i+1}")
                break
            out.write(frame)
            frames_written += 1
        
        out.release()
        cap.release()  # Close the video after each segment
        
        # Verify the segment was written correctly
        if frames_written != expected_frames:
            print(f"⚠️ Segment {i+1}: Expected {expected_frames} frames, wrote {frames_written}")
        
        segment_paths.append(segment_file)
        
        # Report segment statistics
        actual_duration = frames_written / fps
        expected_duration = expected_durations[i]
        ratio = actual_duration / expected_duration if expected_duration > 0 else 0
        chars = sentence_lengths[i]
        
        print(f"{i+1:<4} {start:>5}-{end:<5} {actual_duration:>7.2f}s  {expected_duration:>7.2f}s  {ratio:>6.2f}x  {chars:>5}")
    
    print(f"\n💾 Saved {len(segment_paths)} segments to {segment_dir}/")
    return segment_paths

def plot_enhanced_motion(motion_values, break_indices, expected_positions, 
                         output_path, video_id, gradient_window=3):
    """
    Create enhanced plot with modern Seaborn styling showing motion, breakpoints, and gradient scores.
    """
    # Create figure with custom styling
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1], hspace=0.3)
    
    # Color palette
    colors = {
        'motion': '#2E86AB',      # Deep blue
        'gradient': '#06A77D',     # Teal green
        'breakpoint': '#D62828',   # Red
        'expected': '#F77F00',     # Orange
        'fill': '#A4C3B2'          # Sage green
    }
    
    # Plot 1: Motion values with area fill
    ax1 = fig.add_subplot(gs[0])
    
    # Create smoothed line using rolling average for visual appeal
    window = 5
    if len(motion_values) >= window:
        motion_smooth = np.convolve(motion_values, np.ones(window)/window, mode='same')
    else:
        motion_smooth = motion_values
    
    # Fill area under curve
    ax1.fill_between(range(len(motion_values)), motion_values, alpha=0.3, 
                     color=colors['fill'], label='Motion area')
    
    # Main motion line
    ax1.plot(motion_values, color=colors['motion'], linewidth=2.5, 
             label='Motion magnitude', alpha=0.8)
    
    # Plot smoothed line
    if len(motion_values) >= window:
        ax1.plot(motion_smooth, color=colors['motion'], linewidth=1.5, 
                linestyle='--', alpha=0.5, label='Smoothed trend')
    
    # Expected positions as vertical spans
    if expected_positions:
        for i, pos in enumerate(expected_positions):
            ax1.axvline(x=pos, color=colors['expected'], linestyle=':', 
                       alpha=0.6, linewidth=2, zorder=3)
            ax1.text(pos, ax1.get_ylim()[1] * 0.95, f'E{i+1}', 
                    ha='center', va='top', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['expected'], 
                             alpha=0.3, edgecolor='none'))
    
    # Selected breakpoints
    ax1.scatter(break_indices, motion_values[break_indices], 
                color=colors['breakpoint'], s=200, zorder=5, 
                label='Selected breakpoints', marker='v', 
                edgecolors='white', linewidths=2)
    
    # Add segment regions
    boundaries = [0] + list(break_indices) + [len(motion_values)]
    for i in range(len(boundaries) - 1):
        mid_point = (boundaries[i] + boundaries[i+1]) / 2
        ax1.axvspan(boundaries[i], boundaries[i+1], alpha=0.05, 
                   color='gray' if i % 2 == 0 else 'lightgray', zorder=0)
        ax1.text(mid_point, ax1.get_ylim()[1] * 0.05, f'S{i+1}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         alpha=0.8, edgecolor='gray'))
    
    ax1.set_ylabel("Motion Magnitude", fontsize=12, fontweight='bold')
    # ax1.set_title(f"Motion Detection Analysis with Gradient-Based Breakpoints\nVideo ID: {video_id}", 
    #              fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    sns.despine(ax=ax1, left=False, bottom=False)
    
    # Plot 2: Gradient scores with highlighting
    ax2 = fig.add_subplot(gs[1])
    
    gradient_scores = []
    for i in range(len(motion_values)):
        score = calculate_gradient_score(motion_values, i, gradient_window)
        gradient_scores.append(score)
    
    gradient_scores = np.array(gradient_scores)
    
    # Create color gradient based on score intensity
    norm_scores = (gradient_scores - gradient_scores.min()) / (gradient_scores.max() - gradient_scores.min() + 1e-6)
    
    # Plot with gradient coloring
    ax2.plot(gradient_scores, color=colors['gradient'], linewidth=2.5, 
             label='Gradient score', alpha=0.8)
    
    # Highlight high-score regions
    threshold = np.percentile(gradient_scores, 75)
    high_score_mask = gradient_scores > threshold
    ax2.fill_between(range(len(gradient_scores)), 0, gradient_scores,
                     where=high_score_mask, alpha=0.2, color=colors['gradient'],
                     label=f'High gradient (>75th percentile)')
    
    # Selected breakpoints
    ax2.scatter(break_indices, [gradient_scores[i] for i in break_indices],
                color=colors['breakpoint'], s=200, zorder=5, 
                label='Selected breakpoints', marker='v',
                edgecolors='white', linewidths=2)
    
    # Add horizontal line at mean
    mean_gradient = np.mean(gradient_scores)
    ax2.axhline(y=mean_gradient, color='gray', linestyle='--', 
               alpha=0.5, linewidth=1.5, label=f'Mean ({mean_gradient:.2f})')
    
    ax2.set_ylabel("Gradient Score", fontsize=12, fontweight='bold')
    ax2.set_title("Gradient Analysis (Valley Detection)", fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', framealpha=0.95, fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    sns.despine(ax=ax2, left=False, bottom=False)
    
    # Plot 3: Statistics summary
    # ax3 = fig.add_subplot(gs[2])
    # ax3.axis('off')
    
    # Prepare statistics
    # stats_text = f"""
    # 📊 Analysis Summary:
    # • Total Frames: {len(motion_values):,} | Breakpoints Found: {len(break_indices)} | Expected: {len(expected_positions)}
    # • Motion Range: [{motion_values.min():.2f}, {motion_values.max():.2f}] | Mean: {motion_values.mean():.2f} | Std: {motion_values.std():.2f}
    # • Gradient Range: [{gradient_scores.min():.2f}, {gradient_scores.max():.2f}] | Mean: {mean_gradient:.2f}
    # • Breakpoint Frames: {', '.join([str(bp) for bp in break_indices])}
    # """
    
    # ax3.text(0.05, 0.5, stats_text, transform=ax3.transAxes,
    #         fontsize=10, verticalalignment='center', family='monospace',
    #         bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', 
    #                  edgecolor='#dee2e6', linewidth=2))
    
    # Add xlabel to bottom plot
    ax2.set_xlabel("Frame Number", fontsize=12, fontweight='bold')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📊 Saved enhanced Seaborn plot: {output_path}")

def process_video(video_path, token_path):
    """Process videos with gradient-based splitting and sentence ratios."""
    
    for subdir, dirs, files in os.walk(token_path):
        video_id = os.path.basename(subdir)
        if not video_id:
            continue
        
        print(f"\n{'='*70}")
        print(f"🎥 Processing video ID: {video_id}")
        print('='*70)
        
        # Find token file
        token_file = None
        for file in files:
            if file.endswith("clean.txt"):
                token_file = os.path.join(subdir, file)
                break
        
        if not token_file:
            continue
        
        # Get sentence information
        N, sentence_lengths = get_sentence_lengths(token_file)
        print(f"🧾 Found {N} sentences")
        print(f"📏 Sentence lengths (chars): {sentence_lengths}")
        
        # Find video file
        video_file = None
        video_subdir = None
        
        for vid_subdir, vid_dirs, vid_files in os.walk(video_path):
            if video_id in os.path.basename(vid_subdir):
                print(f"⏳ Found video directory: {vid_subdir}")
                
                for video_file_name in vid_files:
                    if video_file_name.startswith("interpreter"):
                        print(f"🎬 Found video file: {video_file_name}")
                        video_file = os.path.join(vid_subdir, video_file_name)
                        video_subdir = vid_subdir
                        break
                
                if video_file:
                    break
        
        if not video_file:
            print(f"⚠️ No interpreter video found for {video_id}")
            continue
        
        # Get video properties
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"🎞️  Video: {total_frames} frames @ {fps:.2f} fps = {total_frames/fps:.2f}s")
        
        # Calculate motion sequence
        print(f"⏳ Calculating motion sequence...")
        motion_values = get_motion_sequence(video_file)
        motion_values  = trim_trailing_zeros(motion_values)
        
        # Save motion data
        motion_file = os.path.join(video_subdir, f"motion_{video_id}_sent_ratios_1.npy")
        # np.save(motion_file, motion_values)
        print(f"💾 Saved motion data: {motion_file}")
        
        # Check if we have enough frames
        if len(motion_values) < N:
            print(f"⚠️ Not enough motion points to split ({len(motion_values)} < {N})")
            continue
        
        # Find breakpoints using gradient analysis and sentence ratios
        print(f"\n⏳ Finding {N-1} breakpoints using gradient analysis and sentence ratios...")
        
        try:
            break_indices = find_gradient_based_breakpoints(
                motion_values, 
                sentence_lengths,
                min_spacing_seconds=0.5,
                fps=fps,
                gradient_window=3
            )
            
            # Calculate expected positions for plotting
            total_chars = sum(sentence_lengths)
            sentence_ratios = [length / total_chars for length in sentence_lengths]
            expected_positions = []
            cumulative_ratio = 0
            for ratio in sentence_ratios[:-1]:
                cumulative_ratio += ratio
                expected_positions.append(int(cumulative_ratio * len(motion_values)))
            
            # Split video
            # split_paths = split_video_with_ratios(
            #     video_file, break_indices, video_subdir, video_id, sentence_lengths
            # )
            
            # Create enhanced plot
            plot_file = os.path.join(video_subdir, f"motion_plot_{video_id}_gradient_sent_ratios_1_enhanced.pdf")
            plot_enhanced_motion(
                motion_values, break_indices, expected_positions, 
                plot_file, video_id
            )
            
            print(f"\n✅ Successfully processed {video_id}")
            
        except Exception as e:
            print(f"❌ Error processing {video_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

# Example usage
if __name__ == "__main__":
    video_path = "../../../M2M-MT-files/Youtube_Downloads_2/English"
    token_path = "../../../M2M-MT-files/Youtube_Download_2_Transcription_Whisper/"
    process_video(video_path, token_path)
    print("\n🎉 Processing completed.")