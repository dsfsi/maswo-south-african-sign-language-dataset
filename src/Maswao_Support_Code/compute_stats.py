"""
Computes Video Statistics + Text Statistics for the LaTeX table, including
the two new "largest video" / "smallest video" columns (measured by duration).

WORKFLOW (per your instructions):
1. Walk the TRANSCRIPTION root first. Each subfolder under it is expected to
   contain a "<folder_name>_sentences_clean.txt" file (same convention as your
   original splitting script). This subfolder name is the canonical list of
   "videos we actually have transcriptions for" — this is our ground truth
   set, since the raw video folder count may have grown/drifted since
   transcription was done (hence mismatches if you start from videos).
2. For each such folder_name, we look for the matching video under the
   VIDEO root (../../../M2M-MT-files/Youtube_Downloads_2/English). The
   matching folder is found by walking the video root and matching
   directory name == folder_name (the script searches recursively since the
   transcription subfolder name encodes the relative path branch).
3. Inside the matched video folder there may be MULTIPLE video files
   (you confirmed this). We probe every video file we find with ffprobe,
   but only take ONE representative video per folder_name for the
   "Num. Raw Videos" / duration / resolution / fps stats, to avoid double
   counting the same talk re-encoded twice. Selection priority:
       a) a file literally named "<folder_name>.<ext>"
       b) the single largest file by duration in that folder, as a fallback
   The script PRINTS a warning every time it had to fall back to (b) or
   found multiple plausible candidates, so you can sanity check this list
   in the log it writes.
4. "Videos w/ Sign" is something we cannot auto-detect from filenames/ffprobe
   alone in general — there's no universal naming flag for this in your
   pipeline as shown. Leave VIDEOS_WITH_SIGN_LIST below populated manually
   (or point WITH_SIGN_TXT_PATH at a text file with one folder_name per
   line), otherwise this column is left as 'NA' in the output and you can
   fill it by hand.
5. Resolution is recorded as max(width,height) per video (matches "54932"-ish
   style magnitude in your example table, which looks like it might actually
   be width*height area in px^2 -- see RESOLUTION_MODE below, set it to
   match whatever convention produced your original numbers).
6. Text stats (Num. Sent / Num. Words / Unique Vocab) come from the
   "_sentences_clean.txt" files directly, same tokenization (str.split()).

USAGE:
    python3 compute_stats.py

Edit the CONFIG block below to match your paths, then run on the machine
that actually has the M2M-MT-files data (this script can't run inside the
chat sandbox since that data isn't present here).
"""

import os
import re
import json
import math
import subprocess
import statistics as stats

# ============================== CONFIG ==================================
TRANSCRIPTION_ROOT = "../../../M2M-MT-files/Youtube_Download_2_Transcription_Whisper"
VIDEO_ROOT = "../../../M2M-MT-files/Youtube_Downloads_2/English"
OUTPUT_LOG = "stats_run_log.txt"
OUTPUT_JSON = "computed_stats.json"
OUTPUT_TEX = "stats_table.tex"

# Video file extensions to consider when scanning a matched video folder.
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".flv"}

# How to compute "resolution" as a single scalar per video, to match
# whatever convention generated your original table numbers (54932 etc).
# Options: "max_dim" (max(width,height)), "area" (width*height),
# "diagonal" (sqrt(w^2+h^2)), or "width".
# NOTE: 54932 doesn't look like a typical pixel-area or single-dimension
# number for common video resolutions (720p/1080p/4k), so please check
# this against your original code/notes and adjust if needed.
RESOLUTION_MODE = "max_dim"

# Optional manual list (or path to a txt file, one folder_name per line) of
# folders that are known to have a sign-language interpreter / sign overlay.
# Leave WITH_SIGN_TXT_PATH = None and VIDEOS_WITH_SIGN_LIST = [] if unknown;
# the column will be filled with 'NA' for you to complete by hand.
WITH_SIGN_TXT_PATH = None
VIDEOS_WITH_SIGN_LIST = []

LANG_LABEL = "en"  # the language row label used in the table
# ==========================================================================


def log(msg, logf):
    print(msg)
    logf.write(msg + "\n")


def ffprobe_video(path):
    """Return dict with width, height, fps, duration_seconds, or None on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        info = json.loads(out.stdout)
    except Exception as e:
        return None

    stream = (info.get("streams") or [{}])[0]
    fmt = info.get("format", {})

    width = stream.get("width")
    height = stream.get("height")

    # duration: prefer stream duration, fall back to format duration
    duration = stream.get("duration") or fmt.get("duration")
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = None

    # fps comes as "30000/1001" style fraction string
    fps = None
    rfr = stream.get("r_frame_rate")
    if rfr and "/" in rfr:
        try:
            num, den = rfr.split("/")
            den = float(den)
            fps = float(num) / den if den != 0 else None
        except Exception:
            fps = None

    if width is None or height is None or duration is None:
        return None

    return {
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "duration_seconds": duration,
    }


def resolution_scalar(w, h, mode):
    if mode == "max_dim":
        return max(w, h)
    if mode == "area":
        return w * h
    if mode == "diagonal":
        return math.sqrt(w ** 2 + h ** 2)
    if mode == "width":
        return w
    raise ValueError(f"Unknown RESOLUTION_MODE: {mode}")


def find_video_candidates(folder_name, video_root):
    """Recursively search video_root for a directory named folder_name,
    then return all video files inside it."""
    matches = []
    for subdir, dirs, files in os.walk(video_root):
        if os.path.basename(subdir) == folder_name:
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in VIDEO_EXTS:
                    matches.append(os.path.join(subdir, f))
    return matches


def pick_representative_video(folder_name, candidates, logf):
    """Given multiple candidate video files for one folder, pick ONE,
    per the priority rules described in the module docstring."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Priority a: exact filename match "<folder_name>.<ext>"
    for c in candidates:
        stem = os.path.splitext(os.path.basename(c))[0]
        if stem == folder_name:
            log(f"[INFO] {folder_name}: multiple videos found, "
                f"using exact-name match {os.path.basename(c)}", logf)
            return c

    # Priority b: fall back to the longest-duration file
    log(f"[WARN] {folder_name}: {len(candidates)} candidate videos, "
        f"none exactly named '{folder_name}.<ext>'. Probing all and "
        f"picking the longest by duration as the representative video.", logf)
    best = None
    best_dur = -1
    for c in candidates:
        info = ffprobe_video(c)
        if info and info["duration_seconds"] > best_dur:
            best = c
            best_dur = info["duration_seconds"]
    if best is None:
        log(f"[ERROR] {folder_name}: ffprobe failed on all {len(candidates)} "
            f"candidates, skipping this folder for video stats.", logf)
    return best


def collect_with_sign_set():
    names = set(VIDEOS_WITH_SIGN_LIST)
    if WITH_SIGN_TXT_PATH and os.path.exists(WITH_SIGN_TXT_PATH):
        with open(WITH_SIGN_TXT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    names.add(line)
    return names


def main():
    with open(OUTPUT_LOG, "w", encoding="utf-8") as logf:
        log(f"Transcription root: {TRANSCRIPTION_ROOT}", logf)
        log(f"Video root: {VIDEO_ROOT}", logf)

        if not os.path.isdir(TRANSCRIPTION_ROOT):
            raise FileNotFoundError(
                f"Transcription root not found: {TRANSCRIPTION_ROOT}. "
                f"Run this script from the same working directory as your "
                f"original splitting script, or fix the path in CONFIG."
            )
        if not os.path.isdir(VIDEO_ROOT):
            raise FileNotFoundError(
                f"Video root not found: {VIDEO_ROOT}. Fix VIDEO_ROOT in CONFIG."
            )

        with_sign_set = collect_with_sign_set()

        # ---- STEP 1: walk transcription root, collect text stats per folder
        folder_records = {}  # folder_name -> {"sentences": [...], "word_counts": [...]}
        for subdir, _, files in os.walk(TRANSCRIPTION_ROOT):
            folder_name = os.path.basename(subdir)
            target_file = f"{folder_name}_sentences_clean.txt"
            if target_file in files:
                target_path = os.path.join(subdir, target_file)
                try:
                    with open(target_path, "r", encoding="utf-8") as f:
                        sentences = [line.strip() for line in f if line.strip()]
                except Exception as e:
                    log(f"[ERROR] Could not read {target_path}: {e!r}", logf)
                    continue
                folder_records[folder_name] = {
                    "sentences": sentences,
                    "word_counts": [len(s.split()) for s in sentences],
                }

        log(f"\nFound {len(folder_records)} transcribed folders.", logf)
        if not folder_records:
            raise ValueError("No transcription folders found — check TRANSCRIPTION_ROOT.")

        # ---- STEP 2: text-level aggregate stats
        all_sentences = []
        all_words = []
        for rec in folder_records.values():
            all_sentences.extend(rec["sentences"])
            for s in rec["sentences"]:
                all_words.extend(s.split())

        num_sent = len(all_sentences)
        num_words = len(all_words)
        unique_vocab = len(set(w.lower() for w in all_words))

        log(f"Num. Sent: {num_sent}", logf)
        log(f"Num. Words: {num_words}", logf)
        log(f"Unique Vocab: {unique_vocab}", logf)

        # ---- STEP 3: for each transcribed folder, find + probe its video
        widths, heights, resolutions, fpss, durations = [], [], [], [], []
        per_video_records = []  # (folder_name, path, duration_seconds)
        matched_count = 0
        unmatched = []

        for folder_name in folder_records:
            candidates = find_video_candidates(folder_name, VIDEO_ROOT)
            chosen = pick_representative_video(folder_name, candidates, logf)
            if chosen is None:
                unmatched.append(folder_name)
                continue
            info = ffprobe_video(chosen)
            if info is None:
                log(f"[ERROR] ffprobe failed on chosen video for "
                    f"{folder_name}: {chosen}", logf)
                unmatched.append(folder_name)
                continue

            matched_count += 1
            w, h = info["width"], info["height"]
            widths.append(w)
            heights.append(h)
            resolutions.append(resolution_scalar(w, h, RESOLUTION_MODE))
            if info["fps"] is not None:
                fpss.append(info["fps"])
            durations.append(info["duration_seconds"])
            per_video_records.append((folder_name, chosen, info["duration_seconds"]))

        log(f"\nMatched videos: {matched_count} / {len(folder_records)}", logf)
        if unmatched:
            log(f"[WARN] {len(unmatched)} folders had no usable matching "
                f"video: {unmatched[:20]}{' ...' if len(unmatched) > 20 else ''}", logf)

        if not durations:
            raise ValueError("No videos were successfully probed — check VIDEO_ROOT "
                              "and VIDEO_EXTS, and inspect the log for ffprobe errors.")

        # ---- STEP 4: aggregate video stats
        num_raw_videos = matched_count
        num_with_sign = len(with_sign_set & set(folder_records.keys())) if with_sign_set else None

        avg_res = stats.mean(resolutions)
        std_res = stats.pstdev(resolutions) if len(resolutions) > 1 else 0.0
        var_res = stats.pvariance(resolutions) if len(resolutions) > 1 else 0.0

        avg_fps = stats.mean(fpss) if fpss else float("nan")
        std_fps = stats.pstdev(fpss) if len(fpss) > 1 else 0.0
        var_fps = stats.pvariance(fpss) if len(fpss) > 1 else 0.0

        durations_min = [d / 60.0 for d in durations]
        total_hours = sum(durations) / 3600.0
        avg_len_min = stats.mean(durations_min)
        std_len_min = stats.pstdev(durations_min) if len(durations_min) > 1 else 0.0
        var_len_min = stats.pvariance(durations_min) if len(durations_min) > 1 else 0.0

        # ---- STEP 5: largest / smallest video by duration
        per_video_records.sort(key=lambda r: r[2])  # sort by duration ascending
        smallest_folder, smallest_path, smallest_dur = per_video_records[0]
        largest_folder, largest_path, largest_dur = per_video_records[-1]

        def fmt_dur(seconds):
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m}m{s:02d}s"

        results = {
            "lang": LANG_LABEL,
            "num_raw_videos": num_raw_videos,
            "num_with_sign": num_with_sign,
            "avg_resolution": avg_res,
            "std_resolution": std_res,
            "var_resolution": var_res,
            "avg_fps": avg_fps,
            "std_fps": std_fps,
            "var_fps": var_fps,
            "total_hours": total_hours,
            "avg_video_length_min": avg_len_min,
            "std_video_length_min": std_len_min,
            "var_video_length_min": var_len_min,
            "num_sent": num_sent,
            "num_words": num_words,
            "unique_vocab": unique_vocab,
            "largest_video": {
                "folder_name": largest_folder,
                "path": largest_path,
                "duration_seconds": largest_dur,
                "duration_fmt": fmt_dur(largest_dur),
            },
            "smallest_video": {
                "folder_name": smallest_folder,
                "path": smallest_path,
                "duration_seconds": smallest_dur,
                "duration_fmt": fmt_dur(smallest_dur),
            },
        }

        with open(OUTPUT_JSON, "w", encoding="utf-8") as jf:
            json.dump(results, jf, indent=2)
        log(f"\nWrote {OUTPUT_JSON}", logf)

        # ---- STEP 6: emit LaTeX table
        def num(x, d=2):
            return f"{x:.{d}f}"

        sign_cell = str(num_with_sign) if num_with_sign is not None else "NA"

        def sci(x):
            # format like 1.36e9 -> "$1.36\times10^{9}$"
            if x == 0:
                return "0"
            exp = int(math.floor(math.log10(abs(x))))
            mantissa = x / (10 ** exp)
            return f"${mantissa:.2f}\\times10^{{{exp}}}$"

        tex = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.15}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{c|cccccccccccc|cc|ccc}
\toprule
& \multicolumn{12}{c|}{\textbf{Video Statistics}} & \multicolumn{2}{c|}{\textbf{Extremes}} & \multicolumn{3}{c}{\textbf{Text Statistics}} \\
\cmidrule(lr){2-13} \cmidrule(lr){14-15} \cmidrule(lr){16-18}
Lang. &
\rotatebox{90}{\makecell{Num.\\Raw\\Videos}} &
\rotatebox{90}{\makecell{Videos\\w/\\Sign}} &
\rotatebox{90}{\makecell{Avg.\\Resolution}} &
\rotatebox{90}{\makecell{Std.\\Resolution}} &
\rotatebox{90}{\makecell{Var.\\Resolution}} &
\rotatebox{90}{\makecell{Avg.\\FPS}} &
\rotatebox{90}{\makecell{Std.\\FPS}} &
\rotatebox{90}{\makecell{Var.\\FPS}} &
\rotatebox{90}{\makecell{Total\\Hours}} &
\rotatebox{90}{\makecell{Avg.\\Video\\Length}} &
\rotatebox{90}{\makecell{Std.\\Video\\Length}} &
\rotatebox{90}{\makecell{Var.\\Video\\Length}} &
\rotatebox{90}{\makecell{Largest\\Video\\(len.)}} &
\rotatebox{90}{\makecell{Smallest\\Video\\(len.)}} &
\rotatebox{90}{\makecell{Num.\\Sent}} &
\rotatebox{90}{\makecell{Num.\\Words}} &
\rotatebox{90}{\makecell{Unique\\Vocab}} \\
\midrule
""" + LANG_LABEL + r""" &
""" + f"{num_raw_videos} &\n{sign_cell} &\n{num(avg_res, 0)} &\n{num(std_res, 0)} &\n{sci(var_res)} &\n{num(avg_fps)} &\n{num(std_fps)} &\n{num(var_fps)} &\n{num(total_hours)} &\n{num(avg_len_min)} &\n{num(std_len_min)} &\n{num(var_len_min)} &\n{largest_dur/60:.2f} &\n{smallest_dur/60:.2f} &\n{num_sent} &\n{num_words} &\n{unique_vocab} \\\\\n" + r"""\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Represents the statistics of the video and text datasets. Shortened words include Number (Num), with (w/), Average (Avg.), Standard deviation (Std), Variance (Var), Frames per second (FPS), Sentence (Sent), and Vocabulary (vocab). The unit of measure for resolution is pixel. The Avg., Std., and Var. video length, and the Largest/Smallest Video columns, are measured in minutes. Largest video: """ + largest_folder.replace("_", "\\_") + r""" (""" + fmt_dur(largest_dur) + r"""). Smallest video: """ + smallest_folder.replace("_", "\\_") + r""" (""" + fmt_dur(smallest_dur) + r""").}
\label{tab:dipolelo_statistics}
\end{table*}
"""

        with open(OUTPUT_TEX, "w", encoding="utf-8") as tf:
            tf.write(tex)
        log(f"Wrote {OUTPUT_TEX}", logf)
        log("\nDone.", logf)


if __name__ == "__main__":
    main()