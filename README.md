# Maswao Dataset Construction Pipeline

This repository contains the code used to construct the **Maswao** dataset, a weakly aligned spoken-language-to-sign-language corpus for South African Sign Language (SASL).

## Overview

The pipeline consists of the following stages:

1. Download videos from YouTube.
2. Identify videos containing sign language interpretation.
3. Transcribe spoken audio into text using Automatic Speech Recognition (ASR).
4. Segment the transcriptions into sentence-level units.
5. Detect and crop the sign language interpreter region from each video.
6. Segment long sign language videos into sentence-level clips using our proposed gradient-informed segmentation method.
7. Align sentence-level text with the corresponding sign language video segments to construct the Maswao dataset.

## Repository Structure

The repository is organized into modules corresponding to each stage of the pipeline. Additional documentation for each module will be provided in future updates.

## Requirements

We reccommend setting up different conda/python environments for each stage in the pipeline as follows:
1. Dependencies for downloading, interpreter segmentation include - yt_dlp, cv2, moviepy, and mediapipe
2. Dependencies for ASR include - torch, transformers, and nemo 
3. Installation instructions and software dependencies for training each of the SLT models in src folder are provided in their read me files. 

## Usage

1. To run the download script : bash crawl_script.sh  -- you need to create a folder and add cookies files in txt format, also create a YouTube API key
2. To run interpreter python script - python maswao_interpreter_cropping.py
3. To run ASR code - python name_of_asr_script.py  (e.g python whisper-v3_transcription.py)
4. Instructions give in readme for running SLT

## Citation

If you use this code or the Maswao dataset in your research, please cite:

```bibtex
@article{TBD,
  title   = {TBD},
  author  = {TBD},
  journal = {TBD},
  year    = {TBD}
}
```

## License

This repository is released under the license included in this project.

> **Note**
> Due to YouTube licensing restrictions and ethical considerations regarding the privacy of sign language interpreters, the original video clips are **not** redistributed. Instead, we provide the corresponding YouTube video identifiers and accompanying annotations, enabling researchers to reconstruct the dataset from the original publicly available sources.