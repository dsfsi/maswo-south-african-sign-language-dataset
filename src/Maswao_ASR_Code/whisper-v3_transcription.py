import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline


class WhisperASRTranscriber:
    """Transcriber using OpenAI Whisper large-v3"""

    def __init__(self, device=None, language=None, task="transcribe"):
        """
        Initialize the Whisper ASR transcriber.

        Args:
            device (str): Device to run on ('cuda' or 'cpu'). Auto-detect if None.
            language (str): Optional language code (e.g. 'en') to force transcription
                             language. If None, Whisper auto-detects.
            task (str): 'transcribe' or 'translate' (translate always outputs English).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Loading OpenAI Whisper large-v3 on device {self.device}...")

        self.model_name = "openai/whisper-large-v3"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)

        self.language = language
        self.task = task

        # HF pipeline handles resampling, chunking (>30s) and batching for us.
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            chunk_length_s=30,
            stride_length_s=5,
        )

        print("Model loaded successfully!")

    def transcribe_audio(self, audio_path, max_chunk_seconds=30):
        """
        Transcribe an audio file. Whisper's pipeline natively handles
        long-form audio by internally chunking with a sliding window,
        so no manual splitting is required.

        Args:
            audio_path (str): Path to the audio file.
            max_chunk_seconds (int): Chunk length in seconds fed to the model
                                      (kept for interface parity with the
                                      Granite version; passed to the pipeline).

        Returns:
            str: The transcribed text.
        """
        wav, sr = torchaudio.load(audio_path, normalize=True)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)
            sr = 16000

        audio_array = wav.squeeze(0).numpy()

        generate_kwargs = {"task": self.task}
        if self.language:
            generate_kwargs["language"] = self.language

        # chunk_length_s can be overridden per-call if needed
        result = self.pipe(
            {"array": audio_array, "sampling_rate": sr},
            chunk_length_s=max_chunk_seconds,
            generate_kwargs=generate_kwargs,
        )

        return result["text"].strip()

    def transcribe_segments_by_folder(self, segments_df, base_output_dir="../../../M2M-MT-files/Youtube_Download_2_Transcription_Whisper"):
        """
        Transcribe audio segments grouped by folder.

        Args:
            segments_df (pd.DataFrame): DataFrame with columns ['folder_name','segment_path','segment_id','start_time','end_time']
            base_output_dir (str): Directory to save transcriptions.

        Returns:
            dict: Transcription results by folder.
        """
        results = {}
        folder_groups = segments_df.groupby('folder_name')

        for folder_name, group in folder_groups:
            print(f"\nProcessing folder: {folder_name}")
            folder_output_dir = Path(base_output_dir) / folder_name
            folder_output_dir.mkdir(parents=True, exist_ok=True)
            txt_file_path = folder_output_dir / f"{folder_name}.txt"

            sorted_group = group.sort_values('start_time')
            folder_transcriptions = []

            for _, row in tqdm(sorted_group.iterrows(), desc=f"Transcribing {folder_name}", total=len(sorted_group)):
                try:
                    path_runtime = row['audio_path']
                    path_runtime = path_runtime.replace("/home/unarine", "/home/neo")
                    transcription_text = self.transcribe_audio(path_runtime)
                    if transcription_text:
                        folder_transcriptions.append({
                            'segment_id': row['segment_id'],
                            'start_time': row['start_time'],
                            'end_time': row['end_time'],
                            'text': transcription_text
                        })
                except Exception as e:
                    print(f"Error transcribing {path_runtime}: {e}")
                    continue

            if folder_transcriptions:
                with open(txt_file_path, 'w', encoding='utf-8') as f:
                    f.write(" ".join([t['text'] for t in folder_transcriptions]))

                results[folder_name] = {
                    'file_path': str(txt_file_path),
                    'segment_count': len(folder_transcriptions),
                    'total_words': sum(len(t['text'].split()) for t in folder_transcriptions),
                    'transcriptions': folder_transcriptions
                }
                print(f"✅ Saved transcription: {txt_file_path}")
            else:
                results[folder_name] = {'file_path': None, 'segment_count': 0, 'total_words': 0, 'transcriptions': []}
                print(f"❌ No valid transcriptions found for folder: {folder_name}")

        return results

    def transcribe_folder_from_csv(self, csv_path, base_output_dir="../../../M2M-MT-files/Youtube_Download_2_Transcription_Whisper"):
        """
        Convenience method to transcribe from a segments CSV file.

        Args:
            csv_path (str): Path to CSV file.
            base_output_dir (str): Directory to save transcriptions.

        Returns:
            dict: Transcription results by folder.
        """
        print(f"Loading segments from: {csv_path}")
        segments_df = pd.read_csv(csv_path)
        required_columns = ['folder_name', 'segment_path', 'segment_id', 'start_time', 'end_time', 'audio_path']
        missing_columns = [c for c in required_columns if c not in segments_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        return self.transcribe_segments_by_folder(segments_df, base_output_dir)


# Usage Example
if __name__ == "__main__":
    # Pass language="en" (or another code) to skip language auto-detection,
    # or task="translate" to always output English text.
    transcriber = WhisperASRTranscriber()
    results = transcriber.transcribe_folder_from_csv("../../../M2M-MT-files/Youtube_CSV_files/copy_video_name_id_start_end_times_2.csv")