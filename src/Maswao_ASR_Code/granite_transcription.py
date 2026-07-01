import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from huggingface_hub import hf_hub_download

class GraniteASRTranscriber:
    """Transcriber using IBM Granite Speech 3.3 8B"""

    def __init__(self, device=None, system_prompt=None):
        """
        Initialize the Granite ASR transcriber.

        Args:
            device (str): Device to run on ('cuda' or 'cpu'). Auto-detect if None.
            system_prompt (str): Optional system prompt for the model.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading IBM Granite Speech 3.3 8B on device {self.device}...")

        self.model_name = "ibm-granite/granite-speech-3.3-8b"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        )
        self.model = self.model.to(self.device)

        self.system_prompt = system_prompt or (
            "Knowledge Cutoff Date: April 2024.\n"
            "Today's Date: April 9, 2025.\n"
            "You are Granite, developed by IBM. You are a helpful AI assistant."
        )

        print("Model loaded successfully!")

    def transcribe_audio(self, audio_path, max_new_tokens=10000, max_chunk_seconds=30):
            """
            Transcribe an audio file, automatically splitting long files.
            """
            wav, sr = torchaudio.load(audio_path, normalize=True)

            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wav = resampler(wav)
                sr = 16000

            # Split audio if longer than allowed
            max_samples = sr * max_chunk_seconds
            num_chunks = (wav.shape[1] + max_samples - 1) // max_samples

            transcripts = []
            for i in range(num_chunks):
                start = i * max_samples
                end = min((i + 1) * max_samples, wav.shape[1])
                chunk = wav[:, start:end]

                user_prompt = "<|audio|>can you transcribe the speech into a written format?"
                chat = [
                    dict(role="system", content=self.system_prompt),
                    dict(role="user", content=user_prompt),
                ]
                prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

                model_inputs = self.processor(prompt, chunk, device=self.device, return_tensors="pt").to(self.device)

                with torch.inference_mode():
                    outputs = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)

                num_input_tokens = model_inputs["input_ids"].shape[-1]
                new_tokens = outputs[:, num_input_tokens:]
                text = self.tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)[0]
                transcripts.append(text.strip())

            return " ".join(transcripts).strip()


    def transcribe_segments_by_folder(self, segments_df, base_output_dir="../../../M2M-MT-files/Youtube_Download_2_Transcription_Granite"):
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

    def transcribe_folder_from_csv(self, csv_path, base_output_dir="../../../M2M-MT-files/Youtube_Download_2_Transcription_Granite"):
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
    transcriber = GraniteASRTranscriber()
    results = transcriber.transcribe_folder_from_csv("../../../M2M-MT-files/Youtube_CSV_files/copy_video_name_id_start_end_times_2.csv")
