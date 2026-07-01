import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from nemo.collections.speechlm2.models import SALM

class QwenASRTranscriber:
    """Transcriber using NVIDIA's Canary-Qwen-2.5B ASR (NeMo SALM)"""

    def __init__(self, device=None):
        """
        Initialize the transcriber.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading nvidia/canary-qwen-2.5b on device {self.device}...")
        self.model = SALM.from_pretrained("nvidia/canary-qwen-2.5b").to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")

    def transcribe_audio(self, audio_path, max_new_tokens=10000):
        # Optional: make a temporary mono copy
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            temp_path = "/tmp/temp_mono.wav"
            torchaudio.save(temp_path, waveform, sr)
            audio_path = temp_path

        prompts = [
            [{
                "role": "user",
                "content": f"Transcribe the following: {self.model.audio_locator_tag}",
                "audio": [audio_path]
            }]
        ]

        with torch.inference_mode():
            outputs = self.model.generate(prompts=prompts, max_new_tokens=max_new_tokens)

        return self.model.tokenizer.ids_to_text(outputs[0].cpu()).strip()




    def transcribe_segments_by_folder(self, segments_df, base_output_dir):
        results = {}
        folder_groups = segments_df.groupby("folder_name")

        for folder_name, group in folder_groups:
            print(f"\nProcessing folder: {folder_name}")
            folder_output_dir = Path(base_output_dir) / folder_name
            folder_output_dir.mkdir(parents=True, exist_ok=True)
            txt_file_path = folder_output_dir / f"{folder_name}.txt"

            sorted_group = group.sort_values("start_time")
            folder_transcriptions = []

            for _, row in tqdm(sorted_group.iterrows(), desc=f"Transcribing {folder_name}", total=len(sorted_group)):
                try:
                    path_runtime = row["audio_path"].replace("/home/unarine", "/home/neo")
                    transcription_text = self.transcribe_audio(path_runtime)
                    folder_transcriptions.append({
                        "segment_id": row["segment_id"],
                        "start_time": row["start_time"],
                        "end_time": row["end_time"],
                        "text": transcription_text
                    })
                except Exception as e:
                    print(f"❌ Error transcribing {path_runtime}: {e}")
                    continue

            if folder_transcriptions:
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    f.write(" ".join([t["text"] for t in folder_transcriptions]))

                results[folder_name] = {
                    "file_path": str(txt_file_path),
                    "segment_count": len(folder_transcriptions),
                    "total_words": sum(len(t["text"].split()) for t in folder_transcriptions),
                    "transcriptions": folder_transcriptions
                }
                print(f"✅ Saved transcription: {txt_file_path}")
            else:
                results[folder_name] = {"file_path": None, "segment_count": 0, "total_words": 0, "transcriptions": []}
                print(f"❌ No valid transcriptions for folder: {folder_name}")

        return results

    def transcribe_folder_from_csv(self, csv_path, base_output_dir):
        print(f"Loading segments from: {csv_path}")
        segments_df = pd.read_csv(csv_path)
        required_columns = ["folder_name", "segment_path", "segment_id", "start_time", "end_time", "audio_path"]
        missing_columns = [c for c in required_columns if c not in segments_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        return self.transcribe_segments_by_folder(segments_df, base_output_dir)


if __name__ == "__main__":
    transcriber = QwenASRTranscriber()
    results = transcriber.transcribe_folder_from_csv(
        "../../../M2M-MT-files/Youtube_CSV_files/copy_video_name_id_start_end_times_2.csv",
        base_output_dir="../../../M2M-MT-files/Youtube_Download_2_Transcription_Canary"
    )
2048