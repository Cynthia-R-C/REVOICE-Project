from TTS.api import TTS
import torch, os

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS('tts_models/en/vctk/vits').to(device)

os.makedirs('speaker_samples', exist_ok=True)
test_text = "This is a quick test of what this voice sounds like."

for speaker in tts.speakers:
    speaker = speaker.strip()
    if not speaker:
        continue
    try:
        tts.tts_to_file(text=test_text, speaker=speaker, file_path=f'speaker_samples/{speaker}.wav')
        print(f'saved {speaker}')
    except Exception as e:
        print(f'failed on {speaker}: {e}')