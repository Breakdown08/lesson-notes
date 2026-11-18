# lesson-notes

Turns a two-track recording of a foreign-language lesson into two Markdown files:

- `transcript.md`: who said what and when (speaker = audio track, no diarization);
- `summary.md`: a short summary made by an LLM: what was covered, new words, the student's mistakes, topics and grammar, homework if it was mentioned.

The recording is expected to come from OBS with two audio tracks: track 1 is the student's microphone, track 2 is the system sound (the teacher). Speech is recognized locally with faster-whisper; the summary is written by xAI or Anthropic models through LangChain. The pipeline is a LangGraph graph.

## Install

Python 3.14 and [uv](https://docs.astral.sh/uv/) are required. ffmpeg is not needed: PyAV bundles its own libraries.

```
uv sync
cp .env.example .env   # then edit the paths and the API key
```

Whisper weights are downloaded from Hugging Face by faster-whisper on the first run. They are not pinned to a revision or checksum yet.

## Run

```
uv run lesson-notes run [RECORDING] [--force] [--env-file PATH]
```

Without `RECORDING` the newest `.mkv`/`.mp4`/`.mov` file in `LESSON_NOTES_RECORDINGS_DIR` is used. The lesson date comes from an OBS-style file name (`2026-11-16 18-00-03.mkv`), otherwise from the file's modification time.

Output:

- `<output>/<yyyy-mm-dd>/transcript.md` and `summary.md`;
- `<summary copy directory>/<yyyy-mm-dd>.md`, a copy of the summary.

The output and summary copy directories must already exist; they are never created.

If the summary step fails (bad key, network, quota), `transcript.md` stays on disk and the next run resumes from it without recognizing the audio again. You can switch `LESSON_NOTES_LLM_PROVIDER` between runs. If the folder already holds a finished lesson, or a transcript of another recording, use `--force` to overwrite it.

## Configuration

All settings are environment variables, optionally read from an env file (`.env` by default); real environment variables win. See `.env.example` for the full list. Only the API key of the selected provider (`xai` or `anthropic`) is required.

## Development

```
uv run ruff check
uv run ruff format --check
uv run ty check
uv run mypy
uv run pytest
```

The tests are offline: recognition and the LLM are replaced by fakes, and test recordings are generated synthetically.

## Limitations

- No diarization: the speaker is determined only by the track number.
- One recording per run; no watch mode, no video, no GUI.
- Recognition is local faster-whisper only.
- Tracks are assumed to start at the same moment; offsets are not corrected.
- Two lessons on the same day are an error (the folder name is the date).
- No automatic fallback to another LLM provider.
- Recognition quality on mixed Russian/French speech has not been scored against a reference transcript.

## Measured

First real lesson, 2026-11-18: a 1 h 19 min OBS recording, student on track 2, teacher on track 3, `large-v3-turbo` int8 on an Intel Core Ultra 7 155H (CPU only).

- Wall time 1625 s (27 min) for both tracks plus the summary, about 0.34× real time; peak memory 5.2 GB (`/usr/bin/time`).
- VAD dropped 60 min of silence from the student track and 25 min from the teacher track.
- The transcript has 1112 teacher lines and 133 student lines; Russian and French lines alternate within one track.
- Known defects seen by eye: mis-heard words in fast French, one repeated-phrase hallucination, and names spelled several ways. The summary marks unclear spots itself.

## License

Apache 2.0, see [LICENSE](LICENSE).
