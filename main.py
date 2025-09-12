from datasets import load_dataset, Audio
import torchaudio
import torch
import os, json
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, davies_bouldin_score
import whisper
from whisper.model import Whisper
import warnings
warnings.filterwarnings("ignore", module="whisper.timing")


import soundfile as sf
from silero_vad import (
    load_silero_vad,
    read_audio,
    get_speech_timestamps,
    save_audio,
    VADIterator,
    collect_chunks,
)
from typing import Tuple, List, Dict
from speechbrain.utils.fetching import LocalStrategy
from speechbrain.inference import EncoderClassifier
import numpy as np

DATASET_DIR = "audio_out"


def dump_dataset_paths(ds, out_dir, n=10):
    os.makedirs(out_dir, exist_ok=True)

    for i, row in enumerate(ds.select(range(n))):
        sample_dir = os.path.join(out_dir, f"sample_{i}")
        os.makedirs(sample_dir, exist_ok=True)

        # save only the path reference
        audio_path = row["audio"]["path"]

        # save metadata (all other features + audio path)
        meta = {k: v for k, v in row.items() if k != "audio"}
        meta["audio_path"] = audio_path

        with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved {n} sample metadata entries to {out_dir}")


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    # Case 1: flat vector [num_samples]
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)

    # Case 2: stereo or multi-channel [channels, num_samples]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # collapse to mono

    return waveform


def load_audio(file_path, target_sr=16000) -> Tuple[torch.Tensor, int]:
    """
    Load audio from file (wav, mp3, mp4, etc.) and resample if needed.
    Returns: waveform tensor, sample_rate
    """
    waveform, sr = torchaudio.load(file_path)

    # resample if needed
    if target_sr and sr != target_sr:
        print(f"resample the {file_path} from {sr} to {target_sr}")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    waveform = ensure_mono(waveform)
    return waveform, sr


def extract_embedding(waveform, sr, classifier: EncoderClassifier) -> np.ndarray:

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)

    with torch.no_grad():
        embedding = classifier.encode_batch(waveform)
    return embedding.squeeze(0).cpu().numpy()


def clean_segments(starts, ends, min_dur=0.2):
    segments = [(s, e) for s, e in zip(starts, ends) if e > s]
    segments.sort(key=lambda x: x[0])

    cleaned = []
    for seg in segments:
        if seg[1] - seg[0] < min_dur:
            continue  # skip too short
        if not cleaned:
            cleaned.append(seg)
        else:
            prev_start, prev_end = cleaned[-1]
            if seg[0] <= prev_end:  # overlap
                cleaned[-1] = (prev_start, max(prev_end, seg[1]))
            else:
                cleaned.append(seg)
    return cleaned


def eval_detection(detected, ground_truth):
    i, j = 0, 0
    matched_gt, matched_det = set(), set()

    while i < len(detected) and j < len(ground_truth):
        d_start, d_end = detected[i]
        g_start, g_end = ground_truth[j]

        if d_end < g_start:
            i += 1
        elif g_end < d_start:
            j += 1
        else:  # overlap
            matched_gt.add(j)
            matched_det.add(i)
            if d_end < g_end:
                i += 1
            else:
                j += 1

    recall = len(matched_gt) / len(ground_truth)
    fa_rate = 1 - (len(matched_det) / len(detected))

    return recall, fa_rate


def extract_embeddings_from_segments(
    waveform, classifier, speech_timestamps,
) -> torch.Tensor:
    embeddings = []
    for ts in speech_timestamps:
        start, end = ts["start"], ts["end"]  # in samples
        segment = waveform[:, start:end]  # slice
        if segment.numel() == 0:
            continue  # skip empty
        with torch.no_grad():
            emb = classifier.encode_batch(segment.to(classifier.device))
        embeddings.append(emb.cpu())
    assert len(embeddings) > 0, "No embeddings were created"
    return torch.cat(embeddings, dim=0)


asr_model: Whisper = whisper.load_model(
    "small", device="cuda"
)  # or "medium", "large-v2"


def agglomerative_k_search(
    embeddings: np.ndarray,
    metric: str = "cosine",
    linkage_method: str = "average",
    min_clusters: int = 2,
    max_clusters: int = 20,
):
    """
    Cluster embeddings with Agglomerative (dendrogram once, cut by k).
    Picks best k in [min_clusters, max_clusters] using silhouette score.

    Returns:
        best_labels: np.ndarray of shape (N,)
        best_k: int, chosen number of clusters
        best_score: float
        Z: linkage matrix
    """
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        return np.zeros(n_samples, dtype=int), 1, 0.0, None

    # Step 1: compute pairwise distances and dendrogram (once)
    condensed = pdist(embeddings, metric=metric)
    Z = linkage(condensed, method=linkage_method)

    # Step 2: cache full distance matrix for silhouette
    dist_mat = squareform(condensed)

    best_score = -np.inf
    best_labels, best_k = None, None

    # Step 3: sweep possible k values
    for k in range(min_clusters, min(max_clusters, n_samples - 1) + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        n_clusters = len(set(labels))
        if n_clusters < 2:
            continue

        try:
            score = silhouette_score(dist_mat, labels, metric="precomputed")
        except Exception:
            score = -davies_bouldin_score(embeddings, labels)

        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = n_clusters

    # Fallback: if nothing valid, just cut at min_clusters
    if best_labels is None:
        best_labels = fcluster(Z, t=min_clusters, criterion="maxclust")
        best_k = min_clusters
        try:
            best_score = silhouette_score(dist_mat, best_labels, metric="precomputed")
        except Exception:
            best_score = -davies_bouldin_score(embeddings, best_labels)

    return best_labels, best_k, best_score


def group_segments(segments, labels, max_gap=0.8, max_len=30.0, sr=16000):
    """
    Merge speech segments from the same speaker if they are close enough.
    - max_gap: maximum silence (in seconds) to allow merging
    - max_len: maximum total segment length (in seconds)
    """
    grouped = []
    cur_start, cur_end = None, None
    cur_label = None

    for seg, label in zip(segments, labels):
        start, end = seg["start"] / sr, seg["end"] / sr

        if cur_label is None:
            # start new group
            cur_start, cur_end, cur_label = start, end, label
            continue

        if (
            label == cur_label
            and (start - cur_end) <= max_gap
            and (end - cur_start) <= max_len
        ):
            # extend current group
            cur_end = end
        else:
            grouped.append({"speaker": cur_label, "start": cur_start, "end": cur_end})
            cur_start, cur_end, cur_label = start, end, label

    if cur_label is not None:
        grouped.append({"speaker": cur_label, "start": cur_start, "end": cur_end})

    return grouped


def transcribe_chunks(waveform, sr, chunks, asr_model: Whisper):
    """
    Run Whisper ASR on pre-chunked waveform segments.
    - waveform: torch.Tensor [1, num_samples]
    - sr: sample rate
    - chunks: list of dicts with {start, end, segments} in samples
    - asr_model: loaded Whisper model
    Returns transcripts aligned in **samples** (not seconds).
    """
    transcripts = []

    for i, c in enumerate(chunks):
        # Slice chunk by samples
        segment = waveform[:, c["start"] : c["end"]].cpu().numpy()
        segment = segment.squeeze().astype("float32")

        # Normalize if needed
        if segment.max() > 1.0:
            segment = segment / max(1e-9, abs(segment).max())

        # Run Whisper with timestamps
        result = asr_model.transcribe(
            segment,
            fp16=False,  # safer on CPU/small GPU
            word_timestamps=True,  # return per-word timestamps
            beam_size=7,  # beam search for stability
            temperature=(0, 0.1, 0.2, 0.4, 0.8),  # deterministic output
            compression_ratio_threshold=1.8,
        )

        # Convert Whisper word timestamps (sec) → samples
        words = []
        for w in result.get("segments", []):
            for item in w["words"]:
                words.append(
                    {
                        "word": item["word"],
                        "start": c["start"] + int(item["start"] * sr),
                        "end": c["start"] + int(item["end"] * sr),
                    }
                )

        transcripts.append(
            {
                "chunk_id": i,
                "start": c["start"],  # in samples
                "end": c["end"],  # in samples
                "text": result["text"].strip(),
                "words": words,
            }
        )

        # Optional debug log in seconds
        # print(f"[Chunk {i}] {c['start']/sr:.2f}-{c['end']/sr:.2f}s: {result['text'].strip()}")
        # print(f"[Chunk {i}] {c['start']/sr:.4f}-{c['end']/sr:.4f}s", end=" ")

    return transcripts


def chunk_by_silence_and_overlap(
    speech_timestamps,
    sr,
    min_silence=2.0,
    max_chunk=30.0,
    overlap=5.0,
):
    """
    Create ASR chunks (works in samples, not seconds).
    """
    min_silence_samples = int(min_silence * sr)
    max_chunk_samples = int(max_chunk * sr)
    overlap_samples = int(overlap * sr)

    chunks = []
    cur_segments = []
    cur_start, cur_end = None, None

    for seg in speech_timestamps:
        start, end = (seg["start"], seg["end"]) if isinstance(seg, dict) else seg

        if cur_start is None:
            cur_start, cur_end = start, end
            cur_segments = [(start, end)]
            continue

        gap = start - cur_end

        if gap >= min_silence_samples:
            chunks.append(
                {"start": cur_start, "end": cur_end, "segments": cur_segments}
            )
            cur_start, cur_end = start, end
            cur_segments = [(start, end)]
        else:
            cur_end = end
            cur_segments.append((start, end))

    if cur_segments:
        chunks.append({"start": cur_start, "end": cur_end, "segments": cur_segments})

    # --- Second pass: split long chunks with overlap ---
    final_chunks = []
    for c in chunks:
        duration = c["end"] - c["start"]
        if duration <= max_chunk_samples:
            final_chunks.append(c)
        else:
            start = c["start"]
            while start < c["end"]:
                end = min(start + max_chunk_samples, c["end"])
                final_chunks.append(
                    {"start": start, "end": end, "segments": c["segments"]}
                )
                if end == c["end"]:
                    break
                start = end - overlap_samples  # slide with overlap

    return final_chunks


def assign_speakers(transcripts, diar_segments, sr, snap_gap_sec=1):
    snap_gap_samples = int(snap_gap_sec * sr)
    results = []
    cur_speaker, cur_words = None, []

    # flatten word-level info across chunks
    words_all = []
    for t in transcripts:
        words_all.extend(t["words"])

    for w in words_all:
        w_mid = (w["start"] + w["end"]) // 2

        # find diar segment covering this word
        candidates = [
            seg for seg in diar_segments if seg["start"] <= w_mid <= seg["end"]
        ]
        if candidates:
            speaker = candidates[0]["speaker"]
        else:
            # nearest diar segment
            nearest = min(
                diar_segments,
                key=lambda s: min(abs(w_mid - s["start"]), abs(w_mid - s["end"])),
            )
            gap = min(abs(w_mid - nearest["start"]), abs(w_mid - nearest["end"]))
            speaker = nearest["speaker"] if gap <= snap_gap_samples else None

        # group by speaker
        if speaker != cur_speaker:
            if cur_words:
                results.append(
                    {
                        "speaker": (
                            f"Speaker {cur_speaker}"
                            if isinstance(cur_speaker, (int, np.integer))
                            else (cur_speaker or "Unknown")
                        ),
                        "start": min(wd["start"] for wd in cur_words),
                        "end": max(wd["end"] for wd in cur_words),
                        "text": " ".join(wd["word"] for wd in cur_words),
                    }
                )
            cur_speaker, cur_words = speaker, [w]
        else:
            cur_words.append(w)

    # flush last
    if cur_words:
        results.append(
            {
                "speaker": (
                    f"Speaker {cur_speaker}"
                    if isinstance(cur_speaker, (int, np.integer))
                    else (cur_speaker or "Unknown")
                ),
                "start": min(wd["start"] for wd in cur_words),
                "end": max(wd["end"] for wd in cur_words),
                "text": " ".join(wd["word"] for wd in cur_words),
            }
        )

    return results



if __name__ == "__main__":
    print("Running main.py")

    if not os.path.exists(DATASET_DIR):
        # First time: load from HF and save metadata
        ds = load_dataset("diarizers-community/ami", "ihm", split="test")
        ds = ds.cast_column("audio", Audio(decode=False))
        print(ds.features)  # schema of all features
        # Inspect a single row
        row = ds[0]
        print("Keys:", row.keys())
        print("Audio field keys:", row["audio"].keys())
        print("Audio path:", row["audio"]["path"])
        print("Audio bytes length:", len(row["audio"]["bytes"]))
        dump_dataset_paths(ds, out_dir=DATASET_DIR, n=10)
    else:
        print(f"{DATASET_DIR} exists, skipping download.")

    sample_id = "sample_2"

    # --- Load audio ---
    audio_path = os.path.join(DATASET_DIR, sample_id, "audio.wav")
    waveform, sample_rate = load_audio(audio_path)

    # --- Voice Activity Detection (VAD) ---
    vad_model = load_silero_vad(onnx=True)
    speech_segments = get_speech_timestamps(
        waveform, vad_model, sampling_rate=sample_rate
    )

    # --- Speaker Embeddings ---
    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="./pretrained_models/spkrec-ecapa",
        run_opts={"device": "cuda"},
        local_strategy=LocalStrategy.COPY,
    )
    embeddings = extract_embeddings_from_segments(
        waveform, speaker_encoder, speech_segments
    )  # tensor [num_segments, 1, emb_dim]

    # Flatten tensor for clustering
    embeddings = embeddings.squeeze(1).numpy()

    # --- Clustering (Speaker Diarization) ---
    speaker_labels, thr, score = agglomerative_k_search(
        embeddings,
        linkage_method="average",
        metric="cosine",
        min_clusters=2,
        max_clusters=20,
    )

    print("Best threshold:", thr)
    print("Best silhouette score:", score)
    # print("Labels:", speaker_labels[:20])

    # --- Attach labels to speech segments ---
    labeled_segments = [
        {
            "speaker": f"Speaker {label}",
            "start": seg["start"],  # in samples
            "end": seg["end"],  # in samples
        }
        for seg, label in zip(speech_segments, speaker_labels)
    ]

    # --- Ground-truth for evaluation ---
    meta_path = os.path.join(DATASET_DIR, sample_id, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    gt_segments = clean_segments(
        meta["timestamps_start"], meta["timestamps_end"], min_dur=1e-3
    )
    n_speaker_gt = len(set(meta["speakers"]))
    # Convert detected segments to seconds for scoring
    detected_sec = [(s["start"] / sample_rate, s["end"] / sample_rate)
                    for s in speech_segments]

    recall, false_alarm = eval_detection(detected_sec, gt_segments)
    print(f"Recall={recall:.4f}, FA={false_alarm:.4f}")
    print(f"Found {len(set(speaker_labels))} speakers, GT got {n_speaker_gt} speakers")

    chunks = chunk_by_silence_and_overlap(speech_segments, sample_rate, max_chunk=30, overlap=3)
    transcript = transcribe_chunks(waveform, sample_rate, chunks, asr_model)
    final_result = assign_speakers(transcript, labeled_segments, sample_rate)
    print(final_result, sep="\n")
    print("Done.")
