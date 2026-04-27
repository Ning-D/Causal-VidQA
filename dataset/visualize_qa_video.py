#!/usr/bin/env python3
"""
visualize_qa_video.py

Render Causal-VidQA samples as annotated videos (video frame + QA panel).

Layout per frame:
  ┌─────────────────┬──────────────────────┐
  │                 │  [qtype label]        │
  │   video frame   │  Q: ...               │
  │                 │  [0] candidate        │
  │                 │  [1] ✓ correct        │
  │                 │  ...                  │
  └─────────────────┴──────────────────────┘
Each of the 6 QA types is shown for SECS_PER_QTYPE seconds while the
video loops.  Output: one annotated mp4 per sample.

Clip source (in priority order):
  1. Already extracted mp4 in --clip_dir
  2. Extract from local --tar_dir (dataset.tar.[a-f])  ← builds index once
  3. Download via yt-dlp

Usage:
    # single video
    python dataset/visualize_qa_video.py \\
        --vid BkpA2Qc3Jew_000028_000038 \\
        --data_path ./data/QA \\
        --result_file ./result/B2A/B2A_test.json \\
        --tar_dir ./data \\
        --out_dir ./viz_samples

    # N random test samples
    python dataset/visualize_qa_video.py \\
        --split_path ./data/split/test.pkl --n 3 \\
        --data_path ./data/QA --tar_dir ./data --out_dir ./viz_samples
"""

import argparse, bisect, io, json, os, pickle, random, re, subprocess
import sys, tarfile, tempfile
import cv2
import numpy as np

# ── layout / style ─────────────────────────────────────────────────────────────
SECS_PER_QTYPE = 2
PANEL_W        = 560
FRAME_H        = 360
FONT           = cv2.FONT_HERSHEY_SIMPLEX
BG_COLOR       = (30, 30, 30)
TEXT_COLOR     = (220, 220, 220)
CORRECT_COLOR  = (80, 200, 80)
WRONG_COLOR    = (80, 80, 220)

QTYPE_META = {
    0: ('Descriptive',           'descriptive',    'answer',  (100, 200, 255)),
    1: ('Explanatory',           'explanatory',    'answer',  (150, 255, 150)),
    2: ('Predictive-Answer',     'predictive',     'answer',  (255, 200, 100)),
    3: ('Predictive-Reason',     'predictive',     'reason',  (255, 200, 100)),
    4: ('Counterfactual-Answer', 'counterfactual', 'answer',  (200, 150, 255)),
    5: ('Counterfactual-Reason', 'counterfactual', 'reason',  (200, 150, 255)),
}

TAR_PART_NAMES = [f'dataset.tar.{x}' for x in 'abcdef']
TAR_INDEX_NAME = 'dataset_tar_index.json'


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_vid_id(vid):
    m = re.match(r'^(.+)_(\d{6})_(\d{6})$', vid)
    if not m:
        raise ValueError(f'Cannot parse: {vid}')
    return m.group(1), int(m.group(2)), int(m.group(3))


def wrap_text(text, max_chars=54):
    words, lines, cur = text.split(), [], ''
    for w in words:
        if len(cur) + len(w) + 1 > max_chars:
            if cur:
                lines.append(cur)
            cur = w
        else:
            cur = (cur + ' ' + w).strip()
    if cur:
        lines.append(cur)
    return lines


def draw_panel(panel, qtype_id, text_data, answer_data, predictions, vid):
    label, section, key, hdr_color = QTYPE_META[qtype_id]
    h = panel.shape[0]
    panel[:] = BG_COLOR

    if section not in text_data or key not in text_data[section]:
        return

    qns        = text_data[section]['question']
    candidates = text_data[section][key]
    ans_id     = answer_data[section][key]
    pred_id    = (predictions or {}).get(f'{vid}_{qtype_id}', {}).get('prediction')

    y, fs, th, lh = 22, 0.42, 1, 20

    cv2.putText(panel, f'[{qtype_id}] {label}', (10, y),
                FONT, 0.50, hdr_color, 1, cv2.LINE_AA)
    y += lh + 4
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (80, 80, 80), 1)
    y += 10

    for line in wrap_text('Q: ' + qns, 54):
        cv2.putText(panel, line, (10, y), FONT, fs, TEXT_COLOR, th, cv2.LINE_AA)
        y += lh
    y += 6

    for i, cand in enumerate(candidates):
        is_ans, is_pred = (i == ans_id), (pred_id is not None and i == pred_id)
        if is_ans and is_pred:
            color, prefix = CORRECT_COLOR, f'[{i}] pred+ans'
        elif is_ans:
            color, prefix = CORRECT_COLOR, f'[{i}] ans      '
        elif is_pred:
            color, prefix = WRONG_COLOR,   f'[{i}] pred     '
        else:
            color, prefix = TEXT_COLOR,    f'[{i}]          '
        for j, line in enumerate(wrap_text(prefix + cand, 54)):
            if y > h - lh:
                break
            cv2.putText(panel, line, (10, y), FONT, fs, color, th, cv2.LINE_AA)
            y += lh
        y += 3

    cv2.putText(panel, 'green=correct  blue=predicted', (10, h - 28),
                FONT, 0.35, (120, 120, 120), 1, cv2.LINE_AA)


# ── tar index ─────────────────────────────────────────────────────────────────

def _part_cumulative(tar_dir):
    """Return (parts_paths, cumulative_byte_offsets)."""
    parts, cum = [], [0]
    for name in TAR_PART_NAMES:
        p = os.path.join(tar_dir, name)
        if os.path.exists(p):
            parts.append(p)
            cum.append(cum[-1] + os.path.getsize(p))
    return parts, cum


def build_tar_index(tar_dir):
    """
    Stream through dataset.tar.* once, record each .mp4's
    (global_data_offset, size).  Saves index to tar_dir/TAR_INDEX_NAME.
    ~4 min on HDD at 242 MB/s.
    """
    parts, cum = _part_cumulative(tar_dir)
    if not parts:
        return {}

    index_path = os.path.join(tar_dir, TAR_INDEX_NAME)
    print(f'Building tar index (streaming {cum[-1]/1e9:.1f} GB) …')
    print('  This takes ~3-5 min on first run; cached afterwards.')

    index = {}
    proc = subprocess.Popen(['cat'] + parts, stdout=subprocess.PIPE, bufsize=0)

    class _Counted(io.RawIOBase):
        """Wrap a raw pipe so tarfile can call .read(); tracks position."""
        def __init__(self, raw):
            self._raw = raw
            self.pos = 0
        def readable(self):
            return True
        def readinto(self, b):
            n = self._raw.readinto(b)
            if n:
                self.pos += n
            return n
        def read(self, n=-1):
            data = self._raw.read(n)
            self.pos += len(data)
            return data

    raw_io  = proc.stdout.raw if hasattr(proc.stdout, 'raw') else proc.stdout
    counted = _Counted(raw_io)
    buf_io  = io.BufferedReader(counted, buffer_size=8 * 1024 * 1024)

    try:
        with tarfile.open(fileobj=buf_io, mode='r|') as tf:
            for i, member in enumerate(tf):
                if member.isfile() and member.name.endswith('.mp4'):
                    # offset_data is set by tarfile in streaming mode
                    index[member.name] = [member.offset_data, member.size]
                if i % 5000 == 0 and i > 0:
                    print(f'  … {i} entries scanned', end='\r', flush=True)
    finally:
        proc.terminate()
        proc.wait()

    with open(index_path, 'w') as f:
        json.dump(index, f)
    print(f'\n  Index saved: {index_path}  ({len(index)} mp4 files)')
    return index


def load_tar_index(tar_dir):
    path = os.path.join(tar_dir, TAR_INDEX_NAME)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def extract_from_tar(vid, tar_dir, out_path):
    """
    Extract dataset/{vid}/{vid}.mp4 by seeking to its offset in the
    right part file.  Requires tar index.
    """
    index = load_tar_index(tar_dir)
    if index is None:
        index = build_tar_index(tar_dir)
    if not index:
        return False

    key = f'dataset/{vid}/{vid}.mp4'
    if key not in index:
        print(f'  [{vid}] not found in tar index')
        return False

    global_offset, size = index[key]
    parts, cum = _part_cumulative(tar_dir)
    if not parts:
        return False

    # find which part contains global_offset
    part_i = bisect.bisect_right(cum, global_offset) - 1
    part_i = max(0, min(part_i, len(parts) - 1))
    local_offset = global_offset - cum[part_i]

    print(f'  extracting from {os.path.basename(parts[part_i])}'
          f' @{local_offset/1e6:.1f}MB  ({size/1e6:.1f}MB)')

    remaining = size
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, 'wb') as out:
        pi = part_i
        off = local_offset
        while remaining > 0 and pi < len(parts):
            with open(parts[pi], 'rb') as pf:
                pf.seek(off)
                chunk = min(remaining, os.path.getsize(parts[pi]) - off)
                data = pf.read(chunk)
                out.write(data)
                remaining -= len(data)
            pi += 1
            off = 0   # subsequent parts start at offset 0

    return os.path.exists(out_path) and os.path.getsize(out_path) == size


# ── yt-dlp download ────────────────────────────────────────────────────────────

def download_clip_ytdlp(yt_id, t_start, t_end, out_path):
    if os.path.exists(out_path):
        return True
    yt_dlp = os.path.join(os.path.dirname(sys.executable), 'yt-dlp')
    if not os.path.exists(yt_dlp):
        yt_dlp = 'yt-dlp'
    duration = t_end - t_start
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, 'raw.mp4')
        r = subprocess.run([
            yt_dlp, '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            '--no-playlist', '-o', raw,
            f'https://www.youtube.com/watch?v={yt_id}',
        ], capture_output=True)
        if not os.path.exists(raw):
            print(f'  [yt-dlp failed] {r.stderr.decode()[:150]}')
            return False
        subprocess.run([
            'ffmpeg', '-y', '-ss', str(t_start), '-i', raw,
            '-t', str(duration), '-c', 'copy', out_path,
        ], capture_output=True)
    return os.path.exists(out_path)


# ── annotated video renderer ──────────────────────────────────────────────────

def make_annotated_video(vid, clip_path, data_path, result_file, out_path, fps=25):
    tf_path = os.path.join(data_path, vid, 'text.json')
    af_path = os.path.join(data_path, vid, 'answer.json')
    if not os.path.exists(tf_path) or not os.path.exists(af_path):
        print(f'  [skip] no QA files for {vid}')
        return False

    with open(tf_path) as f: text_data   = json.load(f)
    with open(af_path) as f: answer_data = json.load(f)

    predictions = None
    if result_file and os.path.exists(result_file):
        with open(result_file) as f:
            predictions = json.load(f)

    cap     = cv2.VideoCapture(clip_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vid_w = int(FRAME_H * src_w / src_h) if src_h else FRAME_H
    out_w = vid_w + PANEL_W

    # write to a temp file first, then re-encode to H.264 for broad compatibility
    tmp_path = out_path + '.tmp.mp4'
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    writer   = cv2.VideoWriter(tmp_path, fourcc, src_fps, (out_w, FRAME_H))

    src_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        src_frames.append(cv2.resize(frame, (vid_w, FRAME_H)))
    cap.release()

    if not src_frames:
        print(f'  [skip] no frames from {clip_path}')
        return False

    frames_per_qtype = max(1, int(src_fps * SECS_PER_QTYPE))
    panel = np.zeros((FRAME_H, PANEL_W, 3), dtype=np.uint8)

    for qtype_id in range(6):
        draw_panel(panel, qtype_id, text_data, answer_data, predictions, vid)
        for i in range(frames_per_qtype):
            src_frame = src_frames[i % len(src_frames)]
            writer.write(np.hstack([src_frame, panel]))

    writer.release()

    # re-encode to H.264 so VSCode / browsers can play it
    r = subprocess.run([
        'ffmpeg', '-y', '-i', tmp_path,
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
        out_path,
    ], capture_output=True)
    os.remove(tmp_path)
    if r.returncode != 0:
        print(f'  [ffmpeg error] {r.stderr.decode()[:200]}')
        return False

    print(f'  [saved] {out_path}')
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vid',          default=None)
    ap.add_argument('--split_path',   default=None)
    ap.add_argument('--n',            type=int, default=3)
    ap.add_argument('--seed',         type=int, default=42)
    ap.add_argument('--data_path',    default='./data/QA')
    ap.add_argument('--result_file',  default=None)
    ap.add_argument('--tar_dir',      default=None,
                    help='directory containing dataset.tar.[a-f]')
    ap.add_argument('--out_dir',      default='./viz_samples')
    ap.add_argument('--build_index',  action='store_true',
                    help='(re)build tar index and exit')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    clip_dir = os.path.join(args.out_dir, 'clips')
    os.makedirs(clip_dir, exist_ok=True)

    if args.build_index:
        if not args.tar_dir:
            ap.error('--tar_dir required with --build_index')
        build_tar_index(args.tar_dir)
        return

    if args.vid:
        vids = [args.vid]
    elif args.split_path:
        with open(args.split_path, 'rb') as f:
            all_vids = pickle.load(f)
        random.seed(args.seed)
        vids = random.sample(all_vids, min(args.n, len(all_vids)))
    else:
        ap.error('provide --vid or --split_path')

    for vid in vids:
        print(f'\n[{vid}]')
        clip_path = os.path.join(clip_dir, f'{vid}.mp4')

        if not os.path.exists(clip_path):
            # 1) try tar extraction
            if args.tar_dir:
                ok = extract_from_tar(vid, args.tar_dir, clip_path)
            else:
                ok = False
            # 2) fall back to yt-dlp
            if not ok:
                try:
                    yt_id, t_start, t_end = parse_vid_id(vid)
                    print(f'  downloading {yt_id} [{t_start}s–{t_end}s] via yt-dlp …')
                    ok = download_clip_ytdlp(yt_id, t_start, t_end, clip_path)
                except ValueError as e:
                    print(f'  {e}')
                    ok = False
            if not ok:
                print(f'  [skip] could not obtain clip')
                continue

        out_path = os.path.join(args.out_dir, f'{vid}_qa.mp4')
        make_annotated_video(vid, clip_path, args.data_path,
                             args.result_file, out_path)


if __name__ == '__main__':
    main()
