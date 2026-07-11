import os
import re
import sys
import glob
from collections import Counter

# Fix console encoding on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

TRANSCRIPTS_DIR = os.path.join("observations", "outputs")

# Classify text as a hallucination based on known Whisper heuristics.
KNOWN_EXACT_HALLUCINATIONS = {
    "you", "you.", "you!", "you?", "you...",
    "i'm", "i'm.", "i'm...",
    "[unintelligible]", "[silence]", "(music)",
    "what?", "what.",
}

KNOWN_SUBSTRING_HALLUCINATIONS = [
    "thank you for watching", "subs by", "www.", "http", "amara.org",
    "zeoranger", "subtitles", "thank you.", "thank you!"
]

def is_hallucination(text, duration):
    text_lower = text.strip().lower()
    
    # Check exact matches against typical silence artifacts.
    if text_lower in KNOWN_EXACT_HALLUCINATIONS:
        return True
        
    # Check substrings
    for sub in KNOWN_SUBSTRING_HALLUCINATIONS:
        if sub in text_lower:
            # 'thank you' is context-dependent based on segment duration.
            if sub in ["thank you.", "thank you!"] and duration > 1.5:
                continue
            return True
            
    # Filter anomalous short durations.
    if duration <= 0.1 and len(text_lower.split()) <= 2:
        if text_lower not in ["yeah", "yeah.", "ok", "ok.", "okay", "okay."]:
            return True
            
    return False

def analyze_transcripts():
    files = glob.glob(os.path.join(TRANSCRIPTS_DIR, "transcript_*.txt"))
    
    if not files:
        print("No transcripts found.")
        return
        
    total_meetings = len(files)
    all_hallucinations = Counter()
    total_segments_all = 0
    total_hallucinations_all = 0
    
    # Time context counters
    context_stats = {
        "isolated_silence": 0, # Gap before > 2s AND gap after > 2s
        "trailing_speech": 0,  # Gap before < 2s AND gap after > 2s
        "leading_speech": 0,   # Gap before > 2s AND gap after < 2s
        "mid_speech": 0        # Gap before < 2s AND gap after < 2s
    }

    print(f"--- Phase 3.5: Hallucination Measurement Engine ---")
    print(f"Analyzing {total_meetings} meetings...\n")
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        segments = []
        pattern = re.compile(r'\[(\d+\.\d+) - (\d+\.\d+)\] (SPEAKER_\w+) \((RAW|DFN)\): (.*)')
        
        for line in lines:
            # Strip optional line number prefixes
            line = re.sub(r'^\d+:\s+', '', line)
            
            match = pattern.match(line.strip())
            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                speaker = match.group(3)
                source = match.group(4)
                text = match.group(5)
                duration = end - start
                
                segments.append({
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "speaker": speaker,
                    "text": text,
                    "is_hallucination": is_hallucination(text, duration)
                })
        
        file_total = len(segments)
        total_segments_all += file_total
        file_hallucinations = 0
        
        for i, seg in enumerate(segments):
            if seg["is_hallucination"]:
                file_hallucinations += 1
                total_hallucinations_all += 1
                all_hallucinations[seg["text"].strip()] += 1
                
                # Analyze time context
                gap_before = 999.0
                gap_after = 999.0
                
                if i > 0:
                    gap_before = seg["start"] - segments[i-1]["end"]
                if i < len(segments) - 1:
                    gap_after = segments[i+1]["start"] - seg["end"]
                    
                if gap_before > 2.0 and gap_after > 2.0:
                    context_stats["isolated_silence"] += 1
                elif gap_before <= 2.0 and gap_after > 2.0:
                    context_stats["trailing_speech"] += 1
                elif gap_before > 2.0 and gap_after <= 2.0:
                    context_stats["leading_speech"] += 1
                else:
                    context_stats["mid_speech"] += 1
                    
        percent = (file_hallucinations / file_total * 100) if file_total > 0 else 0
        print(f"{os.path.basename(file)}:")
        print(f"  Total Segments: {file_total}")
        print(f"  Hallucinations: {file_hallucinations} ({percent:.2f}%)")
        
    print("\n--- Aggregate Statistics ---")
    print(f"Total Segments Across All Meetings: {total_segments_all}")
    print(f"Total Hallucinations: {total_hallucinations_all}")
    if total_segments_all > 0:
        print(f"Overall Hallucination Rate: {(total_hallucinations_all / total_segments_all * 100):.2f}%")
        
    print("\n--- Most Common Hallucinated Phrases ---")
    for phrase, count in all_hallucinations.most_common(10):
        print(f"  {count}x : '{phrase}'")
        
    print("\n--- Time Context (Where do they occur?) ---")
    print(f"  Isolated Silence (Surrounded by >2s silence): {context_stats['isolated_silence']}")
    print(f"  Trailing Speech (At the end of a speech block): {context_stats['trailing_speech']}")
    print(f"  Leading Speech (At the start of a speech block): {context_stats['leading_speech']}")
    print(f"  Mid-Speech (Embedded within continuous speech): {context_stats['mid_speech']}")

if __name__ == "__main__":
    analyze_transcripts()
