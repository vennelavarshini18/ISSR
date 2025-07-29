
def df_to_srt(df, output_file="subtitles.srt"):
    """
    Convert a DataFrame with speaker diarization data to an SRT subtitle file.
    
    Args:
        df (pandas.DataFrame): DataFrame with columns 'speaker_id', 'start_time', 
                              'end_time', and 'transcribed_content'
        output_file (str): Path to save the SRT file
    """
    def format_time(seconds_str):
        """Convert time string like '12.3s' to SRT format '00:00:12,300'"""
        seconds = float(seconds_str.rstrip('s'))
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            # Write subtitle index
            f.write(f"{idx + 1}\n")
            # Write timestamp
            start = format_time(row['start_time'])
            end = format_time(row['end_time'])
            f.write(f"{start} --> {end}\n")
            # Write speaker and transcription
            f.write(f"{row['speaker_id']}: {row['transcribed_content']}\n")
            # Write blank line
            f.write("\n")