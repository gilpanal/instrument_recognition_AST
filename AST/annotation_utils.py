LOWER_SPEECH_PRED_SCORE = 8
THRESHOLD_SPEECH_PRED_SCORE = 50
SILENCE_RMS_DB_THRESHOLD = -48

def format_bytes(size_bytes):
    size_bytes = int(size_bytes)
    if size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} Bytes"

def format_sample_rate(rate_hz):
    rate_hz = int(rate_hz)
    if rate_hz >= 1000:
        return f"{rate_hz / 1000:.1f} kHz"
    else:
        return f"{rate_hz} Hz"

def format_duration(duration_seconds):
    duration_seconds = float(duration_seconds)
    if duration_seconds >= 60:
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        return f"{minutes}m {seconds}s"
    else:
        return f"{duration_seconds:.2f} sec"

def format_bit_rate(bit_rate):
    bit_rate = int(bit_rate)
    if bit_rate >= 1_000_000:
        return f"{bit_rate / 1_000_000:.2f} Mbps"
    else:
        return f"{bit_rate / 1000:.2f} kbps"

def convertmetadata(toconvert):
    metadata = toconvert
    if metadata["size"] is not None:
        size_bytes = int(toconvert["size"])
        readable_size = format_bytes(size_bytes)
        metadata["size"] = readable_size
    
    if metadata["sample_rate"] is not None:
        sample_rate = toconvert["sample_rate"]
        formatted_rate = format_sample_rate(sample_rate)
        metadata["sample_rate"] = formatted_rate
    
    if metadata["duration"] is not None:
        duration = toconvert["duration"]
        formatted_duration = format_duration(duration)
        metadata["duration"] = formatted_duration
    
    if metadata["bit_rate"] is not None:
        bit_rate = toconvert["bit_rate"]
        formatted_rate = format_bit_rate(bit_rate)
        metadata["bit_rate"] = formatted_rate
    
    return metadata