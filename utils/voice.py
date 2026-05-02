import os
import asyncio
import edge_tts
import uuid
import tempfile

VOICE_MAP = {
    "en": "en-IN-NeerjaNeural", # Natural Indian English
    "hi": "hi-IN-SwaraNeural",  # Natural Hindi
    "gu": "gu-IN-DhwaniNeural"  # Natural Gujarati
}

async def generate_speech(text, lang="en", output_path=None):
    """Generates an MP3 file from text using Edge-TTS."""
    if not output_path:
        # Use /tmp for serverless environments, otherwise use a local temp folder
        if os.path.exists("/tmp") and os.access("/tmp", os.AccessAttr.W_OK if hasattr(os, "AccessAttr") else os.W_OK):
            temp_dir = "/tmp"
        else:
            temp_dir = os.path.join(os.getcwd(), "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
        output_path = os.path.join(temp_dir, f"tts_{uuid.uuid4()}.mp3")

    # Ensure output path is absolute and uses correct separators
    output_path = os.path.abspath(output_path)
    
    voice = VOICE_MAP.get(lang, VOICE_MAP["en"])
    
    # Advanced cleaning: remove common markdown and special characters that shouldn't be spoken
    import re
    clean_text = text
    clean_text = re.sub(r'[*#_~`>]', '', clean_text) # Remove common markdown
    clean_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean_text) # Replace links [text](url) with just text
    clean_text = clean_text.replace("-", " ").replace("\n", " ") # Replace newlines and hyphens with spaces
    
    # Check if there is any speakable text left to avoid edge_tts.exceptions.NoAudioReceived
    if not clean_text.strip() or not any(c.isalnum() for c in clean_text):
        print(f"  Warning: No speakable text found. Skipping TTS for: '{text[:20]}...'")
        return None

    try:
        communicate = edge_tts.Communicate(clean_text, voice)
        await communicate.save(output_path)
        return output_path
    except Exception as e:
        print(f"  TTS Error in generate_speech: {e}")
        raise e

if __name__ == "__main__":
    # Test with actual Gujarati text to verify it works
    test_text = "કુંવરબાઇનું મામેરું યોજના ગુજરાત સરકાર દ્વારા ચલાવવામાં આવે છે."
    asyncio.run(generate_speech(test_text, "gu", "test_gu.mp3"))
    print(f"Done. Saved to test_gu.mp3")
