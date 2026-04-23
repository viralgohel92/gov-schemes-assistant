import os
import asyncio
import edge_tts
import uuid

VOICE_MAP = {
    "en": "en-IN-NeerjaNeural", # Natural Indian English
    "hi": "hi-IN-SwaraNeural",  # Natural Hindi
    "gu": "gu-IN-DhwaniNeural"  # Natural Gujarati
}

async def generate_speech(text, lang="en", output_path=None):
    """Generates an MP3 file from text using Edge-TTS."""
    if not output_path:
        import tempfile
        # Use tempfile.gettempdir() which works on Windows and Vercel
        output_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4()}.mp3")

    
    voice = VOICE_MAP.get(lang, VOICE_MAP["en"])
    
    # We remove some markdown before speaking to make it sound better
    clean_text = text.replace("*", "").replace("#", "").replace("-", " ")
    
    # Check if there is any speakable text left to avoid edge_tts.exceptions.NoAudioReceived
    if not clean_text.strip() or not any(c.isalnum() for c in clean_text):
        print(f"  Warning: No speakable text found. Skipping TTS for: '{text[:20]}...'")
        return None

    communicate = edge_tts.Communicate(clean_text, voice)
    await communicate.save(output_path)
    return output_path

if __name__ == "__main__":
    # Test with actual Gujarati text to verify it works
    test_text = "કુંવરબાઇનું મામેરું યોજના ગુજરાત સરકાર દ્વારા ચલાવવામાં આવે છે."
    asyncio.run(generate_speech(test_text, "gu", "test_gu.mp3"))
    print(f"Done. Saved to test_gu.mp3")
