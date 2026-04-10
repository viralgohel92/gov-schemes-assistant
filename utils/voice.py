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
        # Save in the system's writable /tmp directory for serverless
        temp_dir = "/tmp"
        output_path = os.path.join(temp_dir, f"tts_{uuid.uuid4()}.mp3")
    
    voice = VOICE_MAP.get(lang, VOICE_MAP["en"])
    
    # We remove some markdown before speaking to make it sound better
    clean_text = text.replace("*", "").replace("#", "").replace("-", " ")
    
    communicate = edge_tts.Communicate(clean_text, voice)
    await communicate.save(output_path)
    return output_path

if __name__ == "__main__":
    # Test
    asyncio.run(generate_speech("નમસ્તે, કેમ છો?", "gu", "test_gu.mp3"))
    print("Done")
