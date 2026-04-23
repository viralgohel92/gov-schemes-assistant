import asyncio
import edge_tts

async def test_tts():
    try:
        voice = "en-IN-NeerjaNeural"
        text = "Hello, this is a test of the edge T T S system."
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save("test_audio.mp3")
        print("Success: test_audio.mp3 created")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test_tts())
