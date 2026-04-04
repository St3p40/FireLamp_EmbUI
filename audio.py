import asyncio
import numpy as np
import pyaudiowpatch as pyaudio
import websockets
import queue
import sys
import time
from scipy import signal

# Settings
ESP_IP = "192.168.2.21"
ESP_WS_URL = f"ws://{ESP_IP}/audio"

SEND_SAMPLES = 256

CENTER = 1024
RESOLUTION = 4096
TARGET_ESP_RATE = 18000
GAIN = 1.5  # Reduced to prevent clipping

MIN_SEND_INTERVAL = 0.01
RECONNECT_DELAY = 1

nyquist_freq = TARGET_ESP_RATE / 2 / TARGET_ESP_RATE
b, a = signal.butter(5, 0.45, btype='low')


async def stream_logic():
    p = pyaudio.PyAudio()
    audio_queue = queue.Queue(maxsize=5)
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        loopback = next(d for d in p.get_loopback_device_info_generator() if default_out["name"] in d["name"])

        system_rate = int(loopback["defaultSampleRate"])
        channels = loopback["maxInputChannels"]

        hw_buffer_size = int(SEND_SAMPLES * (system_rate / TARGET_ESP_RATE))
        print(f"[*] Windows Rate: {system_rate}Hz")
        print(f"[*] HW Buffer: {hw_buffer_size} | ESP Rate: {TARGET_ESP_RATE}Hz")
    except Exception as e:
        print(f"[ERR] {e}")
        return

    def pyaudio_callback(in_data, frame_count, time_info, status):
        raw = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        channels = loopback["maxInputChannels"]
        if channels > 1: raw = raw[::channels] * GAIN
        raw_filtered = signal.filtfilt(b, a, raw)

        factor = len(raw_filtered) // SEND_SAMPLES
        if factor > 1:
            processed_raw = raw_filtered[:SEND_SAMPLES*factor].reshape(-1, factor).mean(axis=1)
        else:
            processed_raw = raw_filtered[:SEND_SAMPLES]
        adc_values = (((processed_raw) + 32768) / 65535 * RESOLUTION).astype(np.uint16)
        adc_values = np.clip(adc_values, 0, RESOLUTION - 1)
        try:
            if audio_queue.full(): audio_queue.get_nowait()
            audio_queue.put_nowait(adc_values)
        except: pass
        return (None, pyaudio.paContinue)

    stream = p.open(format=pyaudio.paInt16,
                    channels=loopback["maxInputChannels"],
                    rate=system_rate,
                    input=True,
                    input_device_index=loopback["index"],
                    frames_per_buffer=hw_buffer_size,
                    stream_callback=pyaudio_callback)

    last_send_time = 0

    while True:
        try:
            async with websockets.connect(ESP_WS_URL, close_timeout=1, ping_interval=None) as ws:
                print(f"\r[OK] Connected to {ESP_WS_URL}             ")
                while stream.is_active():
                    current_time = time.time()
                    if not audio_queue.empty() and (current_time - last_send_time) >= MIN_SEND_INTERVAL:
                        adc_array = audio_queue.get_nowait()
                        try:
                            await ws.send(adc_array.tobytes())
                            last_send_time = current_time
                        except: break

                        viz = adc_array[::8]
                        fps = 1 / (time.time() - last_send_time + 0.0001)
                        display = ["\033[H"]
                        display.append(f"FPS: {int(fps)} | BUFF: {len(adc_array)} | Rate: {TARGET_ESP_RATE}Hz")

                        rows = 8
                        for r in range(rows, 0, -1):
                            threshold = (r / rows) * RESOLUTION
                            line = "".join(["█" if v >= threshold else ("=" if r == int(CENTER/(RESOLUTION/rows)) else " ") for v in viz])
                            display.append(line)

                        sys.stdout.write("\n".join(display) + "\n")
                        sys.stdout.flush()
                    else:
                        await asyncio.sleep(0.001)

        except Exception:
            print(f"\r[WAIT] Reconnecting in {RECONNECT_DELAY}s...", end="")
            await asyncio.sleep(RECONNECT_DELAY)

    p.terminate()

if __name__ == "__main__":
    print("\033[2J")
    try: asyncio.run(stream_logic())
    except KeyboardInterrupt: pass