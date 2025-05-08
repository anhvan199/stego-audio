import wave
import numpy as np


def text_to_bits(text):
    return [int(b) for char in text for b in format(ord(char), '08b')]


def generate_pn_sequence(length, seed=42):
    np.random.seed(seed)
    return np.random.choice([-1, 1], length)


def dsss_encode(bits, pn_sequence):
    spread_signal = []
    for bit in bits:
        spread_bit = (1 if bit == 1 else -1) * pn_sequence
        spread_signal.extend(spread_bit)
    return np.array(spread_signal)


def embed_data(input_wav, output_wav, message, seed=42):
    with wave.open(input_wav, 'rb') as wav:
        params = wav.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        frames = np.frombuffer(wav.readframes(nframes), dtype=np.int16).copy()

    # Reshape frames if stereo
    if nchannels == 2:
        frames = frames.reshape(-1, 2)
        target_channel = 0  # Embed into left channel
        audio_samples = frames[:, target_channel]
    else:
        audio_samples = frames

    bits = text_to_bits(message)
    pn = generate_pn_sequence(32, seed)
    spread = dsss_encode(bits, pn)

    print(f"[DEBUG] Spread bits: {len(spread)}")
    print(f"[DEBUG] Available samples: {len(audio_samples)}")

    if len(spread) > len(audio_samples):
        raise ValueError("Audio file is too short to hide the message.")

    # Embed DSSS bits into LSB of audio
    for i in range(len(spread)):
        audio_samples[i] = (audio_samples[i] & ~1) | (0 if spread[i] < 0 else 1)

    # Reassemble audio
    if nchannels == 2:
        frames[:, target_channel] = audio_samples
        frames = frames.reshape(-1)

    with wave.open(output_wav, 'wb') as wav:
        wav.setparams(params)
        wav.writeframes(frames.astype(np.int16).tobytes())

    print(f"✅ Data embedded into {output_wav}")


# Run
embed_data('cover.wav', 'stego.wav', 'Hello DSSS!', seed=42)

