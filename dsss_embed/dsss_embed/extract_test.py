import wave
import numpy as np


def bits_to_text(bits):
    chars = []
    for b in range(0, len(bits), 8):
        byte = bits[b:b + 8]
        if len(byte) < 8:
            break
        chars.append(chr(int(''.join(str(bit) for bit in byte), 2)))
    return ''.join(chars)


def generate_pn_sequence(length, seed=42):
    np.random.seed(seed)
    return np.random.choice([-1, 1], length)


def dsss_decode(spread_signal, pn_sequence):
    bit_chunks = spread_signal.reshape(-1, len(pn_sequence))
    bits = []
    for chunk in bit_chunks:
        correlation = np.dot(chunk, pn_sequence)
        bits.append(1 if correlation > 0 else 0)
    return bits


def extract_data(stego_wav, message_length, seed=42):
    with wave.open(stego_wav, 'rb') as wav:
        params = wav.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        frames = np.frombuffer(wav.readframes(nframes), dtype=np.int16)

    if nchannels == 2:
        frames = frames.reshape(-1, 2)
        samples = frames[:, 0]  # extract from left channel
    else:
        samples = frames

    pn = generate_pn_sequence(32, seed)
    spread_length = message_length * 8 * len(pn)

    if spread_length > len(samples):
        raise ValueError("Audio is too short or message length is incorrect.")

    lsb_samples = samples[:spread_length] & 1
    spread_signal = np.where(lsb_samples == 0, -1, 1)
    bits = dsss_decode(spread_signal, pn)
    message = bits_to_text(bits)

    print(f"📥 Extracted message: {message}")


# Run: truyền vào độ dài gốc của thông điệp (số ký tự)
extract_data('stego.wav', message_length=11, seed=42)

