import json
import pyaudio
import torch
import time
import wave
import zmq
from dataclasses import dataclass

from vap.model import VapGPT, VapConfig


"""
* Listen to stereo audio input
* Add audio into a queue/deque/np.array
    * of a certain size (20s)
* Loop that process the audio input with the model
* Send actions/probs throuch zmk to target path

TODO:
* [ ] Audio + tensor update working
* [ ] Model forward
* [ ] ZMK working

Later Todos:
* [ ] torch.compile
"""

NORM_FACTOR: float = 1 / (2 ** 15)


@dataclass
class SDSConfig:
    # audio
    frame_length: float = 0.02  # time (seconds) of each frame of audio
    sample_width: int = 2
    sample_rate: int = 16_000

    # Model
    context: int = 20  # The size of the audio processed by the model
    state_dict: str = "../example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"

    # TurnTaking
    tt_time: float = 0.5

    # ZMK
    port: int = 5578
    topic: str = "tt_probs"


# TODO: dynamic conf
def load_model(state_dict_path):
    model_conf = VapConfig()
    model = VapGPT(model_conf)
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    return model.eval()


class AudioInputStereo:
    def __init__(
        self,
        filename="data/sds_audio.wav",
        frame_length=0.02,
        sample_width=2,
        sample_rate=16_000,
        channels=2,
        device=None,
        **kwargs,
    ):
        """Initialize the Microphone Module. Args:
        frame_length (float): The length of one frame (i.e., IU) in seconds
        rate (int): The frame rate of the recording
        sample_width (int): The width of a single sample of audio in bytes.
        """
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.sample_width = sample_width
        self.sample_rate = sample_rate
        self.channels = channels
        self.filename = filename
        # self.audio_buffer = []
        self.audio_buffer = b""

        # Pyaudio
        self._p = pyaudio.PyAudio()

        self.device = device
        self.device_index = self.get_device_index(device)
        self.device_info = self._p.get_device_info_by_index(self.device_index)
        self.chunk_size = round(self.sample_rate * frame_length)
        self.stream = self._p.open(
            format=self._p.get_format_from_width(self.sample_width),
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            output=False,
            stream_callback=self.callback,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.device_index,
            start=False,
        )

        # Record audio
        self.wavfile = wave.open(self.filename, "wb")
        self.wavfile.setframerate(self.sample_rate)
        self.wavfile.setnchannels(2)
        self.wavfile.setsampwidth(self.sample_width)

    def __repr__(self):
        s = "MicrophoneModule"
        s += f"\nSampleRate: {self.sample_rate}"
        s += f"\nFrame length: {self.frame_length}"
        s += f"\nSample width: {self.sample_width}"
        s += f"\nChannels: {self.channels}"
        s += f"\nDevice: {self.device}"
        s += f"\nDevice index: {self.device_index}"
        return s

    def get_audio_buffer(self):
        a = self.audio_buffer
        self.audio_buffer = b""
        return a

    def get_device_index(self, device):
        if device is None:
            info = self._p.get_default_input_device_info()
            return int(info["index"])

        bypass_index = -1
        for i in range(self._p.get_device_count()):
            info = self._p.get_device_info_by_index(i)
            if info["name"] == device:
                bypass_index = i
                break
        return bypass_index

    def callback(self, in_data, frame_count, time_info, status):
        """The callback function that gets called by pyaudio.
        Args:
            in_data (bytes[]): The raw audio that is coming in from the
                microphone
            frame_count (int): The number of frames that are stored in in_data
        """
        self.audio_buffer += in_data
        self.wavfile.writeframes(in_data)
        # self.audio_buffer.append(in_data)
        return (in_data, pyaudio.paContinue)

    def start_stream(self):
        self.stream.start_stream()
        print("Started audio stream")

    def stop_stream(self):
        """Close the audio stream."""
        self.stream.stop_stream()
        self.stream.close()
        self.wavfile.close()
        print("Stopped audio stream")
        print("Closed audio file")


class TurnTakingSDS:
    def __init__(
        self,
        conf,
        audio_file: str = "data/audio.wav",
        probs_file: str = "data/probs.txt",
    ):
        self.conf = conf

        self.probs_txt_file = open(probs_file, "w")
        self.model = load_model(conf.state_dict)
        n_samples = round(conf.context * conf.sample_rate)
        self.x = torch.zeros((1, 2, n_samples))
        self.device = "cpu"
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.x = self.x.to("cuda")
            self.device = "cuda"
            print("Moved to CUDA")

        # The number of frames to average the turn-shift probabiltites in
        self.tt_frames = round(conf.tt_time * self.model.frame_hz)

        self.audio_in = AudioInputStereo(
            filename=audio_file,
            frame_length=conf.frame_length,
            sample_width=conf.sample_width,
            sample_rate=conf.sample_rate,
        )

        # Setup zmk
        socket_ip = f"tcp://*:{conf.port}"
        self.socket = zmq.Context().socket(zmq.PUB)
        self.socket.bind(socket_ip)

        # socket_ip = f"tcp://130.229.191.206:{conf.port}"
        # socket_ip = "tcp://130.229.191.206:5558"
        # socket_ip = "tcp://130.229.191.206:5558"
        # socket_ip = "tcp://130.229.191.206:5578"
        # socket_ip = f"tcp://*:{conf.port}"
        # socket_ip = f"130.229.191.206:{conf.port}"
        # self.socket = zmq.Context().socket(zmq.PUB)
        # self.socket.bind(socket_ip)

    def add_audio_bytes_to_tensor(
        self, audio_bytes: bytes, norm_factor: float = NORM_FACTOR
    ) -> None:

        chunk = torch.frombuffer(audio_bytes, dtype=torch.int16).float() * norm_factor

        # Split stereo audio
        a = chunk[::2]
        b = chunk[1::2]
        chunk_size = a.shape[0]

        # Move values back
        self.x = self.x.roll(-chunk_size, -1)
        self.x[0, 0, -chunk_size:] = a.to(self.device)
        self.x[0, 1, -chunk_size:] = b.to(self.device)

    @torch.no_grad()
    def run(self):

        self.audio_in.start_stream()
        start_time = time.time()

        try:
            while True:
                # Get new data from stream
                audio_bytes = self.audio_in.get_audio_buffer()
                if len(audio_bytes) == 0:
                    continue

                # update tensor X
                self.add_audio_bytes_to_tensor(audio_bytes)
                a = (self.x[0, 0, -4000:].abs().mean() * 100).long().item()
                b = (self.x[0, 1, -4000:].abs().mean() * 100).long().item()
                print("Audio ->   A: ", a, "B: ", b)

                # feed through model
                out = self.model.probs(self.x)
                p = out["p_now"][0, -self.tt_frames :, 0].mean().item()
                # d = {"now": p}

                # send through zmk
                self.socket.send_string(self.conf.topic, zmq.SNDMORE)
                # self.socket.send(json.dumps(d).encode())
                self.socket.send(json.dumps(p).encode())  # send a single float

                cur_time = time.time() - start_time
                self.probs_txt_file.write(f"{cur_time} {p}\n")

        except KeyboardInterrupt:
            print("Abort by user KEYBOARD")

        # close
        self.audio_in.stop_stream()
        self.socket.close()
        self.probs_txt_file.close()
        print("Closed socket")


if __name__ == "__main__":
    conf = SDSConfig(context=20)
    for k, v in conf.__dict__.items():
        print(f"{k}: {v}")

    sds = TurnTakingSDS(conf)

    sds.run()
