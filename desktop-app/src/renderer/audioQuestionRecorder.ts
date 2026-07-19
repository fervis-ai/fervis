const OUTPUT_SAMPLE_RATE = 24_000;

export interface AudioQuestionRecording {
  readonly stop: () => Promise<Blob>;
  readonly cancel: () => void;
}

export async function startAudioQuestionRecording(): Promise<AudioQuestionRecording> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      autoGainControl: true,
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true
    }
  });
  const context = new AudioContext();
  const source = context.createMediaStreamSource(stream);
  const processor = context.createScriptProcessor(4096, 1, 1);
  const silentOutput = context.createGain();
  const chunks: Float32Array[] = [];
  let closed = false;

  silentOutput.gain.value = 0;
  processor.onaudioprocess = (event) => {
    chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  processor.connect(silentOutput);
  silentOutput.connect(context.destination);

  const close = (): void => {
    if (closed) {
      return;
    }
    closed = true;
    processor.disconnect();
    source.disconnect();
    silentOutput.disconnect();
    for (const track of stream.getTracks()) {
      track.stop();
    }
    void context.close();
  };

  return {
    cancel: close,
    stop: async () => {
      const samples = join(chunks);
      const normalized = resample(samples, context.sampleRate, OUTPUT_SAMPLE_RATE);
      close();
      return pcm16Wav(normalized, OUTPUT_SAMPLE_RATE);
    }
  };
}

function join(chunks: readonly Float32Array[]): Float32Array {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const output = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}

function resample(
  samples: Float32Array,
  inputRate: number,
  outputRate: number
): Float32Array {
  if (inputRate === outputRate || samples.length === 0) {
    return samples;
  }
  const output = new Float32Array(
    Math.max(1, Math.floor((samples.length * outputRate) / inputRate))
  );
  const ratio = inputRate / outputRate;
  for (let index = 0; index < output.length; index += 1) {
    const position = index * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, samples.length - 1);
    const fraction = position - left;
    output[index] = (samples[left] ?? 0) * (1 - fraction) + (samples[right] ?? 0) * fraction;
  }
  return output;
}

function pcm16Wav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index] ?? 0));
    view.setInt16(44 + index * 2, sample < 0 ? sample * 32768 : sample * 32767, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function writeAscii(view: DataView, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}
