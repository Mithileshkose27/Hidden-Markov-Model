import React, { useState, useRef } from 'react';
import { Mic, Square, Play, Volume2, Sparkles, Activity } from 'lucide-react';
import { PRESET_WORDS, generateSyntheticWordAudio } from '../utils/webAudio';

export default function AudioRecorder({ onAudioReady, activeWord, setActiveWord }) {
  const [isRecording, setIsRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const handlePresetClick = (word) => {
    setActiveWord(word);
    const audioData = generateSyntheticWordAudio(word);
    onAudioReady(audioData, word);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setAudioUrl(URL.createObjectURL(audioBlob));

        // Decode audio buffer for MFCC extraction
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const floatData = audioBuffer.getChannelData(0);

        setActiveWord('Custom Voice');
        onAudioReady(floatData, 'Custom Voice');
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      alert('Microphone access denied or not available. Using synthetic audio presets instead.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  return (
    <div className="card glass-panel padding-lg">
      <div className="flex-between margin-bottom-md">
        <div className="flex-align gap-sm">
          <Activity className="icon-primary animate-pulse" size={22} />
          <h2 className="card-title">1. Speech Input & Audio Processing</h2>
        </div>
        <span className="badge badge-accent">Interactive Studio</span>
      </div>

      <p className="text-secondary text-sm margin-bottom-md">
        Select a spoken vocabulary word to synthesize acoustic speech formants or record your own voice via microphone.
      </p>

      {/* Preset Word Selector */}
      <div className="margin-bottom-lg">
        <label className="label-text margin-bottom-xs">Target Speech Words:</label>
        <div className="grid-preset-words">
          {PRESET_WORDS.map((w) => (
            <button
              key={w}
              onClick={() => handlePresetClick(w)}
              className={`btn btn-preset ${activeWord === w ? 'btn-preset-active' : ''}`}
            >
              <Volume2 size={16} />
              <span>"{w.toUpperCase()}"</span>
            </button>
          ))}
        </div>
      </div>

      {/* Mic Recording Controls */}
      <div className="flex-align gap-md flex-wrap">
        {!isRecording ? (
          <button onClick={startRecording} className="btn btn-record">
            <Mic size={18} />
            <span>Record Microphone</span>
          </button>
        ) : (
          <button onClick={stopRecording} className="btn btn-stop animate-pulse">
            <Square size={18} />
            <span>Stop Recording</span>
          </button>
        )}

        {audioUrl && (
          <audio controls src={audioUrl} className="audio-player-custom" />
        )}
      </div>
    </div>
  );
}
