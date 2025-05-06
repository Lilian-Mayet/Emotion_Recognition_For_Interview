import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Theme } from '@radix-ui/themes';

// Mock data for demonstration since browser can't access local ML models
const mockEmotions = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"];
const mockAudioEmotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"];

const EmotionDetectionApp = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState({ label: "neutral", confidence: 0.0 });
  const [audioEmotion, setAudioEmotion] = useState({ label: "neutral", confidence: 0.0 });

  useEffect(() => {
    let stream = null;
    let animationFrameId = null;

    const startWebcam = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsWebcamActive(true);
        }
      } catch (err) {
        console.error("Error accessing webcam:", err);
      }
    };

    // Mock emotion detection - in real app would use ML models
    const detectEmotions = () => {
      const randomEmotion = mockEmotions[Math.floor(Math.random() * mockEmotions.length)];
      const randomConfidence = Math.random();
      setCurrentEmotion({ label: randomEmotion, confidence: randomConfidence });
      
      const randomAudioEmotion = mockAudioEmotions[Math.floor(Math.random() * mockAudioEmotions.length)];
      const randomAudioConfidence = Math.random();
      setAudioEmotion({ label: randomAudioEmotion, confidence: randomAudioConfidence });
    };

    const drawVideoFrame = () => {
      if (videoRef.current && canvasRef.current && isWebcamActive) {
        const ctx = canvasRef.current.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
        
        // Simulate emotion detection every few frames
        if (Math.random() < 0.1) detectEmotions();
        
        animationFrameId = requestAnimationFrame(drawVideoFrame);
      }
    };

    startWebcam();

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, []);

  return (
    <div className="flex flex-col items-center gap-4 p-4 min-h-screen bg-gray-100">
      <Card className="w-full max-w-4xl">
        <CardHeader>
          <CardTitle>Emotion Detection</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
            <video 
              ref={videoRef}
              autoPlay
              playsInline
              className="absolute inset-0 w-full h-full object-cover"
            />
            <canvas 
              ref={canvasRef}
              className="absolute inset-0 w-full h-full"
              width={1280}
              height={720}
            />
            
            {/* Emotion Overlays */}
            <div className="absolute top-4 left-4 right-4 flex flex-col gap-2">
              <div className="bg-black/50 text-white p-2 rounded-lg backdrop-blur-sm">
                <p className="font-medium">
                  Facial Emotion: {currentEmotion.label} 
                  ({(currentEmotion.confidence * 100).toFixed(1)}%)
                </p>
              </div>
              <div className="bg-black/50 text-white p-2 rounded-lg backdrop-blur-sm">
                <p className="font-medium">
                  Audio Emotion: {audioEmotion.label}
                  ({(audioEmotion.confidence * 100).toFixed(1)}%)
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default EmotionDetectionApp;