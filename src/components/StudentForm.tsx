import React, { useState, useRef, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Mic, Search, Edit2 } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

// Server endpoints
const HLS_SERVER_URL = 'http://localhost:8001';
const API_BASE_URL = 'http://localhost:8001';

interface TranscriptionResponse {
  type: 'transcription' | 'error' | 'config_ack' | 'status' | 'ping';
  text?: string;
  start?: number;
  end?: number;
  message?: string;
  accumulated_seconds?: number;
}

interface Student {
  id: string;
  name: string;
  native_language: string;
  english_level: string;
  job_title?: string;
  hometown?: string;
  current_city?: string;
  interests?: string[];
  hobbies?: string[];
  learning_goals?: string[];
}

interface ExtractedInfo {
  name?: string;
  native_language?: string;
  english_level?: string;
  job_title?: string;
  hometown?: string;
  current_city?: string;
  interests?: string[];
  hobbies?: string[];
  learning_goals?: string[];
  [key: string]: any;
}

const StudentForm = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [transcription, setTranscription] = useState<string>('');
  const [editableTranscription, setEditableTranscription] = useState<string>('');
  const [processingStatus, setProcessingStatus] = useState('');
  const [hasRecorded, setHasRecorded] = useState(false);
  const [totalSeconds, setTotalSeconds] = useState(0);
  const [studentName, setStudentName] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [currentStudentId, setCurrentStudentId] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<Student[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [extractedInfo, setExtractedInfo] = useState<ExtractedInfo | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  
  const websocketRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const uidRef = useRef(crypto.randomUUID());
  const noSpeechTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    setFeedback('');
    setProcessingStatus('');
    return () => {
      stopRecording();
      if (websocketRef.current) {
        websocketRef.current.close();
      }
      if (noSpeechTimeoutRef.current) {
        clearTimeout(noSpeechTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (transcription) {
      setEditableTranscription(transcription);
    }
  }, [transcription]);

  const searchStudents = async (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    try {
      const response = await fetch(`${API_BASE_URL}/students/?skip=0&limit=5&search=${encodeURIComponent(query)}`);
      if (!response.ok) throw new Error('Failed to search students');
      const students = await response.json();
      setSearchResults(students);
    } catch (error) {
      console.error('Error searching students:', error);
      setFeedback('Error searching for students');
    } finally {
      setIsSearching(false);
    }
  };

  const selectExistingStudent = (student: Student) => {
    setStudentName(student.name);
    setCurrentStudentId(student.id);
    setSearchResults([]);
  };

  const previewAnalysis = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/analyze-text/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: editableTranscription,
        }),
      });

      if (!response.ok) throw new Error('Failed to analyze text');
      const analysis = await response.json();
      setExtractedInfo(analysis);
      setShowPreview(true);
    } catch (error) {
      console.error('Error analyzing text:', error);
      setFeedback('Error analyzing transcription');
    }
  };

  const createStudent = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/students/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: studentName,
          ...extractedInfo,
        }),
      });

      if (!response.ok) throw new Error('Failed to create student');
      const student: Student = await response.json();
      setCurrentStudentId(student.id);
      return student.id;
    } catch (error) {
      console.error('Error creating student:', error);
      setFeedback('Error creating student profile');
      throw error;
    }
  };

  const updateStudent = async (studentId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/students/${studentId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...extractedInfo,
        }),
      });

      if (!response.ok) throw new Error('Failed to update student');
      await response.json();
    } catch (error) {
      console.error('Error updating student:', error);
      setFeedback('Error updating student profile');
      throw error;
    }
  };

  const submitTranscription = async () => {
    if (!editableTranscription.trim() || !studentName.trim()) {
      setFeedback('Please enter student name and complete the recording');
      return;
    }

    setIsSubmitting(true);
    try {
      // Create or update student
      const studentId = currentStudentId || await createStudent();
      if (currentStudentId && extractedInfo) {
        await updateStudent(currentStudentId);
      }

      // Submit conversation
      const response = await fetch(`${API_BASE_URL}/conversations/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          student_id: studentId,
          transcription: editableTranscription,
          raw_audio_url: '',
        }),
      });

      if (!response.ok) throw new Error('Failed to submit conversation');

      setFeedback('Interview submitted successfully!');
      // Reset form
      setTranscription('');
      setEditableTranscription('');
      setStudentName('');
      setHasRecorded(false);
      setCurrentStudentId(null);
      setExtractedInfo(null);
      setShowPreview(false);
    } catch (error) {
      console.error('Error submitting interview:', error);
      setFeedback('Error submitting interview. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const initializeWebSocket = () => {
    console.log('Initializing WebSocket connection...');
    const ws = new WebSocket(`ws://localhost:8001/ws/audio/${uidRef.current}`);

    ws.onopen = () => {
      console.log('WebSocket connection established');
      setTotalSeconds(0); // Reset total seconds when starting new recording
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setFeedback('Error connecting to transcription server');
    };

    ws.onmessage = (event) => {
      const response: TranscriptionResponse = JSON.parse(event.data);
      console.log('WebSocket message received:', response);
      
      switch (response.type) {
        case 'ping':
          // Respond to server ping
          console.log('Received ping from server');
          break;
        case 'transcription':
          if (response.text) {
            console.log('Received transcription:', response.text);
            setTranscription(prev => prev + ' ' + response.text);
          }
          break;
        case 'status':
          if (response.message) {
            console.log('Received status:', response.message);
            // Update total seconds if available
            if (response.accumulated_seconds) {
              setTotalSeconds(response.accumulated_seconds);
            }
            
            // Update status for all messages
            const statusText = response.message === 'Processing audio...' || 
                             response.message === 'Starting transcription...'
              ? `${response.message} (Total: ${totalSeconds.toFixed(1)}s)`
              : response.message;
            setProcessingStatus(statusText);
            
            // Only handle no speech timeout if we're actively recording
            if (response.message === 'No speech detected in segment' && isRecording) {
              // Start/reset 60-second timeout for no speech
              if (noSpeechTimeoutRef.current) {
                clearTimeout(noSpeechTimeoutRef.current);
              }
              noSpeechTimeoutRef.current = setTimeout(() => {
                setFeedback('No speech detected for 60 seconds. Stopping recording.');
                stopRecording();
              }, 60000);
            } else {
              // Clear no speech timeout if we detect speech
              if (noSpeechTimeoutRef.current) {
                clearTimeout(noSpeechTimeoutRef.current);
              }
            }
          }
          break;
        case 'error':
          console.error('Received error:', response.message);
          setFeedback(response.message || 'Unknown error');
          break;
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setFeedback('Error connecting to transcription server. Please try again.');
      stopRecording();
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event);
      setProcessingStatus('');
      if (noSpeechTimeoutRef.current) {
        clearTimeout(noSpeechTimeoutRef.current);
      }
      if (isRecording) {
        setFeedback('Connection to transcription server lost. Please try again.');
        stopRecording();
      }
    };

    websocketRef.current = ws;
  };

  const startRecording = async () => {
    try {
      // Initialize WebSocket if not already connected
      if (!websocketRef.current || websocketRef.current.readyState !== WebSocket.OPEN) {
        initializeWebSocket();
      }

      // Get audio stream with specific constraints for Whisper
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,          // Mono
          sampleRate: 16000,        // 16 kHz
          sampleSize: 32,           // 32-bit float
          autoGainControl: false,   // Disable automatic gain control
          echoCancellation: false,  // Disable echo cancellation
          noiseSuppression: false,  // Disable noise suppression
        }
      });

      streamRef.current = stream;

      // Create audio context
      const audioContext = new AudioContext({
        sampleRate: 16000,
        latencyHint: 'interactive'
      });
      audioContextRef.current = audioContext;

      // Create audio source
      const source = audioContext.createMediaStreamSource(stream);

      // Create script processor for raw audio data
      // At 16kHz, 16384 samples = ~1 second of audio
      const processor = audioContext.createScriptProcessor(16384, 1, 1);
      processorRef.current = processor;

      // Process audio data
      processor.onaudioprocess = (e) => {
        if (websocketRef.current?.readyState === WebSocket.OPEN) {
          console.log('Processing audio chunk...');
          // Convert audio data to WAV format
          const audioData = e.inputBuffer.getChannelData(0);
          console.log('Audio data length:', audioData.length);
          
          // Create WAV header
          const buffer = new ArrayBuffer(44 + audioData.length * 2);
          const view = new DataView(buffer);
          console.log('Created buffer of size:', buffer.byteLength);
          
          // Write WAV header
          // "RIFF" chunk descriptor
          writeString(view, 0, 'RIFF');
          view.setUint32(4, 36 + audioData.length * 2, true);
          writeString(view, 8, 'WAVE');
          
          // "fmt " sub-chunk
          writeString(view, 12, 'fmt ');
          view.setUint32(16, 16, true); // fmt chunk size
          view.setUint16(20, 1, true); // audio format (PCM)
          view.setUint16(22, 1, true); // num channels (mono)
          view.setUint32(24, 16000, true); // sample rate
          view.setUint32(28, 16000 * 2, true); // byte rate
          view.setUint16(32, 2, true); // block align
          view.setUint16(34, 16, true); // bits per sample
          
          // "data" sub-chunk
          writeString(view, 36, 'data');
          view.setUint32(40, audioData.length * 2, true);
          
          // Write audio data
          const length = audioData.length;
          let index = 44;
          for (let i = 0; i < length; i++) {
            view.setInt16(index, audioData[i] * 32767, true);
            index += 2;
          }
          
          console.log('Sending audio data of size:', buffer.byteLength);
          websocketRef.current.send(buffer);
        }
      };

      // Connect the audio nodes
      source.connect(processor);
      processor.connect(audioContext.destination);

      setIsRecording(true);
      setHasRecorded(true);
      setFeedback('Recording started... Speak clearly into the microphone');

    } catch (error) {
      console.error('Error starting recording:', error);
      let errorMessage = 'Error accessing microphone. ';
      
      if (error instanceof Error) {
        errorMessage += error.message;
      } else {
        errorMessage += 'Please ensure microphone permissions are granted.';
      }
      
      setFeedback(errorMessage);
      setIsRecording(false);
      
      if (websocketRef.current) {
        websocketRef.current.close();
        websocketRef.current = null;
      }
    }
  };

  const stopRecording = () => {
    if (!hasRecorded) return;  // Don't process if we haven't recorded

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    setIsRecording(false);
    
    if (hasRecorded) {
      setFeedback('Recording stopped. Processing audio...');
    }

    // Clear any existing no-speech timeout
    if (noSpeechTimeoutRef.current) {
      clearTimeout(noSpeechTimeoutRef.current);
      noSpeechTimeoutRef.current = null;
    }

    // Keep connection open for 60 seconds for final processing
    setTimeout(() => {
      if (websocketRef.current) {
        websocketRef.current.close();
        websocketRef.current = null;
        if (hasRecorded) {
          setProcessingStatus(`Transcription complete (Total: ${totalSeconds.toFixed(1)}s)`);
          setFeedback('');
        }
      }
    }, 60000);
  };

  // Helper function to write strings to DataView
  const writeString = (view: DataView, offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto p-4 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl text-center">New Student Interview</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="studentName">Student Name</Label>
            <div className="flex gap-2">
              <Input 
                id="studentName" 
                placeholder="Enter or search student's name"
                className="text-lg"
                value={studentName}
                onChange={(e) => {
                  setStudentName(e.target.value);
                  searchStudents(e.target.value);
                }}
                disabled={isRecording || isSubmitting}
              />
              <Button
                variant="outline"
                size="icon"
                onClick={() => searchStudents(studentName)}
                disabled={isRecording || isSubmitting}
              >
                <Search className="h-4 w-4" />
              </Button>
            </div>
            
            {/* Search Results */}
            {searchResults.length > 0 && (
              <div className="mt-2 border rounded-md p-2 bg-background">
                {searchResults.map((student) => (
                  <button
                    key={student.id}
                    className="w-full text-left px-3 py-2 hover:bg-accent rounded-sm"
                    onClick={() => selectExistingStudent(student)}
                  >
                    {student.name} - {student.english_level}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="flex justify-center space-x-4">
            <Button 
              size="lg"
              className={`w-48 ${isRecording ? 'bg-red-500 hover:bg-red-600' : 'bg-blue-500 hover:bg-blue-600'}`}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={!studentName.trim() || isSubmitting}
            >
              <Mic className="mr-2 h-5 w-5" />
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </Button>

            {hasRecorded && !isRecording && (
              <>
                <Button
                  size="lg"
                  variant="outline"
                  onClick={() => setIsEditing(!isEditing)}
                  disabled={isSubmitting}
                >
                  <Edit2 className="mr-2 h-4 w-4" />
                  Edit Transcript
                </Button>
                <Button
                  size="lg"
                  onClick={previewAnalysis}
                  disabled={isSubmitting}
                >
                  Preview Analysis
                </Button>
              </>
            )}
          </div>

          {feedback && (
            <Alert>
              <AlertDescription>{feedback}</AlertDescription>
            </Alert>
          )}

          {hasRecorded && processingStatus && (
            <Alert className="bg-blue-50">
              <AlertDescription>{processingStatus}</AlertDescription>
            </Alert>
          )}

          {hasRecorded && (
            <div className="space-y-4">
              <div className="p-4 bg-muted rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <h3 className="font-semibold">Transcription:</h3>
                </div>
                {isEditing ? (
                  <Textarea
                    value={editableTranscription}
                    onChange={(e) => setEditableTranscription(e.target.value)}
                    className="min-h-[200px]"
                  />
                ) : (
                  <p>{editableTranscription}</p>
                )}
              </div>

              {showPreview && extractedInfo && (
                <div className="p-4 bg-muted rounded-lg">
                  <h3 className="font-semibold mb-4">Extracted Information:</h3>
                  <div className="grid grid-cols-2 gap-4">
                    {Object.entries(extractedInfo).map(([key, value]) => (
                      value && (
                        <div key={key} className="space-y-1">
                          <Label>{key.replace(/_/g, ' ').toUpperCase()}</Label>
                          <p className="text-sm">
                            {Array.isArray(value) ? value.join(', ') : value}
                          </p>
                        </div>
                      )
                    ))}
                  </div>
                </div>
              )}

              {showPreview && (
                <Button
                  size="lg"
                  className="w-full"
                  onClick={submitTranscription}
                  disabled={isSubmitting}
                >
                  {isSubmitting ? 'Submitting...' : 'Submit Interview'}
                </Button>
              )}
            </div>
          )}

          <div className="text-center text-sm text-gray-500">
            <p>Press "Start Recording" and begin the conversation.</p>
            <p>The app will automatically process the conversation and generate a customized worksheet.</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default StudentForm;
