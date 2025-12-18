import React, { useState, useRef, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  stepNumber?: number;
  imageUrl?: string;
}

interface Step {
  step_number: number;
  instruction: string;
  tips?: string;
  timestamp?: number;
}

interface Recipe {
  title: string;
  description?: string;
  difficulty?: string;
  total_time?: string;
  servings?: string;
  ingredients: Array<{ name: string; amount: string; unit: string; note?: string }>;
  steps: Step[];
  tips?: string[];
}

const API_BASE = '/api/chat';

// 요리 완료 상태
type CookingStatus = 'cooking' | 'finished';

export default function ChatRoom() {
  const location = useLocation();
  const navigate = useNavigate();
  const recipe: Recipe | null = location.state?.recipe || null;

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState(1);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showStepPanel, setShowStepPanel] = useState(true);
  const [cookingStatus, setCookingStatus] = useState<CookingStatus>('cooking');

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 스크롤 자동 이동
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // 세션 시작
  useEffect(() => {
    if (recipe && !sessionId) {
      startSession();
    }
  }, [recipe]);

  const startSession = async () => {
    if (!recipe) return;

    try {
      const response = await fetch(`${API_BASE}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ recipe })
      });
      const data = await response.json();
      setSessionId(data.session_id);

      // 환영 메시지
      setMessages([{
        role: 'assistant',
        content: `안녕! 오늘 **${recipe.title}** 만들어볼 거야 🍳\n\n총 ${recipe.steps.length}단계로 진행할게. 준비되면 **Step 1**부터 시작하자!\n\n궁금한 거 있으면 언제든 물어봐. 사진 찍어서 보여주면 피드백도 해줄게! 📸`
      }]);
    } catch (error) {
      console.error('세션 시작 실패:', error);
    }
  };

  // 이미지 선택
  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // 이미지 제거
  const removeImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // 메시지 전송
  const sendMessage = async () => {
    if (!inputText.trim() && !selectedImage) return;
    if (!sessionId) return;

    const userMessage: Message = {
      role: 'user',
      content: inputText,
      stepNumber: currentStep,
      imageUrl: imagePreview || undefined
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      let imageBase64 = null;
      if (selectedImage) {
        const reader = new FileReader();
        imageBase64 = await new Promise<string>((resolve) => {
          reader.onloadend = () => {
            const base64 = (reader.result as string).split(',')[1];
            resolve(base64);
          };
          reader.readAsDataURL(selectedImage);
        });
      }

      const response = await fetch(`${API_BASE}/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          step_number: currentStep,
          message: inputText || '이 사진 봐줘',
          image_base64: imageBase64
        })
      });

      const data = await response.json();

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.reply,
        stepNumber: currentStep
      }]);

      // 상태 업데이트
      if (data.session_status) {
        setCompletedSteps(data.session_status.completed_steps || []);
      }

    } catch (error) {
      console.error('메시지 전송 실패:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '앗, 오류가 발생했어. 다시 시도해줄래?'
      }]);
    } finally {
      setIsLoading(false);
      removeImage();
    }
  };

  // 단계 완료
  const completeCurrentStep = async () => {
    if (!sessionId) return;

    try {
      const response = await fetch(`${API_BASE}/session/${sessionId}/complete-step/${currentStep}`, {
        method: 'POST'
      });
      const data = await response.json();

      setCompletedSteps(prev => [...prev, currentStep]);

      if (data.is_finished) {
        setCookingStatus('finished');
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `🎉 **축하해! ${recipe?.title} 완성!**\n\n정말 잘했어! 맛있게 먹어 🍽️\n\n오늘 요리 어땠어?`
        }]);
      } else {
        setCurrentStep(data.next_step);
        const nextStepInfo = recipe?.steps[data.next_step - 1];
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `✅ **Step ${currentStep} 완료!**\n\n다음은 **Step ${data.next_step}**이야:\n> ${nextStepInfo?.instruction}\n\n${nextStepInfo?.tips ? `💡 팁: ${nextStepInfo.tips}` : ''}\n\n준비되면 시작해!`
        }]);
      }
    } catch (error) {
      console.error('단계 완료 실패:', error);
    }
  };

  // 단계 선택
  const selectStep = (stepNum: number) => {
    setCurrentStep(stepNum);
    const stepInfo = recipe?.steps[stepNum - 1];
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: `📍 **Step ${stepNum}**로 이동했어!\n\n> ${stepInfo?.instruction}\n\n${stepInfo?.tips ? `💡 팁: ${stepInfo.tips}` : ''}\n\n질문 있으면 말해줘!`
    }]);
  };

  // 엔터 키 처리
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  if (!recipe) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-lg mb-4">레시피 정보가 없습니다.</p>
          <button
            onClick={() => navigate('/')}
            className="px-4 py-2 gradient-bg rounded-full font-bold"
          >
            홈으로 돌아가기
          </button>
        </div>
      </div>
    );
  }

  const progress = recipe.steps.length > 0
    ? Math.round((completedSteps.length / recipe.steps.length) * 100)
    : 0;

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="h-16 flex items-center justify-between px-4 bg-white/80 backdrop-blur-sm border-b border-[var(--line)] sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate('/')}
            className="p-2 hover:bg-gray-100 rounded-full"
          >
            ←
          </button>
          <div>
            <h1 className="font-bold text-sm">{recipe.title}</h1>
            <p className="text-xs text-[var(--muted)]">
              Step {currentStep} / {recipe.steps.length} · {progress}% 완료
            </p>
          </div>
        </div>
        <button
          onClick={() => setShowStepPanel(!showStepPanel)}
          className="px-3 py-1.5 text-xs font-bold bg-gray-100 rounded-full"
        >
          {showStepPanel ? '단계 숨기기' : '단계 보기'}
        </button>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Step Panel */}
        {showStepPanel && (
          <aside className="w-72 border-r border-[var(--line)] bg-white overflow-y-auto">
            {/* Progress */}
            <div className="p-4 border-b border-[var(--line)]">
              <div className="flex justify-between text-xs mb-2">
                <span className="font-bold">진행률</span>
                <span>{progress}%</span>
              </div>
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full gradient-bg transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>

            {/* Steps */}
            <div className="p-2">
              {recipe.steps.map((step, idx) => {
                const stepNum = idx + 1;
                const isCompleted = completedSteps.includes(stepNum);
                const isCurrent = stepNum === currentStep;

                return (
                  <button
                    key={stepNum}
                    onClick={() => selectStep(stepNum)}
                    className={`w-full text-left p-3 rounded-xl mb-2 transition-all ${
                      isCurrent
                        ? 'gradient-bg-soft border-2 border-[rgba(69,197,138,.5)]'
                        : isCompleted
                        ? 'bg-green-50 border border-green-200'
                        : 'bg-gray-50 hover:bg-gray-100 border border-transparent'
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`w-6 h-6 rounded-lg flex items-center justify-center text-xs font-bold ${
                        isCompleted
                          ? 'bg-green-500 text-white'
                          : isCurrent
                          ? 'gradient-bg'
                          : 'bg-gray-300'
                      }`}>
                        {isCompleted ? '✓' : stepNum}
                      </span>
                      <span className="font-bold text-sm">Step {stepNum}</span>
                      {isCurrent && <span className="text-xs">👈 현재</span>}
                    </div>
                    <p className="text-xs text-[var(--muted)] line-clamp-2 ml-8">
                      {step.instruction}
                    </p>
                  </button>
                );
              })}
            </div>

            {/* Complete Step Button */}
            {!completedSteps.includes(currentStep) && (
              <div className="p-4 border-t border-[var(--line)]">
                <button
                  onClick={completeCurrentStep}
                  className="w-full py-3 gradient-bg rounded-xl font-bold text-sm hover:opacity-90 transition"
                >
                  ✅ Step {currentStep} 완료!
                </button>
              </div>
            )}
          </aside>
        )}

        {/* Chat Area */}
        <main className="flex-1 flex flex-col bg-gray-50">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl p-4 ${
                    msg.role === 'user'
                      ? 'bg-[var(--g-200)] rounded-br-sm'
                      : 'bg-white border border-[var(--line)] rounded-bl-sm'
                  }`}
                >
                  {msg.stepNumber && (
                    <span className="text-xs text-[var(--muted)] mb-1 block">
                      Step {msg.stepNumber}
                    </span>
                  )}
                  {msg.imageUrl && (
                    <img
                      src={msg.imageUrl}
                      alt="uploaded"
                      className="max-w-full rounded-lg mb-2 max-h-48 object-cover"
                    />
                  )}
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white border border-[var(--line)] rounded-2xl rounded-bl-sm p-4">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Image Preview */}
          {imagePreview && (
            <div className="px-4 py-2 border-t border-[var(--line)] bg-white">
              <div className="relative inline-block">
                <img
                  src={imagePreview}
                  alt="preview"
                  className="h-20 rounded-lg object-cover"
                />
                <button
                  onClick={removeImage}
                  className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full text-xs font-bold"
                >
                  ✕
                </button>
              </div>
            </div>
          )}

          {/* Input Area */}
          {cookingStatus === 'finished' ? (
            /* 완료 화면 */
            <div className="p-6 border-t border-[var(--line)] bg-gradient-to-br from-[var(--g-50)] to-[var(--o-50)]">
              <div className="text-center">
                <div className="text-4xl mb-3">🎉</div>
                <h3 className="font-black text-lg mb-2">요리 완성!</h3>
                <p className="text-sm text-[var(--muted)] mb-4">
                  {recipe?.title}을(를) 성공적으로 완성했어요!
                </p>
                <button
                  onClick={() => navigate('/')}
                  className="w-full py-3 gradient-bg rounded-xl font-bold text-sm hover:opacity-90 transition"
                >
                  🏠 홈으로 돌아가기
                </button>
              </div>
            </div>
          ) : (
            /* 요리 중 입력 영역 */
            <div className="p-4 border-t border-[var(--line)] bg-white">
              <div className="flex items-end gap-2">
                {/* Image Upload */}
                <input
                  type="file"
                  ref={fileInputRef}
                  accept="image/*"
                  onChange={handleImageSelect}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="p-3 bg-gray-100 rounded-xl hover:bg-gray-200 transition"
                  title="이미지 업로드"
                >
                  📷
                </button>

                {/* Text Input */}
                <div className="flex-1 relative">
                  <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder={`Step ${currentStep}에서 궁금한 거 물어봐!`}
                    rows={1}
                    className="w-full px-4 py-3 border border-[var(--line)] rounded-xl resize-none focus:outline-none focus:border-[rgba(69,197,138,.5)] text-sm"
                    style={{ minHeight: '48px', maxHeight: '120px' }}
                  />
                </div>

                {/* Send Button */}
                <button
                  onClick={sendMessage}
                  disabled={isLoading || (!inputText.trim() && !selectedImage)}
                  className="p-3 gradient-bg rounded-xl font-bold disabled:opacity-50 transition hover:opacity-90"
                >
                  ↑
                </button>
              </div>

              <p className="text-xs text-[var(--muted)] mt-2 text-center">
                📸 사진 찍어서 보내면 현재 Step {currentStep} 기준으로 피드백해줄게!
              </p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
