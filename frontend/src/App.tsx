import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { analyzeVideo, getJobStatus, getResult, AnalysisResult } from './api/index';

function App() {
  const navigate = useNavigate();
  const [url, setUrl] = useState('');
  const [videoId, setVideoId] = useState<string | null>(null);
  const [, setJobId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [displayProgress, setDisplayProgress] = useState(0); // 화면에 보여줄 부드러운 progress
  const [message, setMessage] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [isSaved, setIsSaved] = useState(false);
  const [loopEnabled, setLoopEnabled] = useState(false);
  const [activeSegment, setActiveSegment] = useState<{ start: number; end: number } | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const loopTimerRef = useRef<number | null>(null);
  const progressAnimationRef = useRef<number | null>(null);

  // Toast 표시
  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2200);
  };

  // URL에서 video ID 추출
  const extractVideoId = (inputUrl: string): string | null => {
    const patterns = [
      /(?:youtube\.com\/shorts\/)([a-zA-Z0-9_-]+)/,
      /(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]+)/,
      /(?:youtu\.be\/)([a-zA-Z0-9_-]+)/,
    ];
    for (const pattern of patterns) {
      const match = inputUrl.match(pattern);
      if (match) return match[1];
    }
    return null;
  };

  // URL 변경 시 video ID 업데이트
  useEffect(() => {
    const id = extractVideoId(url);
    setVideoId(id);
  }, [url]);

  // Progress 부드러운 애니메이션
  useEffect(() => {
    // 기존 애니메이션 취소
    if (progressAnimationRef.current) {
      cancelAnimationFrame(progressAnimationRef.current);
      progressAnimationRef.current = null;
    }

    // progress가 0이면 displayProgress도 즉시 0으로
    if (progress === 0) {
      setDisplayProgress(0);
      return;
    }

    // 현재 displayProgress에서 목표 progress까지 부드럽게 증가
    const animate = () => {
      setDisplayProgress(current => {
        const diff = progress - current;

        // 목표에 도달하면 멈춤
        if (Math.abs(diff) < 0.5) {
          return progress;
        }

        // 차이의 일정 비율만큼 증가 (easing 효과)
        const increment = Math.max(0.3, diff * 0.08);
        const next = current + increment;

        // 아직 목표에 도달하지 않았으면 계속 애니메이션
        if (next < progress) {
          progressAnimationRef.current = requestAnimationFrame(animate);
        }

        return Math.min(next, progress);
      });
    };

    progressAnimationRef.current = requestAnimationFrame(animate);

    return () => {
      if (progressAnimationRef.current) {
        cancelAnimationFrame(progressAnimationRef.current);
      }
    };
  }, [progress]);

  // 구간 반복 로직
  useEffect(() => {
    if (loopTimerRef.current) {
      clearInterval(loopTimerRef.current);
      loopTimerRef.current = null;
    }

    if (loopEnabled && activeSegment && videoRef.current) {
      loopTimerRef.current = window.setInterval(() => {
        const video = videoRef.current;
        if (!video || !activeSegment) return;
        const t = video.currentTime || 0;
        if (t > activeSegment.end - 0.05) {
          video.currentTime = activeSegment.start;
          video.play().catch(() => {});
        }
      }, 80);
    }

    return () => {
      if (loopTimerRef.current) {
        clearInterval(loopTimerRef.current);
      }
    };
  }, [loopEnabled, activeSegment]);

  // 상태 폴링
  const pollStatus = useCallback(async (currentJobId: string) => {
    try {
      const status = await getJobStatus(currentJobId);
      setProgress(status.progress);
      setMessage(status.message);

      if (status.status === 'completed') {
        const analysisResult = await getResult(currentJobId);
        setResult(analysisResult);
        setIsLoading(false);
        showToast('레시피 추출 완료!');
      } else if (status.status === 'failed') {
        setError(status.message);
        setIsLoading(false);
        showToast('분석 중 오류가 발생했어요.');
      } else {
        setTimeout(() => pollStatus(currentJobId), 1000);
      }
    } catch (err) {
      setError('상태 확인 중 오류가 발생했습니다.');
      setIsLoading(false);
    }
  }, []);

  // 분석 시작
  const handleAnalyze = async () => {
    if (!url.trim()) {
      showToast('링크를 먼저 넣어줘!');
      return;
    }

    if (!videoId) {
      showToast('유효한 YouTube 링크가 아닌 것 같아.');
      return;
    }

    setError(null);
    setResult(null);
    setIsLoading(true);
    setProgress(0);
    setDisplayProgress(0);
    setMessage('분석 시작 중...');
    setIsSaved(false);
    setLoopEnabled(false);
    setActiveSegment(null);

    showToast('분석 중... STT → 요약 → 컷 추출');

    try {
      const response = await analyzeVideo(url);
      setJobId(response.job_id);
      pollStatus(response.job_id);
    } catch (err) {
      setError('분석 시작 중 오류가 발생했습니다.');
      setIsLoading(false);
      showToast('분석 시작 중 오류가 발생했어요.');
    }
  };

  // Enter 키 처리
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isLoading) {
      handleAnalyze();
    }
  };

  // 클립보드 붙여넣기
  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) {
        setUrl(text.trim());
        showToast('클립보드에서 붙여넣기 완료!');
      } else {
        showToast('클립보드가 비어있어.');
      }
    } catch (err) {
      showToast('브라우저 권한 때문에 실패. 직접 붙여넣어줘!');
    }
  };

  // 저장 토글
  const toggleSave = () => {
    setIsSaved(!isSaved);
    showToast(isSaved ? '저장을 해제했어.' : '레시피를 저장했어!');
  };

  // 구간 반복 토글
  const toggleLoop = () => {
    setLoopEnabled(!loopEnabled);
    showToast(!loopEnabled ? '구간 반복 ON' : '구간 반복 OFF');
  };

  // 타임라인 세그먼트 클릭
  const playSegment = (step: { timestamp?: number; step_number: number; instruction: string }) => {
    const start = step.timestamp || 0;
    const end = start + 10; // 기본 10초 구간

    setActiveSegment({ start, end });

    if (videoRef.current) {
      videoRef.current.currentTime = start;
      videoRef.current.play().catch(() => {});
    }

    if (loopEnabled) {
      showToast(`구간 반복 ON: ${formatTime(start)}~${formatTime(end)} · 단계 ${step.step_number}`);
    } else {
      showToast(`이어서 재생: ${formatTime(start)} · 단계 ${step.step_number}`);
    }
  };

  // 시간 포맷
  const formatTime = (seconds: number): string => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

  return (
    <div className="min-h-screen">
      {/* Topbar */}
      <header className="h-[72px] flex items-center justify-between px-[22px] sticky top-0 z-50 backdrop-blur-[10px] bg-white/[.78] border-b border-[var(--line)]">
        <div className="flex items-center gap-3 select-none">
          <div className="logo-gradient logo-shine w-10 h-10 rounded-[14px] shadow-[0_12px_22px_rgba(17,24,39,.10)] relative overflow-hidden" />
          <h1 className="text-base m-0 tracking-[-0.3px] flex gap-2 items-baseline font-black">
            오늘 뭐먹지
            <span className="text-xs px-[9px] py-[3px] rounded-full border border-[var(--line)] bg-white/85 text-[rgba(23,34,51,.70)] font-extrabold">
              beta
            </span>
          </h1>
        </div>

        <div className="flex items-center gap-[10px]">
          <button
            onClick={toggleSave}
            className="pill px-3 py-[10px] rounded-full border border-[var(--line)] bg-white/90 text-[rgba(23,34,51,.86)] text-[13px] flex gap-2 items-center cursor-pointer transition-all shadow-[var(--shadow2)] font-black hover:translate-y-[-1px] hover:bg-white/[.98] hover:shadow-[var(--shadow)]"
          >
            <span>{isSaved ? '✅' : '⭐'}</span> {isSaved ? '저장됨' : '저장'}
          </button>
        </div>
      </header>

      <main className="w-[min(1140px,calc(100%-32px))] mx-auto py-9 pb-24">
        {/* Hero Section */}
        <section className="mt-11 flex flex-col items-center text-center gap-[14px]">
          <h2 className="text-[46px] leading-[1.05] m-0 tracking-[-1px] font-black">
            쇼츠 레시피, <span className="gradient-text">한눈에 따라하기</span>
          </h2>
          <p className="m-0 text-[15px] text-[var(--muted)] max-w-[820px] leading-[1.65] font-semibold">
            유튜브 쇼츠 링크만 넣으면 재료·조리 순서·타임라인 컷을 정리해서 "따라 하기" 쉬운 형태로 보여줘요.
          </p>

          {/* Search Bar */}
          <div className="mt-6 w-[min(900px,100%)]">
            <div className={`flex items-center gap-[10px] px-[14px] py-[14px] rounded-full bg-white/[.96] border border-[var(--line)] shadow-[var(--shadow)] transition-all ${url ? 'border-[rgba(69,197,138,.45)] translate-y-[-1px] shadow-[0_18px_42px_rgba(17,24,39,.12)]' : ''}`}>
              <svg className="w-[22px] h-[22px]" viewBox="0 0 24 24" fill="none">
                <path d="M10.5 18.5a8 8 0 1 1 5.2-14.1A8 8 0 0 1 10.5 18.5Z" stroke="rgba(23,34,51,.70)" strokeWidth="1.7" />
                <path d="M16.6 16.6 21 21" stroke="rgba(23,34,51,.55)" strokeWidth="1.7" strokeLinecap="round" />
              </svg>
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="유튜브 쇼츠 링크를 붙여넣어 주세요 (https://youtube.com/shorts/...)"
                disabled={isLoading}
                className="flex-1 border-none outline-none bg-transparent text-[var(--ink)] text-[15px] p-1 font-semibold placeholder:text-[rgba(95,109,124,.95)] disabled:opacity-50"
              />
              <button
                onClick={handleAnalyze}
                disabled={isLoading}
                className="border-none cursor-pointer px-[14px] py-[10px] rounded-full text-[13px] text-[rgba(23,34,51,.92)] gradient-bg border border-[rgba(23,34,51,.08)] transition-all font-black flex items-center gap-2 whitespace-nowrap hover:translate-y-[-1px] hover:brightness-[1.02] active:translate-y-0 disabled:opacity-75 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <>⏳ 처리 중...</>
                ) : (
                  <>✨ 레시피 추출</>
                )}
              </button>
            </div>

            {/* Hint Row */}
            <div className="mt-3 flex justify-center gap-[10px] flex-wrap text-[rgba(95,109,124,.95)] text-[12.5px] font-semibold">
              <button
                onClick={handlePaste}
                className="px-[11px] py-2 rounded-full bg-white/[.92] border border-[var(--line)] cursor-pointer transition-all select-none shadow-[var(--shadow2)] font-extrabold hover:translate-y-[-1px] hover:bg-white/[.98] hover:shadow-[var(--shadow)]"
              >
                클립보드 붙여넣기
              </button>
            </div>
          </div>

          {/* Loading Progress */}
          {isLoading && (
            <div className="mt-4 w-[min(600px,100%)] text-center">
              <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full gradient-bg"
                  style={{ width: `${displayProgress}%` }}
                />
              </div>
              <p className="mt-2 text-sm text-[var(--muted)]">{message} ({Math.round(displayProgress)}%)</p>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-4 px-4 py-3 rounded-xl bg-red-50 text-red-600 text-sm">
              {error}
            </div>
          )}

          <p className="text-[rgba(95,109,124,.95)] text-[12.5px] font-semibold mt-2">
            타임라인 컷을 누르면 영상이 그 구간으로 이동하고, "구간 반복"을 켜면 해당 구간만 반복 재생해요.
          </p>
        </section>

        {/* Results Section */}
        {result && (
          <section className="mt-[38px] flex flex-col gap-[18px]">
            {/* Section Header */}
            <div className="sticky top-[calc(var(--stickyTop)+10px)] z-40 p-[14px_16px] rounded-2xl border border-[var(--line)] gradient-bg-soft shadow-[var(--shadow2)] flex items-center justify-between gap-3 backdrop-blur-[8px]">
              <div className="min-w-0 flex flex-col gap-[6px]">
                <div className="text-xs text-[rgba(23,34,51,.70)] font-black tracking-[.1px]">레시피 결과</div>
                <div className="flex items-center gap-[10px] flex-wrap min-w-0">
                  <h3 className="text-xl font-black tracking-[-0.5px] m-0 whitespace-nowrap overflow-hidden text-ellipsis max-w-[min(720px,72vw)]">
                    {result.recipe.title}
                  </h3>
                  <span className="px-[10px] py-[6px] rounded-full border border-[var(--line)] bg-white/[.92] text-[var(--muted)] text-xs font-black whitespace-nowrap">
                    {result.recipe.total_time || '?분'} · {result.recipe.servings || '1인분'}
                  </span>
                </div>
              </div>

              <div className="flex gap-[10px] items-center flex-wrap justify-end">
                <span className="px-[10px] py-[6px] rounded-full border border-[var(--line)] bg-white/[.92] text-[var(--muted)] text-xs font-black whitespace-nowrap">
                  난이도: {result.recipe.difficulty || '보통'}
                </span>
                <button
                  onClick={() => navigate('/chat', { state: { recipe: result.recipe } })}
                  className="gradient-bg shadow-[var(--shadow2)] cursor-pointer px-4 py-[10px] rounded-full font-black text-[13px] flex items-center gap-2 transition-all text-[rgba(23,34,51,.90)] select-none hover:translate-y-[-1px] hover:brightness-[1.05] hover:shadow-[var(--shadow)]"
                >
                  💬 요리 시작하기
                </button>
                <button
                  onClick={toggleSave}
                  className={`border border-[var(--line)] ${isSaved ? 'gradient-bg border-[rgba(23,34,51,.08)]' : 'bg-white/[.92]'} shadow-[var(--shadow2)] cursor-pointer px-3 py-[10px] rounded-full font-black text-[13px] flex items-center gap-2 transition-all text-[rgba(23,34,51,.90)] select-none hover:translate-y-[-1px] hover:bg-white/[.98] hover:shadow-[var(--shadow)]`}
                >
                  <svg className="w-[18px] h-[18px]" viewBox="0 0 24 24" fill="none">
                    <path d="M6.5 4.5h11v16l-5.5-3-5.5 3v-16Z" stroke="rgba(23,34,51,.78)" strokeWidth="1.7" strokeLinejoin="round" />
                  </svg>
                  <span>{isSaved ? '저장됨' : '저장'}</span>
                </button>
              </div>
            </div>

            {/* Recipe Card */}
            <div className="bg-[var(--card)] border border-[var(--line)] rounded-[var(--radius)] shadow-[var(--shadow)] overflow-hidden">
              <div className="p-[16px_18px] flex items-center justify-between border-b border-[rgba(23,34,51,.08)] gradient-bg-soft">
                <h3 className="m-0 text-sm tracking-[.1px] text-[rgba(23,34,51,.92)] flex items-center gap-[10px] font-black">
                  📋 레시피 결과
                </h3>
                <span className="text-xs text-[rgba(23,34,51,.72)] border border-[var(--line)] px-[10px] py-[5px] rounded-full bg-white/[.88] font-black whitespace-nowrap">
                  요약
                </span>
              </div>
              <div className="p-[18px]">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-[14px]">
                  {/* Ingredients */}
                  <div>
                    <h5 className="m-0 mb-[10px] text-[14.5px] font-black tracking-[-0.1px]">🧺 재료</h5>
                    <ul className="m-0 pl-[18px] font-semibold">
                      {result.recipe.ingredients?.map((ing, idx) => (
                        <li key={idx} className="my-2">
                          {ing.name} {ing.amount}{ing.unit}
                          {ing.note && <span className="text-[var(--muted)] text-sm"> ({ing.note})</span>}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Steps */}
                  <div>
                    <h5 className="m-0 mb-[10px] text-[14.5px] font-black tracking-[-0.1px]">🧑‍🍳 조리 순서</h5>
                    <div className="flex flex-col gap-[10px]">
                      {result.recipe.steps?.map((step, idx) => (
                        <div key={idx} className="p-3 rounded-[14px] border border-[var(--line)] bg-white/[.96] flex gap-3 items-start">
                          <div className="min-w-7 h-7 rounded-[10px] grid place-items-center text-xs font-black text-[rgba(23,34,51,.92)] gradient-bg border border-[rgba(23,34,51,.08)]">
                            {step.step_number}
                          </div>
                          <div>
                            <p className="m-0 text-[13.5px] leading-[1.35] font-semibold">{step.instruction}</p>
                            {step.tips && <p className="m-0 mt-1 text-xs text-[var(--muted)]">💡 {step.tips}</p>}
                            {step.timestamp !== undefined && (
                              <div className="text-xs text-[rgba(95,109,124,.98)] mt-1 font-extrabold">
                                {formatTime(step.timestamp)}
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Tips */}
                {result.recipe.tips && result.recipe.tips.length > 0 && (
                  <div className="mt-4 p-3 rounded-[14px] border border-[var(--line)] bg-[var(--o-50)]">
                    <h5 className="m-0 mb-2 text-[14px] font-black">💡 요리 팁</h5>
                    <ul className="m-0 pl-[18px] font-semibold text-sm">
                      {result.recipe.tips.map((tip, idx) => (
                        <li key={idx} className="my-1">{tip}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>

            {/* Video + Timeline Card */}
            <div className="bg-[var(--card)] border border-[var(--line)] rounded-[var(--radius)] shadow-[var(--shadow)] overflow-hidden">
              <div className="p-[16px_18px] flex items-center justify-between border-b border-[rgba(23,34,51,.08)] gradient-bg-soft">
                <h3 className="m-0 text-sm tracking-[.1px] text-[rgba(23,34,51,.92)] flex items-center gap-[10px] font-black">
                  🎬 본 숏츠 & 타임라인 컷
                </h3>
                <span className="text-xs text-[rgba(23,34,51,.72)] border border-[var(--line)] px-[10px] py-[5px] rounded-full bg-white/[.88] font-black whitespace-nowrap">
                  반복: {loopEnabled ? 'ON' : 'OFF'}
                </span>
              </div>
              <div className="p-[18px]">
                <div className="grid grid-cols-1 lg:grid-cols-[1.1fr_.9fr] gap-3 items-start">
                  {/* Video */}
                  <div className="border border-[var(--line)] rounded-2xl overflow-hidden bg-white shadow-[var(--shadow2)]">
                    <div className="p-[10px_12px] flex items-center justify-between gap-[10px] border-b border-[rgba(23,34,51,.08)] gradient-bg-soft">
                      <p className="text-[13px] font-black m-0">본 숏츠</p>
                      <div className="flex items-center gap-[10px] flex-wrap text-[rgba(23,34,51,.78)] text-xs font-black">
                        <button
                          onClick={toggleLoop}
                          className={`flex items-center gap-2 px-[10px] py-[6px] rounded-full border border-[var(--line)] bg-white/[.92] cursor-pointer select-none ${loopEnabled ? 'bg-gradient-to-br from-[var(--g-200)] to-[var(--o-200)]' : ''}`}
                        >
                          <span className={`w-[10px] h-[10px] rounded-full border border-[rgba(23,34,51,.12)] transition-all ${loopEnabled ? 'gradient-bg' : 'bg-[rgba(95,109,124,.35)]'}`} />
                          구간 반복
                        </button>
                      </div>
                    </div>
                    {videoId ? (
                      <iframe
                        className="w-full aspect-[9/16]"
                        src={`https://www.youtube.com/embed/${videoId}?playsinline=1`}
                        title="YouTube video"
                        frameBorder="0"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowFullScreen
                      />
                    ) : (
                      <div className="w-full aspect-[9/16] bg-black flex items-center justify-center text-white/50">
                        영상 없음
                      </div>
                    )}
                  </div>

                  {/* Timeline */}
                  <div className="flex flex-col gap-[10px]">
                    {result.recipe.steps?.map((step, idx) => (
                      <div
                        key={idx}
                        onClick={() => playSegment(step)}
                        className="flex gap-[10px] p-[10px] rounded-2xl border border-[var(--line)] bg-white/[.96] transition-all cursor-pointer shadow-[var(--shadow2)] hover:translate-y-[-1px] hover:shadow-[var(--shadow)] hover:bg-white/[.99]"
                      >
                        <div className="w-[92px] h-16 rounded-[14px] border border-[var(--line)] relative flex-shrink-0 gradient-bg overflow-hidden">
                          <div className="absolute left-2 bottom-2 text-[11px] px-[7px] py-[3px] rounded-full text-[rgba(23,34,51,.86)] bg-white/90 border border-[var(--line)] font-black">
                            {step.timestamp !== undefined ? formatTime(step.timestamp) : `#${step.step_number}`}
                          </div>
                        </div>
                        <div className="flex-1 flex flex-col gap-1 pt-[1px]">
                          <p className="text-[12.8px] font-black m-0">단계 {step.step_number}</p>
                          <p className="m-0 text-xs text-[rgba(95,109,124,.98)] leading-[1.35] font-semibold line-clamp-2">
                            {step.instruction}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Toast */}
      {toast && (
        <div className="fixed left-1/2 bottom-[26px] -translate-x-1/2 bg-[rgba(23,34,51,.90)] text-white/[.95] px-3 py-[10px] rounded-[14px] shadow-[0_18px_40px_rgba(17,24,39,.16)] text-[13px] max-w-[min(560px,calc(100%-24px))] text-center font-extrabold toast-animate">
          {toast}
        </div>
      )}
    </div>
  );
}

export default App;
