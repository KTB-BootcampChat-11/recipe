import React from 'react';
import { AnalysisResult } from '../api';
import { Clock, Users, ChefHat, Lightbulb } from 'lucide-react';

interface RecipeResultProps {
  result: AnalysisResult | null;
  jobId: string | null;
  isLoading: boolean;
  progress: number;
  message: string;
}

const RecipeResult: React.FC<RecipeResultProps> = ({
  result,
  jobId,
  isLoading,
  progress,
  message,
}) => {
  // 로딩 상태
  if (isLoading) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-8">
        <div className="w-full max-w-md">
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-600 mb-2">
              <span>{message}</span>
              <span>{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className="bg-primary-500 h-3 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500" />
          </div>
        </div>
      </div>
    );
  }

  // 결과 없음
  if (!result) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-gray-400 p-8">
        <ChefHat className="w-16 h-16 mb-4" />
        <p className="text-lg">레시피 분석 결과가 여기에 표시됩니다</p>
        <p className="text-sm mt-2">YouTube Shorts URL을 입력하고 분석하기를 눌러주세요</p>
      </div>
    );
  }

  const { recipe } = result;

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="h-full overflow-y-auto p-6">
      {/* 제목 */}
      <h1 className="text-3xl font-bold text-gray-800 mb-2">{recipe.title}</h1>

      {/* 설명 */}
      {recipe.description && (
        <p className="text-gray-600 mb-4">{recipe.description}</p>
      )}

      {/* 기본 정보 */}
      <div className="flex flex-wrap gap-4 mb-6">
        {recipe.servings && (
          <div className="flex items-center gap-2 text-gray-600 bg-gray-100 px-3 py-1 rounded-full">
            <Users className="w-4 h-4" />
            <span className="text-sm">{recipe.servings}</span>
          </div>
        )}
        {recipe.total_time && (
          <div className="flex items-center gap-2 text-gray-600 bg-gray-100 px-3 py-1 rounded-full">
            <Clock className="w-4 h-4" />
            <span className="text-sm">{recipe.total_time}</span>
          </div>
        )}
        {(recipe as any).difficulty && (
          <div className="flex items-center gap-2 text-gray-600 bg-gray-100 px-3 py-1 rounded-full">
            <span className="text-sm">난이도: {(recipe as any).difficulty}</span>
          </div>
        )}
      </div>

      {/* 재료 */}
      <section className="mb-8">
        <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <span className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
            <span className="text-primary-600">🥬</span>
          </span>
          재료
        </h2>
        <div className="bg-orange-50 rounded-xl p-4 border-l-4 border-primary-500">
          <ul className="grid grid-cols-2 gap-2">
            {recipe.ingredients.map((ing, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-primary-500 mt-1">•</span>
                <span>
                  <span className="font-medium">{ing.name}</span>
                  {(ing.amount || ing.unit) && (
                    <span className="text-gray-600 ml-1">
                      {ing.amount}{ing.unit}
                    </span>
                  )}
                  {ing.note && (
                    <span className="text-gray-500 text-sm ml-1">({ing.note})</span>
                  )}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* 조리 순서 */}
      <section className="mb-8">
        <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <span className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
            <span className="text-primary-600">👨‍🍳</span>
          </span>
          조리 순서
        </h2>
        <div className="space-y-6">
          {recipe.steps.map((step) => (
              <div
                key={step.step_number}
                className="bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm hover:shadow-md transition-shadow"
              >
                <div className="p-4">
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0 w-10 h-10 bg-primary-500 rounded-full flex items-center justify-center text-white font-bold">
                      {step.step_number}
                    </div>
                    <div className="flex-grow">
                      <p className="text-gray-800 font-medium text-lg">{step.instruction}</p>
                      <div className="flex gap-3 mt-2">
                        <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                          ⏱ {formatTime(step.timestamp)}
                        </span>
                        {step.duration && (
                          <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                            소요: {step.duration}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* 상세 설명 */}
                  {(step as any).details && (
                    <div className="mt-3 ml-14 text-gray-600 text-sm leading-relaxed">
                      {(step as any).details}
                    </div>
                  )}

                  {/* 팁 */}
                  {step.tips && (
                    <div className="mt-4 bg-green-50 rounded-lg p-3 flex items-start gap-2">
                      <Lightbulb className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                      <p className="text-green-800 text-sm">{step.tips}</p>
                    </div>
                  )}
                </div>
              </div>
          ))}
        </div>
      </section>

      {/* 전체 팁 */}
      {recipe.tips && recipe.tips.length > 0 && (
        <section className="mb-8">
          <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <span className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
              <span className="text-primary-600">💡</span>
            </span>
            요리 팁
          </h2>
          <div className="bg-yellow-50 rounded-xl p-4 border border-yellow-200">
            <ul className="space-y-2">
              {recipe.tips.map((tip, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="text-yellow-600">•</span>
                  <span className="text-gray-700">{tip}</span>
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}

      {/* 처리 시간 정보 */}
      {(result as any).timing && (
        <section className="mb-8">
          <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <span className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
              <span className="text-primary-600">⚡</span>
            </span>
            분석 소요 시간
          </h2>
          <div className="bg-blue-50 rounded-xl p-4 border border-blue-200">
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {(result as any).timing.download}s
                </div>
                <div className="text-xs text-gray-500">다운로드</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {(result as any).timing.transcript}s
                </div>
                <div className="text-xs text-gray-500">텍스트 추출</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {(result as any).timing.parsing}s
                </div>
                <div className="text-xs text-gray-500">레시피 분석</div>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-blue-200 text-center">
              <span className="text-gray-600">총 소요 시간: </span>
              <span className="text-xl font-bold text-blue-700">
                {(result as any).timing.total}초
              </span>
            </div>
          </div>
        </section>
      )}
    </div>
  );
};

export default RecipeResult;
