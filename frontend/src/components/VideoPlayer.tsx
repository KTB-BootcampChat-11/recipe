import React from 'react';

interface VideoPlayerProps {
  videoId: string | null;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ videoId }) => {
  if (!videoId) {
    return (
      <div className="w-full aspect-[9/16] bg-gray-100 rounded-xl flex items-center justify-center">
        <div className="text-center text-gray-400">
          <svg
            className="w-16 h-16 mx-auto mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <p>YouTube URL을 입력하세요</p>
        </div>
      </div>
    );
  }

  // YouTube Shorts 임베드 URL
  const embedUrl = `https://www.youtube.com/embed/${videoId}`;

  return (
    <div className="w-full aspect-[9/16] bg-black rounded-xl overflow-hidden shadow-lg">
      <iframe
        src={embedUrl}
        title="YouTube Shorts"
        className="w-full h-full"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowFullScreen
      />
    </div>
  );
};

export default VideoPlayer;
