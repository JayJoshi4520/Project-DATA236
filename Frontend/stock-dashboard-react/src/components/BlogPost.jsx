import React from 'react';
import { Calendar, Tag } from 'lucide-react';
import TempImage from'../assets/news.jpg';
export default function BlogPost({ title, excerpt, date, readTime, category, imageUrl, author, articleLink }) {
  

  return (
    <article className="bg-white rounded-xl cursor-pointer shadow-md overflow-hidden transition-transform hover:scale-[1.02]" onClick={() => window.open(articleLink)}>
      <div className="relative h-48 overflow-hidden">
      <img
          src={imageUrl || TempImage}
          alt={title || 'Default Image'}
          className="w-full h-full object-cover"
          onError={(e) => {
            e.target.onerror = null; // Prevent infinite loop
            e.target.src = TempImage; // Set fallback image
          }}
        />
        <div className="absolute top-4 right-4">
          <span className="px-3 py-1 bg-blue-600 text-white text-sm rounded-full flex items-center gap-1">
            <Tag size={14} />
            {category}
          </span>
        </div>
      </div>
      <div className="p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-2 line-clamp-2">
          {title}
        </h2>
        <p className="text-gray-600 mb-4 line-clamp-2">
          {excerpt}
        </p>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <img 
              src={author.avatar} 
              alt={author.name}
              className="w-10 h-10 rounded-full"
            />
            <div>
              <p className="text-sm font-medium text-gray-900">{author.name}</p>
              <div className="flex items-center text-sm text-gray-500">
                <Calendar size={14} className="mr-1" />
                {date}
              </div>
            </div>
          </div>
        </div>
      </div>
    </article>
  );
}