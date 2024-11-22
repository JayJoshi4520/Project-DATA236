import React, { useContext, useEffect, useState } from 'react';
import { Search, TrendingUp, BookOpen, DollarSign, LineChart, Bitcoin } from 'lucide-react';
import BlogPost from '../components/BlogPost';
import { fetchNews } from '../utils/api/stock-api';
import e from 'cors';
import ThemeContext from '../context/ThemeContext';

const categories = [
  { name: 'General', icon: LineChart },
  { name: 'Forex', icon: TrendingUp },
  { name: 'Crypto', icon: Bitcoin },
];



export default function Blog() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('General');
  const [blogPosts, setNewsData] = useState([])
  const { darkMode } = useContext(ThemeContext);


  useEffect(async () => {


    try {
      fetchNews(selectedCategory.toLowerCase()).then(res => {
        setNewsData(formatNews(res))
        
      }).catch(e => {
        console.log(e);

      })


    } catch (e) {
      console.log(e);
    }


  }, [selectedCategory])



  const formatNews = (newsData) => {
    return newsData.map((item) => {
      const date = new Date(item.datetime * 1000);
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const day = String(date.getDate()).padStart(2, '0');
      const hours = String(date.getHours()).padStart(2, '0');
      const minutes = String(date.getMinutes()).padStart(2, '0');

      const formattedDate = `${year}-${month}-${day} ${hours}:${minutes}`;

      return {
        id: item.id,
        title: item.headline,
        excerpt: item.summary,
        date: formattedDate, // Set the formatted date as a string
        readTime: '5 min read',
        category: item.category,
        imageUrl: item.image,
        articleLink: item.url,
        author: {
          name: item.source,
          avatar:
            'https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&q=80&w=150',
        },
      };
    });
  };









  const filteredPosts = blogPosts.filter(post => {
    const matchesSearch = post.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      post.excerpt.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = !selectedCategory || post.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className={`min-h-screen ${darkMode ? "bg-gray-900 text-gray-300" : "bg-neutral-100"}`}>
      <div className={`max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12`}>
        <div className="text-center mb-12">
          <h1 className={`text-4xl font-bold mb-4 ${darkMode ? "text-white" : "text-gray-900"}`}>
            Market Insights & Analysis
          </h1>
          <p className="text-xl text-gray-500">
            Stay informed with the latest market trends, analysis, and trading strategies
          </p>
        </div>

        {/* <div className="mb-8">
          <div className="relative max-w-xl mx-auto">
            <input
              type="text"
              placeholder="Search articles..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-4 py-3 pl-12 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <Search className="absolute left-4 top-3.5 text-gray-400" size={20} />
          </div>
        </div> */}

        <div className="mb-12">
          <div className="flex flex-wrap justify-center gap-4">
            {categories.map(({ name, icon: Icon }) => (
              <button
                key={name}
                onClick={() => setSelectedCategory(name)}
                className={`px-6 py-2 rounded-full flex items-center gap-2 ${selectedCategory === name
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
                  } transition-colors duration-200`}
              >
                <Icon size={18} />
                {name}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {blogPosts.map((post, index) => (
            <BlogPost key={index} {...post} />
          ))}
        </div>
      </div>
    </div>
  );
}