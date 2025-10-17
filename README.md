# 🤖 LEGAL VIRTUAL ASSISTANT

A state-of-the-art multilingual AI chatbot designed for campus assistance, supporting **5+ languages** with premium UI/UX and advanced AI capabilities.

## ✨ Features

### 🌍 Multilingual Support
- **English, Hindi, Bengali, Tamil, Telugu** support
- Real-time language detection and translation
- Context-aware multilingual responses
- Seamless language switching

### 🎨 Premium UI/UX
- **Stunning glassmorphism design** with smooth animations
- **Responsive design** for all devices
- **Dark/Light mode** support
- **Voice input/output** capabilities
- **Real-time typing indicators**

### 🚀 Advanced AI Features
- **GPT-4 & Claude integration** for intelligent responses
- **Vector database** for semantic search
- **Context management** across conversations
- **Intent classification** and entity extraction
- **Fallback to human support** when needed

### 📊 Analytics & Monitoring
- **Real-time conversation logging**
- **Performance analytics dashboard**
- **User feedback collection**
- **A/B testing capabilities**

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Next.js 14    │    │   FastAPI       │    │   PostgreSQL    │
│   Frontend      │◄──►│   Backend       │◄──►│   Database      │
│   (Port 3000)   │    │   (Port 8000)   │    │   (Port 5432)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        │              ┌─────────────────┐    ┌─────────────────┐
        │              │     Redis       │    │    MongoDB      │
        └──────────────│   Cache/Session │    │   Analytics     │
                       │   (Port 6379)   │    │   (Port 27017)  │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

### 1. Clone & Setup
```bash
git clone <repository-url>
cd campus-chatbot
cp backend/env.example backend/.env
```

### 2. Configure Environment Variables
Edit `backend/.env` with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PINECONE_API_KEY=your_pinecone_api_key
# ... other configuration
```

### 3. Start with Docker (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 4. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🛠️ Local Development

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Database Setup
```bash
# Using Docker
docker-compose up postgres redis mongodb -d

# Or install locally and run the init script
psql -U postgres -d campus_chatbot -f database/init.sql
```

## 📁 Project Structure

```
campus-chatbot/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── routers/        # API routes
│   │   ├── services/       # Business logic
│   │   ├── models/         # Data models
│   │   ├── utils/          # Utilities
│   │   └── middleware/     # Custom middleware
│   ├── main.py            # Application entry point
│   ├── requirements.txt   # Python dependencies
│   └── Dockerfile        # Backend container
├── frontend/              # Next.js frontend
│   ├── src/
│   │   ├── app/          # App router pages
│   │   ├── components/   # React components
│   │   ├── lib/          # Utilities & stores
│   │   └── hooks/        # Custom hooks
│   ├── package.json     # Node dependencies
│   └── Dockerfile      # Frontend container
├── database/            # Database schemas
│   └── init.sql        # Database initialization
├── nginx/              # Reverse proxy config
├── docker-compose.yml  # Multi-container setup
└── README.md          # This file
```

## 🎯 API Endpoints

### Chat API
- `POST /api/chat/message` - Send message to chatbot
- `GET /api/chat/conversation/{user_id}` - Get conversation history
- `DELETE /api/chat/conversation/{user_id}` - Clear conversation
- `POST /api/chat/voice` - Process voice message
- `POST /api/chat/translate` - Translate text

### Admin API
- `GET /api/admin/analytics` - Get usage analytics
- `POST /api/admin/knowledge` - Add knowledge base entry
- `GET /api/admin/feedback` - Get user feedback

### Files API
- `POST /api/files/upload` - Upload documents
- `GET /api/files/process/{file_id}` - Process uploaded file

## 🌐 Supported Languages

| Language | Code | Native Name | Status |
|----------|------|-------------|---------|
| English  | `en` | English     | ✅ Full Support |
| Hindi    | `hi` | हिन्दी       | ✅ Full Support |
| Bengali  | `bn` | বাংলা       | ✅ Full Support |
| Tamil    | `ta` | தமிழ்       | ✅ Full Support |
| Telugu   | `te` | తెలుగు      | ✅ Full Support |

## 🔧 Configuration

### Environment Variables

#### Backend (.env)
```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/campus_chatbot
REDIS_URL=redis://localhost:6379
MONGODB_URL=mongodb://localhost:27017/campus_chatbot

# AI Services
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
PINECONE_API_KEY=your_pinecone_key

# External APIs
TWILIO_ACCOUNT_SID=your_twilio_sid
TELEGRAM_BOT_TOKEN=your_telegram_token

# Security
SECRET_KEY=your_secret_key
```

#### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## 🚀 Deployment

### Production Deployment

1. **Set up environment variables** in your production environment
2. **Configure database** connections and API keys
3. **Deploy using Docker Compose**:

```bash
# Production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Cloud Deployment Options

#### Vercel (Frontend)
```bash
cd frontend
vercel deploy --prod
```

#### Railway/Render (Backend)
- Connect your GitHub repository
- Set environment variables
- Deploy automatically

#### AWS/GCP/Azure
- Use provided Dockerfiles
- Set up managed databases
- Configure load balancer

## 📊 Monitoring & Analytics

### Built-in Analytics
- **Conversation volume** tracking
- **Language usage** statistics
- **Response accuracy** metrics
- **User satisfaction** scores

### Health Monitoring
- `/health` endpoint for service health
- Database connection monitoring
- AI service availability checks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- 📧 Email: support@campus-chatbot.com
- 📚 Documentation: [docs.campus-chatbot.com](https://docs.campus-chatbot.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)

## 🎉 Acknowledgments

- **OpenAI** for GPT models
- **Anthropic** for Claude AI
- **Vercel** for deployment platform
- **Next.js** team for the amazing framework
- **FastAPI** for the high-performance backend framework

---

Made with ❤️ for campus communities worldwide 🌍
