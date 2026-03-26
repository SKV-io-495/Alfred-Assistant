# Alfred-AI - RAG-Powered Chat Assistant

[![Alfred Assistant Demo](https://github.com/user-attachments/assets/b8eec45e-d819-4579-9e6e-0e0db2e5e0e7)](https://drive.google.com/file/d/180I7kAu0v2blqGuMLc5yticLmQIkEvzf/view?usp=sharing)

**🔴 LIVE DEMO:** [https://alfred-assistant.vercel.app/](https://alfred-assistant.vercel.app/)  
*(Note: Backend is on a free instance. Please allow ~50s for the first cold start.)*


A modern, text-only RAG (Retrieval-Augmented Generation) powered chat assistant that provides real-time information from Wikipedia, news sources, and Reddit discussions. Built with LangChain, GROQ LLM, and comprehensive professional features including memory management, cost monitoring, and automated evaluation.

## 🚀 Features

- **Real-time Information Retrieval**: Fetches fresh content from Wikipedia, news APIs, and Reddit
- **Advanced Memory Management**: Session buffer + conversation summarization
- **Professional Monitoring**: LangSmith integration for cost tracking and performance monitoring
- **Automated Evaluation**: ROUGE-based evaluation pipeline with human-in-the-loop capabilities
- **Modern Tech Stack**: FastAPI backend, Next.js frontend, Docker deployment ready
- **Extensible Architecture**: Easy to add new data sources and customize

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │  Data Sources   │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│  Wikipedia      │
│                 │    │                 │    │  News API       │
│   - Chat UI     │    │   - RAG Chain   │    │  Reddit         │
│   - TailwindCSS │    │   - Vector DB   │    └─────────────────┘
│   - TypeScript  │    │   - Memory Mgmt │
└─────────────────┘    │   - LangSmith   │    ┌─────────────────┐
                       └─────────────────┘    │   Monitoring    │
                                              │   LangSmith     │
                                              │   Evaluation    │
                                              └─────────────────┘
```

## ☁️ Cloud Infrastructure & DevOps

While the live demo runs on a lightweight free-tier architecture (Render + Vercel), this project was originally engineered for **production-grade serverless scaling** on Google Cloud Platform.

The repository includes a complete **CI/CD pipeline** and **Infrastructure as Code** setup:

* **Google Cloud Run**: Containerized backend deployment for auto-scaling.
* **Cloud Build**: Automated CI/CD pipeline defined in [`cloudbuild.yaml`](./cloudbuild.yaml).
* **Docker**: Fully containerized microservices architecture.
* **Secret Management**: Production secrets handling via GCP Secret Manager.

*You can view the production deployment configuration in the `cloudbuild.yaml` file.*

## 🛠️ Tech Stack

### Backend
- **Python 3.11+** with FastAPI
- **LangChain** for RAG implementation
- **GROQ** for LLM inference (free tier)
- **Chroma** for vector storage
- **FastEmbed** for lightweight local ONNX embeddings (bge-small-en-v1.5) bypassing external API limits
- **LangSmith** for monitoring and tracing

### Frontend
- **Next.js 14** with TypeScript
- **React 18** for UI components
- **TailwindCSS** for styling
- **Axios** for API communication

### Data Sources
- **Wikipedia** via wikipedia-python
- **News API** for current events
- **Reddit** via PRAW

## 🚀 Quick Start

### Prerequisites

1. **GROQ API Key** (required) - Get from [console.groq.com](https://console.groq.com)
2. **News API Key** (optional) - Get from [newsapi.org](https://newsapi.org)
3. **Reddit OAuth** (optional) - Get from [reddit.com/prefs/apps](https://reddit.com/prefs/apps)
4. **LangSmith API Key** (optional) - Get from [smith.langchain.com](https://smith.langchain.com)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd yeest
   ```

2. **Backend Setup**
   ```bash
   cd backend
   cp .env.example .env
   # Edit .env with your API keys
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

### Running the Application

#### Option 1: Docker Compose (Recommended)
```bash
# Copy environment variables
cp backend/.env.example backend/.env
# Edit backend/.env with your API keys

# Start both services
docker-compose up --build
```

#### Option 2: Manual Setup
```bash
# Terminal 1 - Backend
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📝 Configuration

### Environment Variables

Create `backend/.env` with the following variables:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here

# Optional - for enhanced functionality
NEWSAPI_KEY=your_newsapi_key_here
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here

# Configuration (optional)
GROQ_MODEL_NAME=mixtral-8x7b-32768
VECTOR_STORE_PATH=./chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=50
MEMORY_BUFFER_SIZE=8
RAG_K=5
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v
```

### Frontend Tests
```bash
cd frontend
npm run lint
npm run type-check
npm run build
```

### Evaluation Pipeline
```bash
cd backend
python -m app.eval_runner
```

## 📊 Monitoring & Evaluation

### LangSmith Integration
- Automatic tracing of all LLM calls
- Cost monitoring and token usage tracking
- Performance metrics and latency analysis
- Access dashboard at [smith.langchain.com](https://smith.langchain.com)

### Automated Evaluation
- ROUGE-based scoring against reference answers
- Continuous evaluation in CI/CD pipeline
- Human-in-the-loop annotation support
- Results logged to LangSmith for analysis

## 🚀 Deployment

### Live Architecture
- **Frontend**: Hosted on Vercel (Next.js). Requires the `BACKEND_URL` environment variable properly configured in the dashboard.
- **Backend**: Hosted on Render (Docker/FastAPI). Our optimized Dockerfile securely caches the FastEmbed ONNX model natively at build-time and single-threads the runtime to completely eliminate Render's 512MB RAM OOM limits and Vercel's 10-second Serverless Gateway timeouts.
- **Note**: The Render free tier spins down after 15 minutes of inactivity, causing a "cold start" delay on the initial request.

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build -d

# Or build individual containers
docker build -t yeest-backend ./backend
docker build -t yeest-frontend ./frontend
```

### Cloud Deployment Options

#### Backend
- **AWS ECS** / **Google Cloud Run** / **Fly.io**
- **Heroku** / **Railway** / **Render**

#### Frontend
- **Vercel** (recommended for Next.js)
- **Netlify** / **AWS Amplify**

### Environment Setup for Production
1. Set up secret management (AWS Secrets Manager, etc.)
2. Configure CORS origins for production domains
3. Set up monitoring and logging
4. Configure auto-scaling based on usage

## 🔧 Development

### Project Structure
```
alfred-app/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Configuration management
│   │   ├── llm.py               # LLM factory
│   │   ├── rag.py               # RAG implementation
│   │   ├── memory.py            # Memory management
│   │   ├── utils.py             # Utility functions
│   │   ├── eval_runner.py       # Evaluation pipeline
│   │   └── retrievers/          # Data source retrievers
│   ├── tests/                   # Backend tests
│   ├── requirements.txt         # Python dependencies
│   └── Dockerfile              # Backend container
├── frontend/
│   ├── pages/                   # Next.js pages
│   ├── components/              # React components
│   ├── styles/                  # CSS styles
│   ├── package.json            # Node.js dependencies
│   └── Dockerfile              # Frontend container
├── .github/workflows/           # CI/CD pipelines
└── docker-compose.yml          # Development setup
```

### Adding New Data Sources
1. Create a new retriever in `backend/app/retrievers/`
2. Implement the retrieval function returning LangChain Documents
3. Add the retriever to `backend/app/rag.py`
4. Update the frontend to display source metadata

### Customizing the LLM
1. Modify `backend/app/llm.py` to use different models
2. Update prompts in `backend/app/rag.py`
3. Adjust memory settings in `backend/app/config.py`

## 📈 Performance Optimization

### Backend
- **Caching**: Implement Redis for popular queries
- **Vector Store**: Consider Pinecone for production scale
- **Async Processing**: Use background tasks for heavy operations
- **Rate Limiting**: Implement API rate limiting

### Frontend
- **Code Splitting**: Automatic with Next.js
- **Image Optimization**: Use Next.js Image component
- **Caching**: Implement SWR or React Query
- **Performance Monitoring**: Add analytics and monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the RAG framework
- **GROQ** for fast LLM inference
- **FastEmbed** for local ONNX embeddings
- **Anthropic** for Claude model inspiration
- **Vercel** for Next.js framework
- **FastAPI** for the backend framework

## 📞 Support

For support, please open an issue on GitHub .

---

**Built with ❤️ for the AI community**
