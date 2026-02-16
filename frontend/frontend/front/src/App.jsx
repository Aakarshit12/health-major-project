import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// --- SUB-COMPONENTS ---

const Sidebar = ({ activeTab, setActiveTab, toggleChat }) => (
  <aside className="sidebar">
    <div className="sidebar-logo">
      <div className="logo-icon">●</div>
    </div>
    
    <div className="sidebar-menu">
      <button 
        className={`menu-item ${activeTab === 'scan' ? 'active' : ''}`}
        onClick={() => setActiveTab('scan')}
        title="New Scan"
      >
        <span className="icon">✛</span>
      </button>
      <button 
        className={`menu-item ${activeTab === 'history' ? 'active' : ''}`} 
        onClick={() => setActiveTab('history')}
        title="History"
      >
        <span className="icon">⌚</span>
      </button>
      <button 
        className={`menu-item ${activeTab === 'patients' ? 'active' : ''}`}
        onClick={() => setActiveTab('patients')}
        title="Patients"
      >
        <span className="icon">☺</span>
      </button>
    </div>

    <div className="sidebar-footer">
      {/* UPDATED: AI Assistant Button with specific class */}
      <button className="menu-item ai-launch-btn" title="AI Assistant" onClick={toggleChat}>
        <span className="icon">✨</span>
      </button>
      
      <button className="menu-item" title="Settings">
        <span className="icon">⚙</span>
      </button>
      <div className="user-avatar">DR</div>
    </div>
  </aside>
);

const ChatPanel = ({ analysisData, onClose }) => {
  const [messages, setMessages] = useState([
    { sender: 'ai', text: 'Hello. I have analyzed the current scan. You can ask me about specific findings, probabilities, or generate a summary.' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userMsg.text,
          context: analysisData || {} 
        }),
      });

      if (!response.ok) throw new Error("Failed to connect");
      
      const data = await response.json();
      setMessages(prev => [...prev, { sender: 'ai', text: data.answer }]);
    } catch (error) {
      setMessages(prev => [...prev, { sender: 'ai', text: "I'm having trouble connecting to the server. Please ensure the backend is running." }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div className="chat-title">
          <span>✨</span> AI Assistant
        </div>
        <button onClick={onClose} className="chat-close">✖</button>
      </div>

      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message-bubble ${msg.sender}`}>
            {msg.text}
          </div>
        ))}
        {isTyping && (
          <div className="message-bubble ai typing">
            <span>●</span><span>●</span><span>●</span>
          </div>
        )}
        <div ref={scrollRef} />
      </div>

      <div className="chat-input-area">
        <input 
          type="text" 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask a question..."
          autoFocus
        />
        <button onClick={handleSend} disabled={isTyping || !input.trim()}>➤</button>
      </div>
    </div>
  );
};

// --- VISUAL COMPONENTS ---

const PatientHeader = () => (
  <div className="patient-header">
    <div className="patient-info-group">
      <div className="patient-avatar-small">JD</div>
      <div>
        
      </div>
    </div>
    <div className="scan-date">
      <span>Scan Date:</span>
      <strong>Nov 28, 2025 </strong>
    </div>
  </div>
);

const ScanMetadata = () => (
  <div className="metadata-card">
    <h4 className="card-title-small">DICOM Metadata</h4>
    <div className="meta-grid">
      <div className="meta-item">
        <label>Modality</label>
        <span>CR (Computed Radiography)</span>
      </div>
      <div className="meta-item">
        <label>Body Part</label>
        <span>CHEST</span>
      </div>
      <div className="meta-item">
        <label>View Position</label>
        <span>PA</span>
      </div>
      <div className="meta-item">
        <label>Tube Voltage</label>
        <span>120 kVp</span>
      </div>
      <div className="meta-item">
        <label>Exposure</label>
        <span>4 mAs</span>
      </div>
      <div className="meta-item">
        <label>Manufacturer</label>
        <span>Siemens Healthineers</span>
      </div>
    </div>
  </div>
);

const AiReport = ({ summary }) => (
  <div className="ai-report-card">
    <h4 className="card-title-small">AI Preliminary Report</h4>
    <p className="report-text">
      Analysis detected a primary finding of <strong>{summary.top_finding}</strong> with a probability of <strong>{summary.percentage.toFixed(1)}%</strong>. 
      The scan is classified as <strong>{summary.risk_level}</strong> based on opacity patterns.
      Please correlate with clinical history.
    </p>
    <div className="report-footer">
      <span className="status-badge">Review Required</span>
      <span className="model-version">Model: PulmoNet v2.4</span>
    </div>
  </div>
);

const ProbabilityChart = ({ findings }) => {
  const topRisks = [...findings].sort((a, b) => b.percentage - a.percentage).slice(0, 4);
  return (
    <div className="prob-chart-card">
      <h4 className="card-title-small">Risk Distribution</h4>
      <div className="chart-bars">
        {topRisks.map((finding) => (
          <div key={finding.disease} className="chart-row">
            <div className="chart-label">
              <span>{finding.disease_name}</span>
              <span className="chart-val">{finding.percentage.toFixed(1)}%</span>
            </div>
            <div className="bar-bg">
              <div 
                className={`bar-fill ${finding.risk_level === 'High Risk' ? 'fill-high' : 'fill-mod'}`}
                style={{ width: `${finding.percentage}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const ScanMetrics = ({ summary }) => {
  return (
    <div className="metrics-grid">
      <div className="metric-item">
        <span className="metric-label">Confidence</span>
        <span className="metric-value">98.2<small>%</small></span>
      </div>
      <div className="metric-item">
        <span className="metric-label">Severity</span>
        <span className="metric-value high">{summary.risk_level === 'High Risk' ? '8.5' : '3.2'}</span>
      </div>
      <div className="metric-item">
        <span className="metric-label">Quality</span>
        <span className="metric-value ok">Optimal</span>
      </div>
      <div className="metric-item">
        <span className="metric-label">Time</span>
        <span className="metric-value">1.4s</span>
      </div>
    </div>
  );
};

const TopFindingCard = ({ summary }) => {
  const [count, setCount] = useState(0);
  
  useEffect(() => {
    let start = 0;
    const end = summary.percentage;
    const duration = 2000;
    const increment = end / (duration / 16);
    
    const timer = setInterval(() => {
      start += increment;
      if (start >= end) {
        setCount(end);
        clearInterval(timer);
      } else {
        setCount(start);
      }
    }, 16);
    
    return () => clearInterval(timer);
  }, [summary.percentage]);

  return (
    <div className="top-finding-card">
      <h4 className="card-title-small">Primary Prediction</h4>
      <div className="top-finding-content">
        <h2 className="top-finding-title">{summary.top_finding}</h2>
        <div className="top-finding-stats">
          <span className="top-finding-percentage">{count.toFixed(1)}%</span>
          <span className="top-finding-risk">{summary.risk_level}</span>
        </div>
      </div>
    </div>
  );
};

const RiskOrb = ({ finding, index, onClick, isSelected, isHovered: isHoveredProp, onHover }) => {
  const [isVisible, setIsVisible] = useState(false);
  
  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), index * 100);
    return () => clearTimeout(timer);
  }, [index]);

  const getRiskColorClass = (risk) => {
    if (risk === "High Risk") return "orb-high-risk";
    if (risk === "Moderate Risk") return "orb-moderate-risk";
    if (risk === "Low Risk") return "orb-low-risk";
    return "orb-very-low-risk";
  };

  const size = 60 + finding.probability * 80; // Slightly smaller to fit compact view
  const angle = (index / 18) * Math.PI * 2;
  const radius = 220; // Compact radius
  const x = Math.cos(angle) * radius;
  const y = Math.sin(angle) * radius;

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => onHover(finding.disease)}
      onMouseLeave={() => onHover(null)}
      className={`risk-orb ${isVisible ? 'risk-orb-visible' : ''} ${isSelected ? 'risk-orb-selected' : ''} ${isHoveredProp ? 'risk-orb-hovered' : ''}`}
      style={{
        left: `calc(50% + ${x}px)`,
        top: `calc(50% + ${y}px)`,
      }}
    >
      <div
        className={`risk-orb-inner ${getRiskColorClass(finding.risk_level)} ${finding.risk_level === "High Risk" ? 'risk-orb-pulse' : ''}`}
        style={{ width: `${size}px`, height: `${size}px` }}
      >
        <div className="risk-orb-glow"></div>
        <div className="risk-orb-content">
          <div className="risk-orb-percentage">{finding.percentage.toFixed(0)}%</div>
          <div className="risk-orb-label">{finding.disease}</div>
        </div>
        {isHoveredProp && (
          <div className="risk-orb-tooltip">
            <div className="tooltip-title">{finding.disease_name}</div>
            <div className="tooltip-risk">{finding.risk_level}</div>
            <div className="tooltip-desc">{finding.description}</div>
          </div>
        )}
      </div>
    </div>
  );
};

const DetailsPanel = ({ finding, onClose }) => {
  if (!finding) return null;
  return (
    <>
      <div className="overlay" onClick={onClose}></div>
      <div className="details-panel">
        <div className="details-panel-header">
          <div>
            <h3 className="details-panel-title">{finding.disease_name}</h3>
            <span className="details-panel-subtitle">{finding.risk_level}</span>
          </div>
          <button onClick={onClose} className="details-panel-close">✖</button>
        </div>
        <div className="details-panel-content">
          <div className="details-panel-stat">
            <div className="details-panel-stat-value">{finding.percentage.toFixed(1)}%</div>
            <div className="details-panel-stat-label">Detection Probability</div>
          </div>
          <p className="details-panel-text">{finding.description}</p>
          <p className="details-panel-action-text">{finding.action}</p>
        </div>
      </div>
    </>
  );
};

// --- MAIN APP COMPONENT ---

export default function LungScanDashboard() {
  const [activeTab, setActiveTab] = useState('scan');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedFinding, setSelectedFinding] = useState(null);
  const [hoveredOrb, setHoveredOrb] = useState(null);
  const [filter, setFilter] = useState('All');
  const [searchTerm, setSearchTerm] = useState('');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isChatOpen, setIsChatOpen] = useState(false); 
  const fileInputRef = useRef(null);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedImage(URL.createObjectURL(file));
      processImage(file);
    }
  };

  const processImage = async (file) => {
    setLoading(true);
    setTimeout(async () => {
      try {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch("http://127.0.0.1:8000/analyze", {
            method: "POST",
            body: formData,
        });
        if (!response.ok) throw new Error("API request failed");
        const result = await response.json();
        setData(result);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }, 1500);
  };

  const filteredFindings = data?.findings.filter(f => {
    const matchesFilter = filter === 'All' || f.risk_level === filter;
    const matchesSearch = f.disease_name.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesFilter && matchesSearch;
  }) || [];

  return (
    <div className="desktop-app">
      <div className="bg-graphics">
        <div className="bg-orb orb-1"></div>
        <div className="bg-orb orb-2"></div>
        <div className="bg-grid"></div>
      </div>

      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        toggleChat={() => setIsChatOpen(!isChatOpen)} 
      />
      
      <main className="main-stage">
        <header className="stage-header">
          <div className="breadcrumbs">Workstation / <strong>Lung Analysis</strong></div>
          <div className="header-status">System Operational ●</div>
        </header>

        <div className="stage-content">
          {/* LEFT PANEL: INPUT */}
          <div className={`panel input-panel ${data ? 'panel-compact' : 'panel-full'}`}>
            <div className="panel-header">
              <h3>Scan Input</h3>
            </div>
            
            <div className="upload-zone">
              {!uploadedImage ? (
                <div className="upload-empty-state" onClick={() => fileInputRef.current?.click()}>
                   <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="file-input"
                  />
                  <div className="upload-icon-large">+</div>
                  <p>Drop DICOM / JPEG here</p>
                  <span>or click to browse</span>
                </div>
              ) : (
                <div className="upload-preview-container">
                  <img src={uploadedImage} alt="Scan" className="scan-image" />
                  <div className="scan-meta">
                    <span className="scan-id">ID: 883-XC-29</span>
                    <button className="btn-reset" onClick={() => {
                      setUploadedImage(null);
                      setData(null);
                    }}>New Scan</button>
                  </div>
                  {/* Show Metrics in Left Panel */}
                  {data && <ScanMetrics summary={data.summary} />}
                  {loading && <div className="scan-loader-bar"><div className="scan-loader-progress"></div></div>}
                </div>
              )}
            </div>
            
            {!data && (
              <div className="intro-text">
                <h1>PulmoScan AI</h1>
                <p>Advanced diagnostic support system.</p>
              </div>
            )}
          </div>

          {/* RIGHT PANEL: VISUALIZATION */}
          {data && (
            <div className="panel results-panel">
               <PatientHeader />

               <div className="panel-header with-controls sub-header">
                <h3>Analysis Model</h3>
                <div className="panel-controls">
                  <select value={filter} onChange={(e) => setFilter(e.target.value)}>
                    <option value="All">All Risks</option>
                    <option value="High Risk">High Risk</option>
                    <option value="Moderate Risk">Moderate Risk</option>
                    <option value="Low Risk">Low Risk</option>
                  </select>
                </div>
              </div>

              <div className="visualization-area">
                {/* Bento Grid Layout */}
                <div className="dashboard-grid">
                    <TopFindingCard summary={data.summary} />
                    <ProbabilityChart findings={data.findings} />
                    <ScanMetadata />
                    <AiReport summary={data.summary} />
                </div>
                
                {/* Orbital Map */}
                <div className="orbital-section">
                    <h4 className="section-label">Full Neural Map</h4>
                    <div className="orbital-wrapper-compact">
                      {filteredFindings.map((finding, index) => (
                        <RiskOrb
                          key={finding.disease}
                          finding={finding}
                          index={index}
                          onClick={() => setSelectedFinding(finding)}
                          isSelected={selectedFinding?.disease === finding.disease}
                          isHovered={hoveredOrb === finding.disease}
                          onHover={setHoveredOrb}
                        />
                      ))}
                    </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {selectedFinding && (
        <DetailsPanel finding={selectedFinding} onClose={() => setSelectedFinding(null)} />
      )}

      {isChatOpen && (
        <ChatPanel analysisData={data} onClose={() => setIsChatOpen(false)} />
      )}
    </div>
  );
}