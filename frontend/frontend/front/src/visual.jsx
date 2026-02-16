import React, { useState, useMemo, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, 
  Activity, 
  AlertTriangle, 
  ShieldCheck, 
  ChevronRight, 
  X,
  Stethoscope,
  Microscope,
  Thermometer,
  ScanEye
} from 'lucide-react';

// --- DATA ---
const MEDICAL_DATA = {
  "success": true,
  "timestamp": "2025-11-17T12:40:47.417819",
  "summary": {
    "top_finding": "Hiatal Hernia",
    "probability": 0.9935550689697266,
    "percentage": 99.4,
    "risk_level": "Very High Risk"
  },
  "findings": [
    {
      "disease": "Hernia",
      "disease_name": "Hiatal Hernia",
      "probability": 0.9935550689697266,
      "percentage": 99.4,
      "risk_level": "Very High Risk",
      "emoji": "🔴",
      "description": "Part of stomach pushes through diaphragm",
      "action": "Gastroenterology consultation if symptomatic"
    },
    {
      "disease": "Mass",
      "disease_name": "Lung Mass",
      "probability": 0.8545779371261597,
      "percentage": 85.5,
      "risk_level": "High Risk",
      "emoji": "🟠",
      "description": "Larger growth in lung tissue",
      "action": "Immediate specialist evaluation and possible biopsy"
    },
    {
      "disease": "Atelectasis",
      "disease_name": "Atelectasis",
      "probability": 0.74051274061203,
      "percentage": 74.1,
      "risk_level": "High Risk",
      "emoji": "🟠",
      "description": "Partial or complete collapse of lung tissue",
      "action": "Consult a pulmonologist for breathing exercises and treatment"
    },
    {
      "disease": "Pleural_Thickening",
      "disease_name": "Pleural Thickening",
      "probability": 0.5325583219528198,
      "percentage": 53.3,
      "risk_level": "Moderate Risk",
      "emoji": "🟠",
      "description": "Thickened lining around the lungs",
      "action": "Monitor and follow up with your doctor"
    },
    {
      "disease": "Infiltration",
      "disease_name": "Pulmonary Infiltration",
      "probability": 0.5185866355895996,
      "percentage": 51.9,
      "risk_level": "Moderate Risk",
      "emoji": "🟠",
      "description": "Abnormal substances in lung tissue",
      "action": "Further imaging and clinical correlation recommended"
    },
    {
      "disease": "Fibrosis",
      "disease_name": "Pulmonary Fibrosis",
      "probability": 0.5155581831932068,
      "percentage": 51.6,
      "risk_level": "Moderate Risk",
      "emoji": "🟠",
      "description": "Scarring and thickening of lung tissue",
      "action": "Specialist evaluation and monitoring required"
    },
    {
      "disease": "Fracture",
      "disease_name": "Rib Fracture",
      "probability": 0.513129472732544,
      "percentage": 51.3,
      "risk_level": "Moderate Risk",
      "emoji": "🟠",
      "description": "Broken rib bone",
      "action": "Pain management and rest; heals naturally"
    },
    {
      "disease": "Nodule",
      "disease_name": "Pulmonary Nodule",
      "probability": 0.4094887018203735,
      "percentage": 40.9,
      "risk_level": "Moderate Risk",
      "emoji": "🟠",
      "description": "Small round growth in the lung",
      "action": "Follow-up CT scan needed for characterization"
    },
    {
      "disease": "Emphysema",
      "disease_name": "Emphysema",
      "probability": 0.5004621744155884,
      "percentage": 50.0,
      "risk_level": "Moderate Risk",
      "emoji": "🟠",
      "description": "Damaged air sacs in the lungs",
      "action": "Consult pulmonologist for management plan"
    },
    {
      "disease": "Cardiomegaly",
      "disease_name": "Cardiomegaly",
      "probability": 0.46688202023506165,
      "percentage": 46.7,
      "risk_level": "Moderate Risk",
      "emoji": "🟡",
      "description": "Enlarged heart",
      "action": "Cardiology consultation recommended"
    },
    {
      "disease": "Lung Opacity",
      "disease_name": "Lung Opacity",
      "probability": 0.3616490662097931,
      "percentage": 36.2,
      "risk_level": "Moderate Risk",
      "emoji": "🟡",
      "description": "Area of increased density in lung",
      "action": "Clinical correlation and possible follow-up imaging"
    },
    {
      "disease": "Consolidation",
      "disease_name": "Lung Consolidation",
      "probability": 0.17454077303409576,
      "percentage": 17.5,
      "risk_level": "Low Risk",
      "emoji": "🟢",
      "description": "Air spaces filled with fluid or tissue",
      "action": "Medical evaluation needed - may require antibiotics"
    },
    {
      "disease": "Effusion",
      "disease_name": "Pleural Effusion",
      "probability": 0.12558937072753906,
      "percentage": 12.6,
      "risk_level": "Very Low Risk",
      "emoji": "⚪",
      "description": "Fluid buildup around the lungs",
      "action": "Pulmonologist consultation for possible drainage"
    },
    {
      "disease": "Enlarged Cardiomediastinum",
      "disease_name": "Enlarged Cardiomediastinum",
      "probability": 0.10844509303569794,
      "percentage": 10.8,
      "risk_level": "Very Low Risk",
      "emoji": "⚪",
      "description": "Widened central chest area",
      "action": "Cardiac evaluation recommended"
    },
    {
      "disease": "Pneumothorax",
      "disease_name": "Pneumothorax",
      "probability": 0.10749135911464691,
      "percentage": 10.7,
      "risk_level": "Very Low Risk",
      "emoji": "⚪",
      "description": "Collapsed lung due to air leakage",
      "action": "URGENT: Seek emergency medical care immediately"
    },
    {
      "disease": "Pneumonia",
      "disease_name": "Pneumonia",
      "probability": 0.04354599863290787,
      "percentage": 4.4,
      "risk_level": "Very Low Risk",
      "emoji": "⚪",
      "description": "Infection causing lung inflammation",
      "action": "Immediate medical attention and antibiotics needed"
    },
    {
      "disease": "Lung Lesion",
      "disease_name": "Lung Lesion",
      "probability": 0.0022418485023081303,
      "percentage": 0.2,
      "risk_level": "Very Low Risk",
      "emoji": "⚪",
      "description": "Abnormal area in lung tissue",
      "action": "Further investigation with CT scan recommended"
    },
    {
      "disease": "Edema",
      "disease_name": "Pulmonary Edema",
      "probability": 0.0010575245833024383,
      "percentage": 0.1,
      "risk_level": "Very Low Risk",
      "emoji": "⚪",
      "description": "Excess fluid in the lungs",
      "action": "Medical attention needed within 24 hours"
    }
  ]
};

// --- UTILITIES ---

const getRiskColor = (level) => {
  if (level.includes("Very High")) return "#ef4444"; // Red 500
  if (level.includes("High")) return "#f97316"; // Orange 500
  if (level.includes("Moderate")) return "#f59e0b"; // Amber 500
  return "#10b981"; // Emerald 500 (Safe)
};

const getOrbSize = (probability) => {
  // Map probability 0-1 to size 30-90px
  return Math.max(30, probability * 90);
};

// Phyllotaxis (Sunflower) Layout Algorithm
// Scatters items evenly in a spiral from the center outwards
const calculateScatteredPosition = (index, total) => {
  const scaling = 50; // Spread factor
  const angleIncrement = 137.5 * (Math.PI / 180); // Golden angle in radians
  
  // Angle
  const angle = index * angleIncrement;
  
  // Radius - Square root ensures even area distribution
  // (index + 1) prevents placing the first item exactly at 0,0, giving it a bit of space
  const radius = scaling * Math.sqrt(index + 2); 
  
  const x = radius * Math.cos(angle);
  const y = radius * Math.sin(angle);
  
  return { x, y };
};

// --- COMPONENTS ---

const CountUp = ({ value, suffix = "" }) => {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    let start = 0;
    const end = parseFloat(value);
    if (start === end) return;

    let totalDuration = 1000;
    let incrementTime = (totalDuration / end) * 10;

    let timer = setInterval(() => {
      start += 1;
      if (start > end) {
        start = end;
        clearInterval(timer);
      }
      setDisplayValue(start);
    }, 10);

    return () => clearInterval(timer);
  }, [value]);

  return <span>{displayValue}{suffix}</span>;
};

const TopFindingHero = ({ finding }) => {
  return (
    <motion.div 
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-50 to-white border border-amber-200 p-8 shadow-xl mb-8"
    >
      <div className="absolute top-0 right-0 p-4 opacity-10">
        <Activity size={180} className="text-amber-500" />
      </div>
      
      <div className="relative z-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="animate-pulse w-2 h-2 rounded-full bg-red-500"></span>
            <span className="text-red-500 font-mono text-sm tracking-wider uppercase">Critical Detection</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-2 tracking-tight">
            {finding.disease_name}
          </h1>
          <p className="text-slate-600 max-w-md text-lg">
            AI analysis indicates a very high probability match based on current imaging data.
          </p>
        </div>

        <div className="flex items-end gap-2">
          <span className="text-7xl font-black text-transparent bg-clip-text bg-gradient-to-br from-amber-500 to-orange-600">
            <CountUp value={finding.percentage} suffix="%" />
          </span>
          <span className="text-amber-700/60 font-mono mb-4 text-sm font-bold">CONFIDENCE</span>
        </div>
      </div>
      
      <motion.div 
        className="absolute bottom-0 left-0 h-1 bg-gradient-to-r from-transparent via-amber-500 to-transparent w-full"
        animate={{ x: ['-100%', '100%'] }}
        transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
      />
    </motion.div>
  );
};

const RiskOrb = ({ data, index, total, onClick, isSelected, constraintsRef }) => {
  // Use new scattered position function
  const pos = useMemo(() => calculateScatteredPosition(index, total), [index, total]);
  const color = getRiskColor(data.risk_level);
  const size = getOrbSize(data.probability);
  
  const floatDuration = 3 + Math.random() * 2;
  const floatY = Math.random() * 10 - 5;
  const floatX = Math.random() * 10 - 5;

  return (
    <motion.div
      drag
      dragConstraints={constraintsRef}
      dragElastic={0.1} // Reduced elasticity so it doesn't bounce far out
      dragMomentum={false} // Disable momentum to keep it strictly inside
      initial={{ scale: 0, opacity: 0, x: 0, y: 0 }}
      animate={{ 
        scale: 1, 
        opacity: 1,
        x: pos.x, 
        y: pos.y,
        transition: { type: "spring", stiffness: 50, damping: 20 }
      }}
      whileHover={{ scale: 1.3, zIndex: 100, cursor: 'grab' }}
      whileDrag={{ scale: 1.2, zIndex: 100, cursor: 'grabbing' }}
      style={{
        width: size,
        height: size,
        position: 'absolute',
        left: '50%',
        top: '50%',
        marginLeft: -size / 2,
        marginTop: -size / 2,
        boxShadow: isSelected ? `0 0 30px ${color}, inset 0 0 20px ${color}` : `0 0 10px ${color}40`
      }}
      className="flex items-center justify-center rounded-full group z-20 touch-none"
      onClick={() => onClick(data)}
    >
      <motion.div
        animate={{ x: [0, floatX, 0], y: [0, floatY, 0] }}
        transition={{ duration: floatDuration, repeat: Infinity, ease: "easeInOut" }}
        className="w-full h-full"
      >
        <motion.div 
          className="rounded-full w-full h-full border border-white/50 shadow-sm backdrop-blur-md transition-colors duration-300 relative flex items-center justify-center"
          style={{ 
            background: `radial-gradient(circle at 35% 35%, ${color}, ${color}60)`,
            borderColor: isSelected ? 'white' : `${color}40`
          }}
        >
          <div className="text-white font-bold text-[10px] md:text-xs opacity-95 drop-shadow-sm pointer-events-none select-none">
            {Math.round(data.percentage)}%
          </div>
        </motion.div>
      </motion.div>

      <div className="absolute -bottom-14 left-1/2 -translate-x-1/2 pointer-events-none opacity-0 group-hover:opacity-100 transition-all duration-200 z-50 whitespace-nowrap scale-90 group-hover:scale-100 origin-top">
        <div className="bg-white/95 text-slate-900 px-4 py-2 rounded-lg border border-amber-200 shadow-xl flex flex-col items-center gap-1 backdrop-blur-xl">
          <span className="font-bold text-sm text-slate-900">{data.disease_name}</span>
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider font-bold">
            <span style={{ color: color }}>{data.risk_level}</span>
            <span className="text-slate-400">•</span>
            <span className="text-slate-600">Confidence: {data.percentage}%</span>
          </div>
          <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-white border-t border-l border-amber-200 transform rotate-45"></div>
        </div>
      </div>
    </motion.div>
  );
};

const RadarGrid = () => (
  <div className="absolute inset-0 z-0 pointer-events-none flex items-center justify-center">
    {/* Single guide circle visual */}
    <div className="absolute w-[500px] h-[500px] rounded-full border border-slate-200/50"></div>
    {/* Optional: Inner guide for scattered layout visual structure */}
    <div className="absolute w-[250px] h-[250px] rounded-full border border-slate-100 border-dashed"></div>
  </div>
);

const DetailsPanel = ({ data, onClose }) => {
  if (!data) return null;

  return (
    <motion.div
      initial={{ x: "100%", opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: "100%", opacity: 0 }}
      className="fixed inset-y-0 right-0 w-full md:w-96 bg-white/95 backdrop-blur-xl border-l border-slate-200 p-6 shadow-2xl z-50 overflow-y-auto"
    >
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-sm font-mono text-amber-600 uppercase tracking-widest">Diagnostic Detail</h2>
        <button onClick={onClose} className="p-2 hover:bg-slate-100 rounded-full text-slate-500 transition-colors">
          <X size={20} />
        </button>
      </div>

      <div className="flex items-center gap-4 mb-6">
        <div 
          className="w-16 h-16 rounded-2xl flex items-center justify-center text-3xl shadow-lg border border-slate-200"
          style={{ backgroundColor: `${getRiskColor(data.risk_level)}20` }}
        >
          {data.emoji || <Activity />}
        </div>
        <div>
          <h3 className="text-2xl font-bold text-slate-900">{data.disease_name}</h3>
          <span className="text-sm font-bold uppercase tracking-wide" style={{ color: getRiskColor(data.risk_level) }}>
            {data.risk_level}
          </span>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-slate-100 p-4 rounded-xl border border-slate-200">
          <div className="flex justify-between items-end mb-2">
            <span className="text-slate-600 text-sm">AI Confidence</span>
            <span className="text-2xl font-mono text-slate-900">{data.percentage}%</span>
          </div>
          <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${data.percentage}%` }}
              className="h-full rounded-full"
              style={{ backgroundColor: getRiskColor(data.risk_level) }}
            />
          </div>
        </div>

        <div>
          <h4 className="flex items-center gap-2 text-slate-700 font-semibold mb-2">
            <Microscope size={16} className="text-amber-600" /> Clinical Description
          </h4>
          <p className="text-slate-600 leading-relaxed text-sm border-l-2 border-slate-200 pl-4">
            {data.description}
          </p>
        </div>

        <div>
          <h4 className="flex items-center gap-2 text-slate-700 font-semibold mb-2">
            <Stethoscope size={16} className="text-amber-600" /> Recommended Action
          </h4>
          <div className="bg-amber-50 border border-amber-200 p-4 rounded-xl">
            <p className="text-amber-800 text-sm">
              {data.action}
            </p>
          </div>
        </div>
      </div>
      
      {/* Button removed as requested */}
    </motion.div>
  );
};

export default function App() {
  const [filter, setFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [selectedFinding, setSelectedFinding] = useState(null);
  const constraintsRef = useRef(null);

  // Filter and sort data
  const filteredData = useMemo(() => {
    let result = MEDICAL_DATA.findings;

    if (filter !== "All") {
      if (filter === "High Risk") {
        result = result.filter(f => f.risk_level.includes("High"));
      } else if (filter === "Moderate") {
        result = result.filter(f => f.risk_level.includes("Moderate"));
      } else {
        result = result.filter(f => f.risk_level.includes("Low"));
      }
    }

    if (search) {
      result = result.filter(f => 
        f.disease_name.toLowerCase().includes(search.toLowerCase())
      );
    }
    
    // Sort by probability is typically good, but for scattered spiral 
    // we just want consistent order. The spiral naturally places index 0 at center
    // and higher indices outwards.
    return result.sort((a, b) => b.probability - a.probability);
  }, [filter, search]);

  return (
    <div className="min-h-screen bg-white text-slate-900 font-sans selection:bg-amber-500/20">
      
      {/* Header */}
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-md sticky top-0 z-40 px-6 py-4 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-gradient-to-br from-amber-500 to-orange-600 rounded-lg flex items-center justify-center shadow-md">
            <ScanEye size={20} className="text-white" />
          </div>
          <span className="text-lg font-bold text-slate-900 tracking-tight">Visual<span className="text-amber-600">Cortex</span> Medical</span>
        </div>
        <div className="flex items-center gap-4">
          <div className="hidden md:flex items-center gap-2 text-xs font-mono text-amber-700 bg-amber-50 px-3 py-1 rounded-full border border-amber-200">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            SYSTEM ONLINE
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-7xl">
        
        {/* Top Finding Hero */}
        <TopFindingHero finding={MEDICAL_DATA.summary} />

        {/* Controls */}
        <div className="flex flex-col md:flex-row justify-between items-center gap-4 mb-8">
          
          {/* Tabs */}
          <div className="flex bg-slate-100 p-1 rounded-xl border border-slate-200">
            {["All", "High Risk", "Moderate", "Low"].map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  filter === f 
                    ? "bg-white text-amber-600 shadow-sm border border-slate-200" 
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {f}
              </button>
            ))}
          </div>

          {/* Search */}
          <div className="relative w-full md:w-64">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
            <input 
              type="text" 
              placeholder="Filter diagnosis..." 
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full bg-slate-100 border border-slate-200 rounded-xl pl-10 pr-4 py-2 text-sm text-slate-900 focus:ring-2 focus:ring-amber-500/30 focus:border-amber-500/50 outline-none transition-all placeholder:text-slate-500"
            />
          </div>
        </div>

        {/* Visualization Container - Updated for Strict Constraints */}
        <div className="relative h-[650px] w-full flex items-center justify-center overflow-hidden rounded-3xl bg-slate-50 border border-slate-200 shadow-xl cursor-crosshair">
          
          <RadarGrid />

          {/* Constraint Box: Strictly matches the 500x500 visual circle area */}
          <div 
            ref={constraintsRef} 
            className="absolute w-[500px] h-[500px] rounded-full pointer-events-none"
          />

          {/* Orbit System */}
          <div className="relative z-10 w-full h-full">
            <AnimatePresence>
              {filteredData.map((finding, index) => (
                <RiskOrb 
                  key={finding.disease} 
                  data={finding} 
                  index={index}
                  total={filteredData.length}
                  isSelected={selectedFinding?.disease === finding.disease}
                  onClick={setSelectedFinding}
                  constraintsRef={constraintsRef}
                />
              ))}
            </AnimatePresence>
          </div>
          
          {/* Empty State */}
          {filteredData.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center text-slate-500 z-20">
              No findings match your criteria
            </div>
          )}
        </div>
        
        <p className="text-center text-slate-500 text-xs mt-8 font-mono">
          AI Analysis generated on {new Date(MEDICAL_DATA.timestamp).toLocaleDateString()} via VisualCortex Engine
        </p>
      </main>

      {/* Details Panel Overlay */}
      <AnimatePresence>
        {selectedFinding && (
          <>
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelectedFinding(null)}
              className="fixed inset-0 bg-slate-900/20 backdrop-blur-sm z-40"
            />
            <DetailsPanel data={selectedFinding} onClose={() => setSelectedFinding(null)} />
          </>
        )}
      </AnimatePresence>
    </div>
  );
}