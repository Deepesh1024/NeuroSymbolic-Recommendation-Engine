import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
import re
import json
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()

# --------------------------------------------------------
# 1. Graph-Based Knowledge Representation System
# --------------------------------------------------------

class GraphRecommendationSystem:
    def __init__(self):
        self.job_graph = nx.DiGraph()
        self.developer_graphs = {}
        self.global_graph = nx.Graph()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        
    # --------------------------------------------------------
    # 2. Job Description Graph Construction
    # --------------------------------------------------------
    
    def extract_skills_from_text(self, text: str) -> Dict[str, float]:
        """Fast skill extraction using predefined patterns"""
        
        # Comprehensive skill patterns with importance weights
        skill_patterns = {
            # Programming Languages
            'Python': (r'\bpython\b', 0.9),
            'C++': (r'\bc\+\+|cpp\b', 0.8),
            'CUDA': (r'\bcuda\b', 0.7),
            'JAX': (r'\bjax\b', 0.7),
            
            # ML/AI Frameworks
            'PyTorch': (r'\bpytorch|torch\b', 0.8),
            'PyTorch3D': (r'\bpytorch3d\b', 0.75),
            'Kaolin': (r'\bkaolin\b', 0.7),
            'TensorFlow': (r'\btensorflow|tf\b', 0.8),
            
            # 3D/Simulation Technologies
            '3D Simulation': (r'\b3d simulation|simulation.*3d\b', 0.9),
            'NeRF': (r'\bnerf|neural radiance field\b', 0.85),
            'Gaussian Splatting': (r'\bgaussian splatting|gaussian.*splatting\b', 0.85),
            'Isaac Sim': (r'\bisaac sim|isaac.*sim\b', 0.7),
            'Unreal Engine': (r'\bunreal engine|unreal\b', 0.7),
            'Unity': (r'\bunity\b', 0.65),
            'Mujoco': (r'\bmujoco\b', 0.6),
            'Bullet': (r'\bbullet|pybullet\b', 0.6),
            
            # Machine Learning Concepts
            'Reinforcement Learning': (r'\brl\b|\breinforcement learning\b|ppo|sac|ddpg', 0.8),
            'Deep Learning': (r'\bdeep learning|neural network\b', 0.8),
            'Computer Vision': (r'\bcomputer vision|cv\b', 0.75),
            'Differentiable Physics': (r'\bdifferentiable.*physics\b', 0.8),
            'Domain Randomization': (r'\bdomain randomization\b', 0.75),
            'Sim-to-Real': (r'\bsim.*real|sim2real\b', 0.8),
            'Synthetic Data': (r'\bsynthetic data\b', 0.75),
            
            # Infrastructure/DevOps
            'Docker': (r'\bdocker\b', 0.5),
            'Kubernetes': (r'\bkubernetes|k8s\b', 0.5),
            'Ray': (r'\bray\b.*parallel|ray.*distributed', 0.6),
            'Slurm': (r'\bslurm\b', 0.6),
            'CI/CD': (r'\bci/cd|continuous integration\b', 0.5),
            
            # Robotics
            'ROS2': (r'\bros2|ros 2\b', 0.6),
            'ROS': (r'\bros\b(?!\d)', 0.55),
            
            # Graphics/Rendering
            'OpenGL': (r'\bopengl\b', 0.6),
            'Vulkan': (r'\bvulkan\b', 0.6),
        }
        
        skills = {}
        text_lower = text.lower()
        
        for skill_name, (pattern, importance) in skill_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                # Calculate proficiency based on context
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                context_boost = 0.0
                
                # Context-based importance boosting
                if re.search(rf'{pattern}.*expert|expert.*{pattern}', text_lower, re.IGNORECASE):
                    context_boost += 0.2
                elif re.search(rf'{pattern}.*experience|experience.*{pattern}', text_lower, re.IGNORECASE):
                    context_boost += 0.1
                
                # Final skill weight
                skills[skill_name] = min(1.0, importance + context_boost + (matches * 0.05))
        
        return skills
    
    def build_job_graph(self, job_description: str) -> nx.DiGraph:
        """Build job requirements graph"""
        self.job_graph.clear()
        
        # Extract skills and their importance
        skills = self.extract_skills_from_text(job_description)
        
        # Add job node
        self.job_graph.add_node("JOB", type="job", text=job_description)
        
        # Add skill nodes and connections
        for skill, importance in skills.items():
            self.job_graph.add_node(skill, type="skill", importance=importance)
            self.job_graph.add_edge("JOB", skill, weight=importance, relation="requires")
            
            # Add skill relationships (e.g., PyTorch relates to Deep Learning)
            skill_relationships = {
                'PyTorch': ['Deep Learning', 'Python'],
                'PyTorch3D': ['PyTorch', '3D Simulation', 'Computer Vision'],
                'NeRF': ['3D Simulation', 'Deep Learning', 'Computer Vision'],
                'Reinforcement Learning': ['Python', 'Deep Learning'],
                'Isaac Sim': ['3D Simulation', 'CUDA'],
                'Unreal Engine': ['3D Simulation', 'C++'],
            }
            
            if skill in skill_relationships:
                for related_skill in skill_relationships[skill]:
                    if related_skill in skills:
                        self.job_graph.add_edge(skill, related_skill, 
                                              weight=0.3, relation="relates_to")
        
        return self.job_graph
    
    # --------------------------------------------------------
    # 3. Developer Profile Graph Construction
    # --------------------------------------------------------
    
    def build_developer_graph(self, dev_name: str, profile_text: str) -> nx.DiGraph:
        """Build individual developer skill graph"""
        dev_graph = nx.DiGraph()
        
        # Extract skills from developer profile
        skills = self.extract_skills_from_text(profile_text)
        
        # Add developer node
        dev_graph.add_node(dev_name, type="developer", profile=profile_text)
        
        # Add skill nodes with proficiency scores
        for skill, proficiency in skills.items():
            # Enhanced proficiency calculation
            profile_lower = profile_text.lower()
            skill_lower = skill.lower()
            
            # Base proficiency from pattern matching
            base_proficiency = proficiency
            
            # Boost based on experience indicators
            if re.search(r'(\d+).*year.*' + re.escape(skill_lower), profile_lower):
                years_match = re.search(r'(\d+).*year.*' + re.escape(skill_lower), profile_lower)
                if years_match:
                    years = int(years_match.group(1))
                    base_proficiency += min(0.3, years * 0.05)
            
            # Boost for projects
            if re.search(rf'{re.escape(skill_lower)}.*project|project.*{re.escape(skill_lower)}', profile_lower):
                base_proficiency += 0.1
            
            # Boost for advanced terms
            if re.search(rf'{re.escape(skill_lower)}.*\b(expert|advanced|mastery|extensive)\b', profile_lower):
                base_proficiency += 0.2
            
            final_proficiency = min(1.0, base_proficiency)
            
            dev_graph.add_node(skill, type="skill", proficiency=final_proficiency)
            dev_graph.add_edge(dev_name, skill, weight=final_proficiency, relation="proficient_in")
        
        self.developer_graphs[dev_name] = dev_graph
        return dev_graph
    
    # --------------------------------------------------------
    # 4. Graph Matching and Scoring Algorithm
    # --------------------------------------------------------
    
    def calculate_match_score(self, dev_name: str) -> Tuple[float, Dict]:
        """Calculate match score between developer and job using graph analysis"""
        
        if dev_name not in self.developer_graphs:
            return 0.0, {}
        
        dev_graph = self.developer_graphs[dev_name]
        job_skills = {node: data for node, data in self.job_graph.nodes(data=True) 
                     if data.get('type') == 'skill'}
        
        total_score = 0.0
        max_possible_score = 0.0
        skill_scores = {}
        
        for skill_node, job_data in job_skills.items():
            importance = job_data.get('importance', 0.5)
            max_possible_score += importance
            
            if skill_node in dev_graph.nodes():
                # Direct skill match
                dev_skill_data = dev_graph.nodes[skill_node]
                proficiency = dev_skill_data.get('proficiency', 0.0)
                
                skill_score = proficiency * importance
                total_score += skill_score
                
                skill_scores[skill_node] = {
                    'proficiency': proficiency,
                    'importance': importance,
                    'score': skill_score,
                    'match_type': 'direct'
                }
            else:
                # Check for related skills using graph relationships
                related_score = 0.0
                best_related = None
                
                for dev_skill in dev_graph.nodes():
                    if dev_graph.nodes[dev_skill].get('type') == 'skill':
                        # Calculate semantic similarity for related skills
                        similarity = self.calculate_skill_similarity(skill_node, dev_skill)
                        if similarity > 0.6:  # Threshold for related skills
                            dev_proficiency = dev_graph.nodes[dev_skill].get('proficiency', 0.0)
                            related_skill_score = dev_proficiency * similarity * 0.7  # Penalty for indirect match
                            
                            if related_skill_score > related_score:
                                related_score = related_skill_score
                                best_related = dev_skill
                
                if related_score > 0:
                    final_score = related_score * importance
                    total_score += final_score
                    
                    skill_scores[skill_node] = {
                        'proficiency': related_score,
                        'importance': importance,
                        'score': final_score,
                        'match_type': 'related',
                        'related_skill': best_related
                    }
                else:
                    skill_scores[skill_node] = {
                        'proficiency': 0.0,
                        'importance': importance,
                        'score': 0.0,
                        'match_type': 'missing'
                    }
        
        # Normalize score
        normalized_score = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        
        return normalized_score, skill_scores
    
    def calculate_skill_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate semantic similarity between skills"""
        try:
            # Use embeddings for semantic similarity
            emb1 = self.embedder.encode(skill1)
            emb2 = self.embedder.encode(skill2)
            
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except:
            return 0.0
    
    # --------------------------------------------------------
    # 5. Recommendation Engine
    # --------------------------------------------------------
    
    def rank_candidates(self, job_description: str, developer_profiles: Dict[str, str]) -> List[Tuple]:
        """Main ranking function using graph-based analysis"""
        
        print("🔧 Building job requirements graph...")
        self.build_job_graph(job_description)
        
        print("👥 Building developer skill graphs...")
        for dev_name, profile in developer_profiles.items():
            self.build_developer_graph(dev_name, profile)
        
        print("⚡ Calculating match scores...")
        results = []
        
        for dev_name in developer_profiles.keys():
            score, skill_breakdown = self.calculate_match_score(dev_name)
            results.append((dev_name, score, skill_breakdown))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def generate_recommendations(self, results: List[Tuple]) -> Dict:
        """Generate detailed recommendations"""
        
        recommendations = {
            'ranking': [],
            'insights': {},
            'job_requirements_analysis': {}
        }
        
        # Job requirements analysis
        job_skills = {node: data for node, data in self.job_graph.nodes(data=True) 
                     if data.get('type') == 'skill'}
        
        recommendations['job_requirements_analysis'] = {
            'total_skills_required': len(job_skills),
            'critical_skills': [skill for skill, data in job_skills.items() 
                               if data.get('importance', 0) > 0.8],
            'important_skills': [skill for skill, data in job_skills.items() 
                                if 0.6 < data.get('importance', 0) <= 0.8],
            'nice_to_have_skills': [skill for skill, data in job_skills.items() 
                                   if data.get('importance', 0) <= 0.6]
        }
        
        # Candidate analysis
        for rank, (dev_name, score, skill_breakdown) in enumerate(results, 1):
            
            # Calculate strengths and gaps
            strengths = []
            gaps = []
            partial_matches = []
            
            for skill, details in skill_breakdown.items():
                if details['match_type'] == 'direct' and details['proficiency'] > 0.7:
                    strengths.append(skill)
                elif details['match_type'] == 'related':
                    partial_matches.append(skill)
                elif details['match_type'] == 'missing' and details['importance'] > 0.6:
                    gaps.append(skill)
            
            candidate_analysis = {
                'rank': rank,
                'overall_score': round(score, 2),
                'match_percentage': f"{score:.1f}%",
                'strengths': strengths,
                'partial_matches': partial_matches,
                'critical_gaps': gaps,
                'recommendation': self.generate_candidate_recommendation(score, strengths, gaps)
            }
            
            recommendations['ranking'].append((dev_name, candidate_analysis))
            recommendations['insights'][dev_name] = skill_breakdown
        
        return recommendations
    
    def generate_candidate_recommendation(self, score: float, strengths: List, gaps: List) -> str:
        """Generate recommendation text for each candidate"""
        
        if score >= 80:
            return f"🌟 EXCELLENT MATCH - Strong candidate with {len(strengths)} key strengths. Highly recommended for interview."
        elif score >= 65:
            return f"✅ GOOD MATCH - Solid candidate with {len(strengths)} strengths. Consider for interview with focus on {len(gaps)} skill gaps."
        elif score >= 50:
            return f"⚠️ PARTIAL MATCH - Has potential but significant gaps in {len(gaps)} areas. May need training or different role scope."
        else:
            return f"❌ POOR MATCH - Limited alignment with job requirements. Not recommended for this specific role."

# --------------------------------------------------------
# 6. Main Execution Function (FIXED)
# --------------------------------------------------------

def run_graph_recommendation_system():
    """Main function to run the complete recommendation system"""
    
    # Initialize system
    system = GraphRecommendationSystem()
    
    # Job description
    job_description = """
    Simulation/Machine Learning Engineer — 3D Simulation & Generative Modeling
    Role Summary
    Design and deploy ML-driven 3D simulation systems that model complex real-world environments, agents, and physics. You will fuse differentiable simulation, generative 3D (NeRF/Gaussian Splatting/Mesh), and reinforcement learning to create high-fidelity, data-efficient virtual worlds for experimentation, training, and digital twin use cases.
    Key Responsibilities
    Build end-to-end simulation pipelines: scene synthesis, asset generation, physics, sensor models, and evaluation.
    Develop ML models for 3D representation (implicit & explicit), sim-to-real transfer, and domain randomization.
    Train RL/IL agents within simulators; optimize for sample efficiency and stability.
    Implement differentiable/grad-aware physics or surrogate models to accelerate learning.
    Create synthetic data engines for perception (depth, segmentation, pose, LiDAR/Radar).
    Optimize performance for GPU clusters; profile memory/throughput; enable scalable batch sims.
    Establish metrics and CI for simulation fidelity, generalization, and downstream task ROI.
    Collaborate with product, research, and platform teams to align scenarios with business goals.
    Minimum Qualifications
    3–6+ years in ML or simulation (or equivalent research experience).
    Strong Python; solid C++ for performance-critical components.
    Proficiency in PyTorch/JAX; CUDA fundamentals and vectorization.
    Hands-on with at least one real-time or offline engine: Unity, Unreal, Omniverse/Isaac Sim, Mujoco, Bullet, PyBullet, ODE, NVIDIA Warp.
    Experience with 3D geometry & vision: camera models, SFM/SLAM, meshes/point clouds, NeRFs, Gaussians, TSDF/Occupancy.
    Competence in RL or control (PPO/SAC/DDPG, model-based RL, MPC).
    Understanding of numerical methods & physics (rigid/soft bodies, constraints, integrators).
    Strong software engineering: testing, CI/CD, containers, experiment tracking.
    Preferred Qualifications
    Differentiable rendering (Mitsuba, Kaolin, PyTorch3D) or differentiable physics (Brax, Tiny Differentiable Simulator).
    Synthetic data for CV: domain randomization, photoreal lighting/material pipelines.
    Sim-to-real: dynamics randomization, residual learning, system ID.
    Scaling: Ray/Slurm/K8s; multi-GPU/multi-node training; mixed precision.
    3D generative models (Score-based, diffusion-based, Gaussian Splatting, Instant-NGP).
    Robotics stacks (ROS2), sensor simulation (LiDAR, radar), and embedded deployment.
    Publications or open-source contributions in sim/3D/RL.
    Tooling & Stack (example)
    Languages: Python, C++, CUDA
    ML: PyTorch/JAX, PyTorch3D/Kaolin, Diffusers
    Sim/3D: Isaac Sim/Omniverse, Unity/Unreal, Mujoco, Bullet, Blender
    Data/Infra: Ray, Dask, Weights & Biases/MLflow, Docker, Slurm/K8s
    Rendering: USD, MaterialX, HDRI pipelines
    """
    
    # Developer profiles
    developer_profiles = {
        "Ananya Sharma": """
        Built end-to-end 3D simulation pipelines in Isaac Sim & Unreal Engine, with GPU-accelerated synthetic data generation for autonomous navigation projects.
        Extensive hands-on with NeRFs, Gaussian Splatting, PyTorch3D, Kaolin, and differentiable rendering (Mitsuba).
        Strong foundation in RL (PPO, SAC, DreamerV3) — deployed trained policies from sim-to-real on quadruped robots.
        Optimized large-scale simulation workloads using Ray + Slurm, achieving 10x faster rollouts in parallel clusters.
        Published research in differentiable physics and domain randomization for sim-to-real transfer.
        Moderate exposure to ROS2. 5 years experience in ML & 3D simulation, M.S. in Computer Vision & Robotics, NVIDIA Omniverse team background.
        """,
        "Raghav Mehta": """
        Hands-on with PyTorch, JAX, and RL frameworks — trained agents in Mujoco & PyBullet.
        Solid understanding of numerical optimization, RL algorithms, and control theory.
        Experience with real-world robotics (ROS1, embedded deployments).
        Strong Python coder, clean experimentation practices, CI/CD familiar.
        Limited exposure to photorealistic 3D simulation (has only used Mujoco/Bullet, no Unreal/Omniverse).
        No direct experience with differentiable rendering, NeRF, or synthetic data pipelines.
        GPU optimization knowledge is basic — hasn't scaled beyond single GPU. 3 years in ML research.
        """,
        "Priya Nair": """
        Expert in Unreal, Unity, and GPU rendering pipelines — built large-scale simulation environments for gaming and AR/VR.
        Mastery of C++, CUDA, OpenGL, Vulkan for high-performance graphics.
        Led teams delivering production-quality simulation engines with photoreal realism.
        Excellent in project leadership, architecture design, and optimization.
        Limited exposure to machine learning & RL — hasn't worked with PyTorch/JAX or trained ML-based agents.
        No hands-on with NeRFs, differentiable physics, or generative 3D models.
        Strong graphics expertise, but missing the ML-driven simulation integration required for this role. 12+ years in graphics simulation.
        """
    }
    
    # Run analysis
    print("🚀 Starting Graph-Based Recommendation System")
    print("=" * 60)
    
    results = system.rank_candidates(job_description, developer_profiles)
    recommendations = system.generate_recommendations(results)
    
    # Display results
    print("\n📊 JOB REQUIREMENTS ANALYSIS")
    print("-" * 40)
    analysis = recommendations['job_requirements_analysis']
    print(f"Total Skills Required: {analysis['total_skills_required']}")
    print(f"Critical Skills: {', '.join(analysis['critical_skills'])}")
    print(f"Important Skills: {', '.join(analysis['important_skills'])}")
    print(f"Nice-to-Have: {', '.join(analysis['nice_to_have_skills'])}")
    
    print(f"\n🏆 CANDIDATE RANKING & RECOMMENDATIONS")
    print("=" * 60)
    
    for dev_name, analysis in recommendations['ranking']:
        print(f"\n#{analysis['rank']} {dev_name}")
        print(f"   Score: {analysis['overall_score']}/100 ({analysis['match_percentage']})")
        print(f"   {analysis['recommendation']}")
        print(f"   ✅ Strengths ({len(analysis['strengths'])}): {', '.join(analysis['strengths'][:3])}{'...' if len(analysis['strengths']) > 3 else ''}")
        if analysis['critical_gaps']:
            print(f"   ⚠️  Critical Gaps: {', '.join(analysis['critical_gaps'])}")
        if analysis['partial_matches']:
            print(f"   🔄 Partial Matches: {', '.join(analysis['partial_matches'])}")
    
    print(f"\n✨ FINAL RECOMMENDATION")
    print("-" * 40)
    best_candidate = recommendations['ranking'][0]
    print(f"🎯 RECOMMENDED HIRE: {best_candidate}")
    print(f"   {best_candidate[1]['recommendation']}")  # Fixed: access dict properly
    
    return recommendations

# --------------------------------------------------------
# 7. Run the system
# --------------------------------------------------------

if __name__ == "__main__":
    recommendations = run_graph_recommendation_system()
