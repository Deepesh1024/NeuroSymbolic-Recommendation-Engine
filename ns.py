import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set
import re
import json
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------
# 1. Data Structures
# --------------------------------------------------------

@dataclass
class ReasoningStep:
    step_type: str  # 'match', 'infer', 'validate', 'synthesize'
    input_facts: List[str]
    reasoning_rule: str
    output_conclusion: str
    confidence: float

# --------------------------------------------------------
# 2. Neural Knowledge Graph Learner (Simplified for Demo)
# --------------------------------------------------------

class NeuralKnowledgeGraphLearner:
    """Simplified neural component for skill relationship learning"""
    
    def __init__(self, embedding_dim=256):
        self.embedding_dim = embedding_dim
        self.skill_embeddings = {}
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def get_skill_embedding(self, skill: str) -> np.ndarray:
        if skill not in self.skill_embeddings:
            self.skill_embeddings[skill] = self.embedder.encode(skill)
        return self.skill_embeddings[skill]
    
    def calculate_similarity(self, skill1: str, skill2: str) -> float:
        emb1 = self.get_skill_embedding(skill1)
        emb2 = self.get_skill_embedding(skill2)
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

# --------------------------------------------------------
# 3. Multi-Step Logical Reasoning Engine
# --------------------------------------------------------

class LogicalReasoningEngine:
    """Advanced logical reasoning with multi-step inference"""
    
    def __init__(self):
        self.reasoning_rules = self.load_reasoning_rules()
        self.inference_history = []
        
    def load_reasoning_rules(self) -> Dict[str, Dict]:
        """Define comprehensive logical reasoning rules"""
        return {
            # Skill implication rules
            'skill_implies': {
                'PyTorch3D': {'implies': ['PyTorch', 'Computer Vision', '3D Graphics'], 'confidence': 0.9},
                'NeRF': {'implies': ['Deep Learning', 'Computer Vision', '3D Reconstruction'], 'confidence': 0.85},
                'Gaussian Splatting': {'implies': ['NeRF', '3D Graphics', 'Computer Vision'], 'confidence': 0.9},
                'Isaac Sim': {'implies': ['CUDA', '3D Simulation', 'Robotics'], 'confidence': 0.8},
                'Unreal Engine': {'implies': ['3D Graphics', 'C++', 'Game Development'], 'confidence': 0.85},
                'Differentiable Physics': {'implies': ['PyTorch', 'Numerical Methods', 'Physics'], 'confidence': 0.85},
                'Reinforcement Learning': {'implies': ['Python', 'Deep Learning', 'Machine Learning'], 'confidence': 0.9},
                'Ray': {'implies': ['Python', 'Distributed Computing', 'Parallel Processing'], 'confidence': 0.8},
                'Slurm': {'implies': ['Linux', 'HPC', 'Job Scheduling'], 'confidence': 0.75}
            },
            
            # Experience level inference
            'experience_inference': {
                'senior_indicators': ['led teams', 'architecture', 'published research', 'optimization', 'expert', 'mastery'],
                'intermediate_indicators': ['hands-on', 'experience with', 'familiar with', 'solid understanding'],
                'junior_indicators': ['exposure to', 'basic knowledge', 'learning', 'moderate']
            },
            
            # Transferable skills
            'skill_transfer': {
                ('Unreal Engine', 'Unity'): 0.8,
                ('PyTorch', 'TensorFlow'): 0.7,
                ('CUDA', 'OpenCL'): 0.6,
                ('ROS', 'ROS2'): 0.9,
                ('Mujoco', 'Bullet'): 0.7,
                ('Isaac Sim', 'Gazebo'): 0.6
            },
            
            # Complementary skills (synergy multipliers)
            'skill_synergy': {
                frozenset(['Python', 'PyTorch', 'CUDA']): 1.2,
                frozenset(['C++', 'Unreal Engine', '3D Graphics']): 1.15,
                frozenset(['NeRF', 'Gaussian Splatting', 'Computer Vision']): 1.3,
                frozenset(['Reinforcement Learning', 'Python', 'Deep Learning']): 1.25,
                frozenset(['3D Simulation', 'Isaac Sim', 'Robotics']): 1.2
            }
        }
    
    def multi_step_reasoning(self, candidate_profile: str, job_requirements: Dict[str, float]) -> List[ReasoningStep]:
        """Perform comprehensive multi-step logical reasoning"""
        
        reasoning_chain = []
        candidate_facts = self.extract_facts(candidate_profile)
        
        # Step 1: Direct skill matching
        direct_matches = self.step1_direct_matching(candidate_facts, job_requirements)
        reasoning_chain.extend(direct_matches)
        
        # Step 2: Skill implication inference
        implied_skills = self.step2_skill_implication(candidate_facts)
        reasoning_chain.extend(implied_skills)
        
        # Step 3: Experience level inference
        experience_level = self.step3_experience_inference(candidate_profile)
        reasoning_chain.extend(experience_level)
        
        # Step 4: Transferable skill analysis
        transferable = self.step4_transferable_skills(candidate_facts, job_requirements)
        reasoning_chain.extend(transferable)
        
        # Step 5: Skill synergy analysis
        synergies = self.step5_skill_synergy(candidate_facts)
        reasoning_chain.extend(synergies)
        
        # Step 6: Gap analysis and recommendations
        gaps = self.step6_gap_analysis(candidate_facts, job_requirements, reasoning_chain)
        reasoning_chain.extend(gaps)
        
        self.inference_history.append(reasoning_chain)
        return reasoning_chain
    
    def step1_direct_matching(self, candidate_facts: Set[str], job_requirements: Dict[str, float]) -> List[ReasoningStep]:
        """Direct skill matching with confidence scores"""
        steps = []
        
        for skill, importance in job_requirements.items():
            # Check for direct mentions with fuzzy matching
            for fact in candidate_facts:
                if self.fuzzy_skill_match(skill, fact):
                    step = ReasoningStep(
                        step_type='match',
                        input_facts=[f"Candidate profile contains: '{fact}'"],
                        reasoning_rule="Direct skill match detected",
                        output_conclusion=f"Candidate directly matches required skill: {skill}",
                        confidence=0.95
                    )
                    steps.append(step)
                    break
        
        return steps
    
    def fuzzy_skill_match(self, skill: str, fact: str) -> bool:
        """Fuzzy matching for skill detection"""
        skill_lower = skill.lower()
        fact_lower = fact.lower()
        
        # Direct match
        if skill_lower in fact_lower:
            return True
            
        # Handle variations
        skill_variations = {
            'nerf': ['neural radiance field', 'nerf'],
            'gaussian splatting': ['gaussian', 'splatting'],
            'reinforcement learning': ['rl', 'reinforcement learning', 'ppo', 'sac', 'ddpg'],
            'computer vision': ['cv', 'computer vision', 'vision'],
            '3d simulation': ['3d', 'simulation', '3d simulation'],
            'pytorch': ['pytorch', 'torch'],
            'c++': ['c++', 'cpp'],
            'differentiable physics': ['differentiable physics', 'differentiable'],
            'domain randomization': ['domain randomization'],
            'sim-to-real': ['sim-to-real', 'sim2real']
        }
        
        for canonical_skill, variations in skill_variations.items():
            if canonical_skill in skill_lower:
                return any(var in fact_lower for var in variations)
        
        return False
    
    def step2_skill_implication(self, candidate_facts: Set[str]) -> List[ReasoningStep]:
        """Infer implied skills from stated skills"""
        steps = []
        
        for fact in candidate_facts:
            for skill, rule_data in self.reasoning_rules['skill_implies'].items():
                if self.fuzzy_skill_match(skill, fact):
                    for implied_skill in rule_data['implies']:
                        step = ReasoningStep(
                            step_type='infer',
                            input_facts=[f"Candidate has experience with {skill}"],
                            reasoning_rule=f"IF has {skill} THEN likely has {implied_skill}",
                            output_conclusion=f"Inferred skill: {implied_skill}",
                            confidence=rule_data['confidence']
                        )
                        steps.append(step)
        
        return steps
    
    def step3_experience_inference(self, candidate_profile: str) -> List[ReasoningStep]:
        """Infer experience level from language patterns"""
        steps = []
        profile_lower = candidate_profile.lower()
        
        # Count indicators
        senior_count = sum(1 for indicator in self.reasoning_rules['experience_inference']['senior_indicators'] 
                          if indicator in profile_lower)
        intermediate_count = sum(1 for indicator in self.reasoning_rules['experience_inference']['intermediate_indicators'] 
                               if indicator in profile_lower)
        
        # Extract years of experience
        years_match = re.search(r'(\d+)\s*years?\s*(?:of\s*)?(?:experience|exp)', profile_lower)
        years = int(years_match.group(1)) if years_match else 0
        
        if senior_count >= 2 or years >= 5:
            step = ReasoningStep(
                step_type='infer',
                input_facts=[f"Profile contains {senior_count} senior indicators, {years} years experience"],
                reasoning_rule="IF profile has senior language patterns OR 5+ years THEN senior experience level",
                output_conclusion=f"Inferred experience level: Senior ({years} years)",
                confidence=0.8
            )
            steps.append(step)
        elif intermediate_count >= 2 or years >= 2:
            step = ReasoningStep(
                step_type='infer',
                input_facts=[f"Profile contains {intermediate_count} intermediate indicators"],
                reasoning_rule="IF profile has intermediate language patterns THEN intermediate experience",
                output_conclusion="Inferred experience level: Intermediate",
                confidence=0.7
            )
            steps.append(step)
        
        return steps
    
    def step4_transferable_skills(self, candidate_facts: Set[str], job_requirements: Dict[str, float]) -> List[ReasoningStep]:
        """Analyze transferable skills"""
        steps = []
        
        for (skill1, skill2), transfer_score in self.reasoning_rules['skill_transfer'].items():
            has_skill1 = any(self.fuzzy_skill_match(skill1, fact) for fact in candidate_facts)
            needs_skill2 = skill2 in job_requirements
            
            if has_skill1 and needs_skill2:
                step = ReasoningStep(
                    step_type='infer',
                    input_facts=[f"Candidate has {skill1}", f"Job requires {skill2}"],
                    reasoning_rule=f"Skills {skill1} and {skill2} are transferable (score: {transfer_score})",
                    output_conclusion=f"Transferable skill match: {skill1} → {skill2}",
                    confidence=transfer_score
                )
                steps.append(step)
        
        return steps
    
    def step5_skill_synergy(self, candidate_facts: Set[str]) -> List[ReasoningStep]:
        """Identify skill synergies and combinations"""
        steps = []
        
        # Extract candidate skills
        candidate_skills = set()
        for fact in candidate_facts:
            for skill in ['Python', 'PyTorch', 'CUDA', 'C++', 'Unreal Engine', '3D Graphics',
                         'NeRF', 'Gaussian Splatting', 'Computer Vision', 'Reinforcement Learning',
                         'Deep Learning', '3D Simulation', 'Isaac Sim', 'Robotics']:
                if self.fuzzy_skill_match(skill, fact):
                    candidate_skills.add(skill)
        
        for skill_combo, multiplier in self.reasoning_rules['skill_synergy'].items():
            if skill_combo.issubset(candidate_skills):
                step = ReasoningStep(
                    step_type='synthesize',
                    input_facts=[f"Candidate has complementary skills: {', '.join(skill_combo)}"],
                    reasoning_rule=f"Skill combination synergy multiplier: {multiplier}",
                    output_conclusion=f"Strong skill synergy detected (multiplier: {multiplier})",
                    confidence=0.9
                )
                steps.append(step)
        
        return steps
    
    def step6_gap_analysis(self, candidate_facts: Set[str], job_requirements: Dict[str, float], 
                          reasoning_chain: List[ReasoningStep]) -> List[ReasoningStep]:
        """Identify critical gaps and provide recommendations"""
        steps = []
        
        # Extract skills covered by reasoning chain
        covered_skills = set()
        for step in reasoning_chain:
            if step.step_type in ['match', 'infer']:
                # Extract skill from conclusion
                if 'skill:' in step.output_conclusion:
                    skill = step.output_conclusion.split('skill:')[1].strip()
                    covered_skills.add(skill.lower())
                elif 'matches required skill:' in step.output_conclusion:
                    skill = step.output_conclusion.split('matches required skill:')[1].strip()
                    covered_skills.add(skill.lower())
        
        # Find critical gaps
        critical_gaps = []
        for skill, importance in job_requirements.items():
            if importance > 0.7:
                has_direct = any(self.fuzzy_skill_match(skill, fact) for fact in candidate_facts)
                has_inferred = skill.lower() in covered_skills
                
                if not has_direct and not has_inferred:
                    critical_gaps.append(skill)
        
        if critical_gaps:
            step = ReasoningStep(
                step_type='validate',
                input_facts=[f"Required critical skills: {critical_gaps}"],
                reasoning_rule="Critical gap analysis for high-importance skills (>0.7)",
                output_conclusion=f"Critical skill gaps identified: {', '.join(critical_gaps)}",
                confidence=0.95
            )
            steps.append(step)
        
        return steps
    
    def extract_facts(self, candidate_profile: str) -> Set[str]:
        """Extract factual statements from candidate profile"""
        facts = set()
        
        # Split into sentences and clean
        sentences = re.split(r'[.!?]+', candidate_profile)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Meaningful sentences only
                facts.add(sentence)
        
        return facts

# --------------------------------------------------------
# 4. Dynamic Knowledge Graph Manager
# --------------------------------------------------------

class DynamicKnowledgeGraph:
    """Manage knowledge graph with learning capabilities"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.neural_learner = NeuralKnowledgeGraphLearner()
        self.initialize_graph()
    
    def initialize_graph(self):
        """Initialize with basic skill relationships"""
        basic_relationships = [
            ('PyTorch', 'Deep Learning', 0.9),
            ('PyTorch3D', 'PyTorch', 0.85),
            ('PyTorch3D', 'Computer Vision', 0.8),
            ('NeRF', 'Computer Vision', 0.85),
            ('NeRF', 'Deep Learning', 0.8),
            ('Gaussian Splatting', 'NeRF', 0.9),
            ('Isaac Sim', 'CUDA', 0.7),
            ('Isaac Sim', '3D Simulation', 0.9),
            ('Unreal Engine', 'C++', 0.8),
            ('Reinforcement Learning', 'Deep Learning', 0.85),
            ('Ray', 'Python', 0.7),
            ('Slurm', 'HPC', 0.8)
        ]
        
        for skill1, skill2, weight in basic_relationships:
            self.graph.add_edge(skill1, skill2, weight=weight, type='static')
    
    def find_skill_path(self, from_skill: str, to_skill: str) -> Tuple[List[str], float]:
        """Find connection path between skills"""
        try:
            path = nx.shortest_path(self.graph, from_skill, to_skill, weight=lambda u, v, d: 1-d['weight'])
            path_score = self.calculate_path_score(path)
            return path, path_score
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], 0.0
    
    def calculate_path_score(self, path: List[str]) -> float:
        """Calculate score for a path through the knowledge graph"""
        if len(path) <= 1:
            return 1.0
        
        total_weight = 1.0
        for i in range(len(path) - 1):
            if self.graph.has_edge(path[i], path[i+1]):
                edge_weight = self.graph[path[i]][path[i+1]].get('weight', 0.5)
                total_weight *= edge_weight
        
        # Penalize longer paths
        path_penalty = 0.9 ** (len(path) - 2)
        return total_weight * path_penalty

# --------------------------------------------------------
# 5. Complete Neurosymbolic Recommendation System
# --------------------------------------------------------

class AdvancedNeurosymbolicRecommendationSystem:
    """Complete neurosymbolic system with learning and reasoning"""
    
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.knowledge_graph = DynamicKnowledgeGraph()
        self.reasoning_engine = LogicalReasoningEngine()
        
    def extract_skills_from_text(self, text: str) -> Dict[str, float]:
        """Extract skills with importance/proficiency scores"""
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
            
            # 3D/Simulation Technologies
            '3D Simulation': (r'\b3d simulation|simulation.*3d\b', 0.9),
            'NeRF': (r'\bnerf|neural radiance field\b', 0.85),
            'Gaussian Splatting': (r'\bgaussian splatting|gaussian.*splatting\b', 0.85),
            'Isaac Sim': (r'\bisaac sim|isaac.*sim\b', 0.7),
            'Unreal Engine': (r'\bunreal engine|unreal\b', 0.7),
            'Unity': (r'\bunity\b', 0.65),
            'Mujoco': (r'\bmujoco\b', 0.6),
            'Bullet': (r'\bbullet|pybullet\b', 0.6),
            
            # ML Concepts
            'Reinforcement Learning': (r'\brl\b|\breinforcement learning\b|ppo|sac|ddpg|dreamerv3', 0.8),
            'Deep Learning': (r'\bdeep learning|neural network\b', 0.8),
            'Computer Vision': (r'\bcomputer vision|cv\b', 0.75),
            'Differentiable Physics': (r'\bdifferentiable.*physics\b', 0.8),
            'Domain Randomization': (r'\bdomain randomization\b', 0.75),
            'Sim-to-Real': (r'\bsim.*real|sim2real\b', 0.8),
            'Synthetic Data': (r'\bsynthetic data\b', 0.75),
            
            # Infrastructure
            'Docker': (r'\bdocker\b', 0.5),
            'Kubernetes': (r'\bkubernetes|k8s\b', 0.5),
            'Ray': (r'\bray\b', 0.6),
            'Slurm': (r'\bslurm\b', 0.6),
            'CI/CD': (r'\bci/cd|continuous integration\b', 0.5),
            'ROS2': (r'\bros2|ros 2\b', 0.6)
        }
        
        skills = {}
        text_lower = text.lower()
        
        for skill_name, (pattern, base_importance) in skill_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                
                # Context boosting
                context_boost = 0.0
                if re.search(rf'{pattern}.*expert|expert.*{pattern}', text_lower, re.IGNORECASE):
                    context_boost += 0.2
                elif re.search(rf'{pattern}.*experience|experience.*{pattern}', text_lower, re.IGNORECASE):
                    context_boost += 0.1
                
                # Years boosting
                years_match = re.search(rf'(\d+).*year.*{pattern}|{pattern}.*(\d+).*year', text_lower, re.IGNORECASE)
                if years_match:
                    years = int(years_match.group(1) or years_match.group(2) or 0)
                    context_boost += min(0.3, years * 0.05)
                
                final_importance = min(1.0, base_importance + context_boost + (matches * 0.02))
                skills[skill_name] = final_importance
        
        return skills
    
    def evaluate_candidate(self, candidate_profile: str, job_requirements: Dict[str, float]) -> Dict:
        """Complete neurosymbolic evaluation"""
        
        # Step 1: Multi-step logical reasoning
        reasoning_chain = self.reasoning_engine.multi_step_reasoning(
            candidate_profile, job_requirements
        )
        
        # Step 2: Neural similarity computation
        neural_scores = self.compute_neural_similarities(candidate_profile, job_requirements)
        
        # Step 3: Graph-based relationship analysis
        graph_scores = self.analyze_graph_relationships(candidate_profile, job_requirements)
        
        # Step 4: Combine all evidence
        final_score, explanation = self.synthesize_evidence(
            reasoning_chain, neural_scores, graph_scores, job_requirements
        )
        
        return {
            'final_score': final_score,
            'reasoning_chain': reasoning_chain,
            'neural_scores': neural_scores,
            'graph_scores': graph_scores,
            'explanation': explanation,
            'recommendation': self.generate_recommendation(final_score)
        }
    
    def compute_neural_similarities(self, candidate_profile: str, job_requirements: Dict[str, float]) -> Dict:
        """Compute neural similarity scores"""
        candidate_embedding = self.embedder.encode(candidate_profile)
        
        neural_scores = {}
        for skill, importance in job_requirements.items():
            skill_embedding = self.embedder.encode(skill)
            similarity = np.dot(candidate_embedding, skill_embedding) / (
                np.linalg.norm(candidate_embedding) * np.linalg.norm(skill_embedding)
            )
            neural_scores[skill] = {
                'similarity': float(similarity),
                'weighted_score': float(similarity * importance)
            }
        
        return neural_scores
    
    def analyze_graph_relationships(self, candidate_profile: str, job_requirements: Dict[str, float]) -> Dict:
        """Analyze relationships using knowledge graph"""
        candidate_skills = set(self.extract_skills_from_text(candidate_profile).keys())
        
        graph_scores = {}
        for req_skill, importance in job_requirements.items():
            best_path_score = 0.0
            best_path = []
            
            for cand_skill in candidate_skills:
                path, path_score = self.knowledge_graph.find_skill_path(cand_skill, req_skill)
                if path_score > best_path_score:
                    best_path_score = path_score
                    best_path = path
            
            graph_scores[req_skill] = {
                'path_score': best_path_score,
                'best_path': best_path,
                'weighted_score': best_path_score * importance
            }
        
        return graph_scores
    
    def synthesize_evidence(self, reasoning_chain: List[ReasoningStep], 
                          neural_scores: Dict, graph_scores: Dict, 
                          job_requirements: Dict[str, float]) -> Tuple[float, str]:
        """Combine all evidence into final score and explanation"""
        
        # Weight different types of evidence
        reasoning_weight = 0.5
        neural_weight = 0.25
        graph_weight = 0.25
        
        # Calculate reasoning score
        high_confidence_steps = [step for step in reasoning_chain if step.confidence > 0.7]
        reasoning_score = np.mean([step.confidence for step in high_confidence_steps]) if high_confidence_steps else 0.0
        
        # Calculate neural score
        neural_score = np.mean([scores['similarity'] for scores in neural_scores.values()])
        
        # Calculate graph score
        graph_score = np.mean([scores['path_score'] for scores in graph_scores.values()])
        
        # Apply skill coverage bonus
        matched_skills = len([step for step in reasoning_chain if step.step_type == 'match'])
        total_required_skills = len(job_requirements)
        coverage_bonus = (matched_skills / total_required_skills) * 0.1
        
        # Final weighted score
        final_score = (reasoning_score * reasoning_weight + 
                      neural_score * neural_weight + 
                      graph_score * graph_weight + 
                      coverage_bonus)
        
        # Generate explanation
        explanation = self.generate_explanation(reasoning_chain, neural_scores, graph_scores)
        
        return min(100.0, final_score * 100), explanation
    
    def generate_explanation(self, reasoning_chain: List[ReasoningStep], 
                           neural_scores: Dict, graph_scores: Dict) -> str:
        """Generate human-readable explanation"""
        
        explanation_parts = []
        
        # Reasoning chain summary
        strong_inferences = [step for step in reasoning_chain if step.confidence > 0.8]
        if strong_inferences:
            explanation_parts.append(f"Strong logical inferences: {len(strong_inferences)} high-confidence conclusions")
        
        # Neural similarity insights
        high_similarity_skills = [skill for skill, data in neural_scores.items() if data['similarity'] > 0.4]
        if high_similarity_skills:
            explanation_parts.append(f"High semantic similarity for: {', '.join(high_similarity_skills[:3])}")
        
        # Graph relationship insights
        connected_skills = [skill for skill, data in graph_scores.items() if data['path_score'] > 0.6]
        if connected_skills:
            explanation_parts.append(f"Strong knowledge graph connections for: {', '.join(connected_skills[:3])}")
        
        return " | ".join(explanation_parts)
    
    def generate_recommendation(self, score: float) -> str:
        """Generate hiring recommendation based on score"""
        if score >= 80:
            return "🌟 EXCELLENT MATCH - Highly recommended for interview"
        elif score >= 65:
            return "✅ GOOD MATCH - Strong candidate, recommend interview"
        elif score >= 50:
            return "⚠️ PARTIAL MATCH - Potential candidate with some gaps"
        else:
            return "❌ POOR MATCH - Not recommended for this role"

# --------------------------------------------------------
# 6. Main Execution Function
# --------------------------------------------------------

def run_complete_neurosymbolic_system():
    """Run the complete system on the previous job and candidates"""
    
    system = AdvancedNeurosymbolicRecommendationSystem()
    
    # Job description from previous example
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
    
    # Extract job requirements
    job_requirements = system.extract_skills_from_text(job_description)
    
    # Developer profiles from previous example
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
    
    print("🧠 ADVANCED NEUROSYMBOLIC RECOMMENDATION SYSTEM")
    print("=" * 70)
    
    # Show job requirements
    print("\n📋 JOB REQUIREMENTS ANALYSIS:")
    print("-" * 50)
    critical_skills = [(skill, imp) for skill, imp in job_requirements.items() if imp > 0.8]
    important_skills = [(skill, imp) for skill, imp in job_requirements.items() if 0.6 < imp <= 0.8]
    nice_to_have = [(skill, imp) for skill, imp in job_requirements.items() if imp <= 0.6]
    
    print(f"🔴 Critical Skills ({len(critical_skills)}): {', '.join([s for s, _ in critical_skills[:5]])}")
    print(f"🟡 Important Skills ({len(important_skills)}): {', '.join([s for s, _ in important_skills[:5]])}")
    print(f"🟢 Nice-to-Have ({len(nice_to_have)}): {', '.join([s for s, _ in nice_to_have[:5]])}")
    
    # Evaluate all candidates
    results = []
    for dev_name, profile in developer_profiles.items():
        evaluation = system.evaluate_candidate(profile, job_requirements)
        results.append((dev_name, evaluation))
    
    # Sort by score
    results.sort(key=lambda x: x[1]['final_score'], reverse=True)
    
    # Display detailed results
    print(f"\n🏆 CANDIDATE EVALUATION RESULTS")
    print("=" * 70)
    
    for rank, (dev_name, evaluation) in enumerate(results, 1):
        print(f"\n#{rank} {dev_name}")
        print(f"   Final Score: {evaluation['final_score']:.1f}/100")
        print(f"   {evaluation['recommendation']}")
        print(f"   Explanation: {evaluation['explanation']}")
        
        # Show reasoning chain summary
        reasoning_types = {}
        for step in evaluation['reasoning_chain']:
            reasoning_types[step.step_type] = reasoning_types.get(step.step_type, 0) + 1
        
        print(f"   🔗 Reasoning: {dict(reasoning_types)}")
        
        # Show top neural matches
        top_neural = sorted(evaluation['neural_scores'].items(), 
                           key=lambda x: x[1]['similarity'], reverse=True)[:3]
        neural_summary = [f"{skill}({sim['similarity']:.2f})" for skill, sim in top_neural]
        print(f"   🧮 Top Neural Matches: {', '.join(neural_summary)}")
        
        # Show graph connections
        graph_connections = [skill for skill, data in evaluation['graph_scores'].items() 
                           if data['path_score'] > 0.5]
        if graph_connections:
            print(f"   📊 Graph Connections: {', '.join(graph_connections[:3])}")
    
    # Final recommendation
    print(f"\n✨ FINAL HIRING RECOMMENDATION")
    print("-" * 50)
    best_candidate = results[0]
    print(f"🎯 RECOMMENDED HIRE: {best_candidate}")
    print(f"   Score: {best_candidate[1]['final_score']:.1f}/100")
    print(f"   {best_candidate[1]['recommendation']}")
    
    return results

# --------------------------------------------------------
# 7. Run the Complete System
# --------------------------------------------------------

if __name__ == "__main__":
    results = run_complete_neurosymbolic_system()
