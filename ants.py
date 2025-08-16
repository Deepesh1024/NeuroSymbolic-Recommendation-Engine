import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import re
import json
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dataclasses import dataclass
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# --------------------------------------------------------
# 1. Enhanced LLM Pattern Generator with Robust JSON Parsing
# --------------------------------------------------------

class AdvancedDynamicLLMPatternGenerator:
    """Generate comprehensive patterns with robust JSON handling"""
    
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.llm = ChatGroq(
            model="moonshotai/kimi-k2-instruct", 
            api_key=self.api_key,
            temperature=0.1,
            max_tokens=2000
        )
    
    def generate_comprehensive_skill_extraction(self, job_description: str) -> Dict[str, Dict]:
        """Generate comprehensive skill extraction with highly robust prompts"""
        
        comprehensive_skill_prompt = ChatPromptTemplate.from_template("""
        You are an expert technical recruiter and skills analyst. Extract ALL technical skills from this job description.

        INSTRUCTIONS:
        1. Extract programming languages, frameworks, tools, methodologies
        2. Include domain-specific concepts and technologies
        3. Consider both explicit mentions and implied skills
        4. Return ONLY valid JSON - no explanations or extra text

        REQUIRED JSON FORMAT (exactly):
        {{
            "Python": {{
                "primary_pattern": "\\\\bpython\\\\b",
                "variations": ["py", "python3"],
                "importance": 0.9,
                "skill_category": "programming_language",
                "proficiency_indicators": {{
                    "expert": ["expert", "mastery", "architect"],
                    "senior": ["senior", "extensive", "5+ years"],
                    "intermediate": ["experience", "familiar", "worked with"],
                    "beginner": ["basic", "learning", "exposure"]
                }},
                "context_clues": ["programming", "development"],
                "domain_relevance": 0.9,
                "market_demand": 0.8
            }}
        }}

        Job Description: {job_description}

        Return only the JSON object with at least 8 skills:
        """)
        
        return self._execute_robust_llm_prompt(comprehensive_skill_prompt, {"job_description": job_description[:1200]})
    
    def generate_advanced_skill_relationships(self, extracted_skills: List[str], job_context: str) -> Dict[str, Dict]:
        """Generate skill relationships with simplified but robust prompts"""
        
        relationships_prompt = ChatPromptTemplate.from_template("""
        Create skill relationships for these technical skills. Return ONLY valid JSON.

        Skills: {skills}

        Required JSON format:
        {{
            "PyTorch": {{
                "implies": ["Python", "Deep Learning"],
                "confidence": 0.9,
                "reasoning": "PyTorch requires Python and indicates ML knowledge"
            }}
        }}

        Create relationships for at least 5 skills. Return only JSON:
        """)
        
        return self._execute_robust_llm_prompt(relationships_prompt, {
            "skills": extracted_skills[:8]
        })
    
    def generate_sophisticated_transferable_skills(self, extracted_skills: List[str]) -> Dict[str, Dict]:
        """Generate transferable skills with focus on robust JSON"""
        
        transferable_prompt = ChatPromptTemplate.from_template("""
        Identify transferable skills from this list. Return ONLY valid JSON.

        Skills: {skills}

        Required JSON format:
        {{
            "Unity|Unreal Engine": {{
                "transferability": 0.8,
                "reasoning": "Both are game engines with similar concepts"
            }}
        }}

        Create at least 4 transferable skill pairs. Return only JSON:
        """)
        
        return self._execute_robust_llm_prompt(transferable_prompt, {
            "skills": extracted_skills[:8]
        })
    
    def generate_comprehensive_skill_synergies(self, extracted_skills: List[str]) -> Dict[str, Dict]:
        """Generate skill synergies with robust output"""
        
        synergy_prompt = ChatPromptTemplate.from_template("""
        Find powerful skill combinations from this list. Return ONLY valid JSON.

        Skills: {skills}

        Required JSON format:
        {{
            "ml_stack": {{
                "skills": ["Python", "PyTorch", "Computer Vision"],
                "synergy_multiplier": 1.3,
                "business_impact": "Enables end-to-end ML development"
            }}
        }}

        Create at least 3 synergy combinations. Return only JSON:
        """)
        
        return self._execute_robust_llm_prompt(synergy_prompt, {
            "skills": extracted_skills[:8]
        })
    
    def generate_contextual_evaluation_framework(self, job_description: str) -> Dict[str, any]:
        """Generate evaluation framework with simplified structure"""
        
        framework_prompt = ChatPromptTemplate.from_template("""
        Create evaluation weights for technical hiring. Return ONLY valid JSON.

        Job: {job_description}

        Required JSON format:
        {{
            "primary_weights": {{
                "technical_depth": 0.4,
                "technical_breadth": 0.2,
                "domain_expertise": 0.25,
                "learning_agility": 0.15
            }},
            "evaluation_criteria": {{
                "direct_match_bonus": 0.25,
                "related_skill_credit": 0.15,
                "synergy_bonus": 0.2,
                "experience_multiplier": 1.2
            }}
        }}

        Return only JSON:
        """)
        
        return self._execute_robust_llm_prompt(framework_prompt, {
            "job_description": job_description[:800]
        })
    
    def _execute_robust_llm_prompt(self, prompt_template: ChatPromptTemplate, variables: Dict) -> Dict:
        """Execute LLM prompt with extremely robust JSON parsing"""
        try:
            chain = prompt_template | self.llm
            result = chain.invoke(variables)
            
            if not result or not hasattr(result, 'content') or not result.content:
                return {}
                
            content = result.content
            parsed = self._clean_and_parse_json_robust(content)
            
            return parsed if isinstance(parsed, dict) else {}
            
        except Exception as e:
            print(f"LLM prompt execution failed: {e}")
            return {}
    
    def _clean_and_parse_json_robust(self, content: str) -> Dict:
        """Extremely robust JSON cleaning and parsing"""
        if not content:
            return {}
            
        content = content.strip()
        
        # Remove markdown code fences
        if content.startswith('```'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        
        content = content.strip()
        
        # Remove trailing commas before closing braces/brackets (common LLM error)
        content = re.sub(r",\s*([}$$])", r"\1", content)
        
        # Remove any text before first { or after last }
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            content = content[first_brace:last_brace+1]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            # Try to fix common issues
            try:
                # Fix unescaped quotes
                content = content.replace('"', '"').replace('"', '"')
                return json.loads(content)
            except:
                return {}
    
    def generate_complete_advanced_config(self, job_description: str) -> Dict:
        """Generate complete configuration with robust error handling"""
        
        print("🔄 Performing comprehensive skill extraction...")
        skill_patterns = self.generate_comprehensive_skill_extraction(job_description)
        
        if not skill_patterns:
            skill_patterns = self._get_comprehensive_fallback_patterns()
        
        extracted_skills = list(skill_patterns.keys())
        
        print("🔄 Generating advanced skill relationships...")
        skill_relationships = self.generate_advanced_skill_relationships(extracted_skills, job_description)
        
        print("🔄 Generating sophisticated transferable skills...")
        transferable_skills = self.generate_sophisticated_transferable_skills(extracted_skills)
        
        print("🔄 Generating comprehensive skill synergies...")
        skill_synergies = self.generate_comprehensive_skill_synergies(extracted_skills)
        
        print("🔄 Generating contextual evaluation framework...")
        evaluation_framework = self.generate_contextual_evaluation_framework(job_description)
        
        # Ensure we have a valid evaluation framework
        if not evaluation_framework:
            evaluation_framework = {
                "primary_weights": {
                    "technical_depth": 0.4,
                    "technical_breadth": 0.2,
                    "domain_expertise": 0.25,
                    "learning_agility": 0.15
                },
                "evaluation_criteria": {
                    "direct_match_bonus": 0.25,
                    "related_skill_credit": 0.15,
                    "synergy_bonus": 0.2,
                    "experience_multiplier": 1.2
                }
            }
        
        # Generate skill importance
        skill_importance = {}
        for skill, data in skill_patterns.items():
            base_importance = data.get('importance', 0.5)
            domain_relevance = data.get('domain_relevance', 0.8)
            skill_importance[skill] = min(1.0, base_importance * domain_relevance)
        
        return {
            "skill_patterns": skill_patterns,
            "skill_relationships": skill_relationships,
            "transferable_skills": transferable_skills,
            "skill_synergies": skill_synergies,
            "evaluation_framework": evaluation_framework,
            "skill_importance": skill_importance,
            "extracted_skills": extracted_skills,
            "generation_metadata": {
                "total_skills_extracted": len(extracted_skills),
                "relationship_mappings": len(skill_relationships),
                "transferable_mappings": len(transferable_skills),
                "synergy_combinations": len(skill_synergies)
            }
        }
    
    def _get_comprehensive_fallback_patterns(self) -> Dict[str, Dict]:
        """Comprehensive fallback patterns with all required fields"""
        return {
            "Python": {
                "primary_pattern": r"\bpython\b",
                "variations": ["py", "python3"],
                "importance": 0.95,
                "skill_category": "programming_language",
                "proficiency_indicators": {
                    "expert": ["expert", "mastery", "architect"],
                    "senior": ["senior", "extensive", "5+ years"],
                    "intermediate": ["experience", "familiar", "worked with"],
                    "beginner": ["basic", "learning", "exposure"]
                },
                "context_clues": ["programming", "development"],
                "domain_relevance": 0.95,
                "market_demand": 0.9
            },
            "PyTorch": {
                "primary_pattern": r"\bpytorch|torch\b",
                "variations": ["pytorch", "torch"],
                "importance": 0.9,
                "skill_category": "ml_framework",
                "proficiency_indicators": {
                    "expert": ["expert", "extensive", "deep"],
                    "senior": ["hands-on", "experience", "implemented"],
                    "intermediate": ["familiar", "worked with", "used"],
                    "beginner": ["exposure", "basic", "learning"]
                },
                "context_clues": ["deep learning", "neural networks"],
                "domain_relevance": 0.9,
                "market_demand": 0.85
            },
            "NeRF": {
                "primary_pattern": r"\bnerf|neural radiance field\b",
                "variations": ["nerf", "neural radiance"],
                "importance": 0.9,
                "skill_category": "domain_specific",
                "proficiency_indicators": {
                    "expert": ["expert", "extensive", "research"],
                    "senior": ["hands-on", "implemented", "experience"],
                    "intermediate": ["familiar", "worked with"],
                    "beginner": ["exposure", "learning"]
                },
                "context_clues": ["3d reconstruction", "computer vision"],
                "domain_relevance": 0.95,
                "market_demand": 0.8
            },
            "3D Simulation": {
                "primary_pattern": r"\b3d simulation|simulation.*3d\b",
                "variations": ["3d", "simulation"],
                "importance": 0.95,
                "skill_category": "domain_specific",
                "proficiency_indicators": {
                    "expert": ["expert", "built", "pipelines"],
                    "senior": ["extensive", "implemented", "developed"],
                    "intermediate": ["hands-on", "worked with"],
                    "beginner": ["exposure", "basic"]
                },
                "context_clues": ["virtual reality", "game engine"],
                "domain_relevance": 0.95,
                "market_demand": 0.8
            },
            "Computer Vision": {
                "primary_pattern": r"\bcomputer vision|cv\b",
                "variations": ["cv", "vision"],
                "importance": 0.85,
                "skill_category": "core_technical",
                "proficiency_indicators": {
                    "expert": ["expert", "extensive", "specialist"],
                    "senior": ["hands-on", "experience", "implemented"],
                    "intermediate": ["familiar", "worked with"],
                    "beginner": ["exposure", "basic"]
                },
                "context_clues": ["image", "visual", "detection"],
                "domain_relevance": 0.9,
                "market_demand": 0.85
            }
        }

# --------------------------------------------------------
# 2. Enhanced Neural Components
# --------------------------------------------------------

class EnhancedCandidateJobMatchingNetwork(nn.Module):
    def __init__(self, input_dim: int = 384):
        super().__init__()
        
        self.candidate_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        self.job_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        self.compatibility_scorer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, candidate_embedding: torch.Tensor, job_embedding: torch.Tensor):
        cand_features = self.candidate_encoder(candidate_embedding)
        job_features = self.job_encoder(job_embedding)
        combined_features = torch.cat([cand_features, job_features], dim=-1)
        compatibility_score = self.compatibility_scorer(combined_features)
        return compatibility_score, cand_features, job_features

@dataclass
class ReasoningStep:
    step_type: str
    input_facts: List[str]
    reasoning_rule: str
    output_conclusion: str
    confidence: float
    neural_support: float = 0.0

# --------------------------------------------------------
# 3. Fixed Advanced Neurosymbolic System (Dictionary-Based)
# --------------------------------------------------------

class AdvancedNeurosymbolicRecommendationSystem:
    """Fixed neurosymbolic system using dictionaries with robust evaluation"""
    
    def __init__(self):
        self.pattern_generator = AdvancedDynamicLLMPatternGenerator()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.matching_network = EnhancedCandidateJobMatchingNetwork(input_dim=384)
        self.knowledge_graph = nx.Graph()
        self.matching_network.eval()
        
        self.dynamic_config = {}
    
    def initialize_from_job_description(self, job_description: str) -> Dict:
        """Initialize system with comprehensive configuration"""
        
        print("🚀 Generating comprehensive dynamic configuration...")
        self.dynamic_config = self.pattern_generator.generate_complete_advanced_config(job_description)
        
        self._build_knowledge_graph()
        
        return self.dynamic_config
    
    def _build_knowledge_graph(self):
        """Build knowledge graph from generated relationships"""
        
        self.knowledge_graph.clear()
        
        for skill, data in self.dynamic_config.get('skill_relationships', {}).items():
            for implied_skill in data.get('implies', []):
                self.knowledge_graph.add_edge(
                    skill, implied_skill,
                    weight=data.get('confidence', 0.5),
                    type='implication'
                )
    
    def evaluate_candidate_comprehensive(self, candidate_name: str, candidate_profile: str, job_description: str) -> Dict:
        """Comprehensive candidate evaluation returning dictionary"""
        
        if not self.dynamic_config:
            self.initialize_from_job_description(job_description)
        
        print("🧠 Performing advanced neural analysis...")
        neural_results = self._neural_analysis(candidate_profile, job_description)
        
        print("🔗 Executing sophisticated symbolic reasoning...")
        reasoning_chain = self._symbolic_reasoning(candidate_profile)
        
        print("⚖️ Synthesizing evidence with framework...")
        final_evaluation = self._synthesize_evidence(neural_results, reasoning_chain)
        
        insights = self._generate_insights(candidate_profile, reasoning_chain)
        
        # FIXED: Return dictionary structure
        return {
            'name': candidate_name,
            'final_score': final_evaluation['final_score'],
            'neural_results': neural_results,
            'reasoning_chain': reasoning_chain,
            'insights': insights,
            'evidence_synthesis': final_evaluation,
            'recommendation': self._generate_recommendation(final_evaluation['final_score'], insights),
            'confidence_level': self._calculate_confidence(reasoning_chain, neural_results),
            'skill_breakdown': self._generate_skill_breakdown(reasoning_chain),
            'evaluation_metadata': {
                'total_reasoning_steps': len(reasoning_chain),
                'high_confidence_steps': len([s for s in reasoning_chain if s.confidence > 0.8]),
                'neural_confidence': neural_results.get('confidence', 0.0),
                'evaluation_completeness': min(1.0, len(reasoning_chain) / 10)
            }
        }
    
    def _neural_analysis(self, candidate_profile: str, job_description: str) -> Dict:
        """Neural analysis with enhanced features"""
        
        candidate_emb = torch.tensor(self.embedder.encode(candidate_profile), dtype=torch.float32)
        job_emb = torch.tensor(self.embedder.encode(job_description), dtype=torch.float32)
        
        if candidate_emb.dim() == 1:
            candidate_emb = candidate_emb.unsqueeze(0)
        if job_emb.dim() == 1:
            job_emb = job_emb.unsqueeze(0)
        
        with torch.no_grad():
            compatibility_score, cand_features, job_features = self.matching_network(
                candidate_emb, job_emb
            )
            
            feature_similarity = F.cosine_similarity(cand_features, job_features, dim=-1)
            confidence = 1.0 - min(1.0, torch.std(torch.cat([compatibility_score, feature_similarity.unsqueeze(0)])).item() * 2)
        
        return {
            'compatibility_score': compatibility_score.item(),
            'feature_similarity': feature_similarity.item(),
            'candidate_features': cand_features.numpy(),
            'job_features': job_features.numpy(),
            'confidence': confidence
        }
    
    def _symbolic_reasoning(self, candidate_profile: str) -> List[ReasoningStep]:
        """Symbolic reasoning with dynamic patterns"""
        
        reasoning_chain = []
        
        skill_patterns = self.dynamic_config.get('skill_patterns', {})
        skill_importance = self.dynamic_config.get('skill_importance', {})
        
        # Skill matching
        for skill, pattern_data in skill_patterns.items():
            pattern_match = self._pattern_match(candidate_profile, pattern_data)
            neural_similarity = self._compute_neural_similarity(candidate_profile, skill)
            
            proficiency_level = self._assess_proficiency(candidate_profile, pattern_data)
            importance = skill_importance.get(skill, 0.5)
            
            combined_confidence = (pattern_match * 0.6 + neural_similarity * 0.4) * importance
            
            if combined_confidence > 0.25:
                step = ReasoningStep(
                    step_type='skill_match',
                    input_facts=[f"Pattern: {pattern_match:.3f}, Neural: {neural_similarity:.3f}"],
                    reasoning_rule=f"Multi-signal matching for {skill}",
                    output_conclusion=f"Candidate demonstrates {proficiency_level} proficiency in {skill}",
                    confidence=combined_confidence,
                    neural_support=neural_similarity
                )
                reasoning_chain.append(step)
        
        # Skill implications
        skill_relationships = self.dynamic_config.get('skill_relationships', {})
        matched_skills = [self._extract_skill_from_conclusion(step.output_conclusion) for step in reasoning_chain]
        
        for skill, relationship_data in skill_relationships.items():
            if skill in matched_skills:
                for implied_skill in relationship_data.get('implies', []):
                    confidence = relationship_data.get('confidence', 0.5)
                    
                    step = ReasoningStep(
                        step_type='implication',
                        input_facts=[f"Confirmed {skill}"],
                        reasoning_rule=relationship_data.get('reasoning', f"{skill} implies {implied_skill}"),
                        output_conclusion=f"Inferred capability in {implied_skill}",
                        confidence=confidence,
                        neural_support=self._compute_neural_similarity(candidate_profile, implied_skill)
                    )
                    reasoning_chain.append(step)
        
        return reasoning_chain
    
    def _pattern_match(self, text: str, pattern_data: Dict) -> float:
        """Pattern matching with context awareness"""
        
        primary_pattern = pattern_data.get('primary_pattern', '')
        variations = pattern_data.get('variations', [])
        
        text_lower = text.lower()
        score = 0.0
        
        try:
            if re.search(primary_pattern, text_lower, re.IGNORECASE):
                score += 0.8
        except:
            pass
        
        for variation in variations:
            if variation.lower() in text_lower:
                score += 0.3
        
        return min(1.0, score)
    
    def _assess_proficiency(self, text: str, pattern_data: Dict) -> str:
        """Assess proficiency using indicators"""
        
        proficiency_indicators = pattern_data.get('proficiency_indicators', {})
        text_lower = text.lower()
        
        expert_count = sum(1 for indicator in proficiency_indicators.get('expert', []) 
                          if indicator.lower() in text_lower)
        senior_count = sum(1 for indicator in proficiency_indicators.get('senior', []) 
                          if indicator.lower() in text_lower)
        intermediate_count = sum(1 for indicator in proficiency_indicators.get('intermediate', []) 
                               if indicator.lower() in text_lower)
        
        if expert_count >= 1:
            return "expert"
        elif senior_count >= 1:
            return "senior"
        elif intermediate_count >= 1:
            return "intermediate"
        else:
            return "beginner"
    
    def _extract_skill_from_conclusion(self, conclusion: str) -> str:
        """Extract skill name from reasoning conclusion"""
        words = conclusion.split()
        if 'in' in words:
            skill_index = words.index('in') + 1
            if skill_index < len(words):
                return words[skill_index]
        return ""
    
    def _compute_neural_similarity(self, text: str, skill: str) -> float:
        """Compute neural similarity"""
        try:
            text_emb = self.embedder.encode(text)
            skill_emb = self.embedder.encode(skill)
            similarity = np.dot(text_emb, skill_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(skill_emb))
            return float(max(0, similarity))
        except:
            return 0.0
    
    def _synthesize_evidence(self, neural_results: Dict, reasoning_chain: List[ReasoningStep]) -> Dict:
        """Synthesize evidence using framework"""
        
        evaluation_framework = self.dynamic_config.get('evaluation_framework', {})
        primary_weights = evaluation_framework.get('primary_weights', {})
        evaluation_criteria = evaluation_framework.get('evaluation_criteria', {})
        
        neural_score = neural_results['compatibility_score']
        reasoning_scores = [step.confidence for step in reasoning_chain]
        symbolic_score = np.mean(reasoning_scores) if reasoning_scores else 0.0
        
        skill_matches = len([s for s in reasoning_chain if s.step_type == 'skill_match'])
        implications = len([s for s in reasoning_chain if s.step_type == 'implication'])
        
        direct_match_bonus = skill_matches * evaluation_criteria.get('direct_match_bonus', 0.15)
        implication_bonus = implications * evaluation_criteria.get('related_skill_credit', 0.1)
        
        technical_depth_score = (symbolic_score * 0.7) + (neural_score * 0.3)
        technical_breadth_score = min(1.0, skill_matches / 10.0)
        
        final_score = (
            technical_depth_score * primary_weights.get('technical_depth', 0.4) +
            technical_breadth_score * primary_weights.get('technical_breadth', 0.2) +
            neural_results.get('feature_similarity', 0) * primary_weights.get('domain_expertise', 0.25) +
            0.15 * primary_weights.get('learning_agility', 0.15) +
            direct_match_bonus +
            implication_bonus
        ) * 100
        
        return {
            'final_score': min(100.0, final_score),
            'component_scores': {
                'neural_score': neural_score,
                'symbolic_score': symbolic_score,
                'technical_depth': technical_depth_score,
                'technical_breadth': technical_breadth_score
            },
            'bonuses': {
                'direct_match_bonus': direct_match_bonus,
                'implication_bonus': implication_bonus
            }
        }
    
    def _generate_insights(self, candidate_profile: str, reasoning_chain: List[ReasoningStep]) -> Dict:
        """Generate insights about candidate"""
        
        insights = {
            'strengths': [],
            'growth_areas': [],
            'unique_advantages': [],
            'cultural_fit_indicators': []
        }
        
        # Extract strengths
        high_confidence_steps = [s for s in reasoning_chain if s.confidence > 0.7]
        for step in high_confidence_steps:
            if 'expert' in step.output_conclusion.lower():
                skill = step.output_conclusion.split()[-1]
                insights['strengths'].append(f"Expert-level {skill}")
        
        # Cultural fit indicators
        profile_lower = candidate_profile.lower()
        if any(word in profile_lower for word in ['team', 'collaboration', 'mentored']):
            insights['cultural_fit_indicators'].append("Strong collaboration indicators")
        
        if any(word in profile_lower for word in ['research', 'published', 'innovation']):
            insights['cultural_fit_indicators'].append("Research and innovation oriented")
        
        return insights
    
    def _generate_recommendation(self, final_score: float, insights: Dict) -> str:
        """Generate hiring recommendation"""
        
        if final_score >= 85:
            recommendation = "🌟 STRONG HIRE - Exceptional candidate with excellent skill alignment"
        elif final_score >= 75:
            recommendation = "✅ HIRE - Strong candidate who meets requirements"
        elif final_score >= 65:
            recommendation = "🤔 CONSIDER - Solid candidate with some gaps"
        elif final_score >= 50:
            recommendation = "⚠️ WEAK HIRE - Significant skill gaps"
        else:
            recommendation = "❌ NO HIRE - Insufficient alignment"
        
        if len(insights.get('cultural_fit_indicators', [])) >= 2:
            recommendation += " (Good cultural fit indicators)"
        
        return recommendation
    
    def _calculate_confidence(self, reasoning_chain: List[ReasoningStep], neural_results: Dict) -> float:
        """Calculate overall confidence"""
        
        if not reasoning_chain:
            return 0.3
        
        high_conf_steps = len([s for s in reasoning_chain if s.confidence > 0.8])
        total_steps = len(reasoning_chain)
        
        reasoning_confidence = high_conf_steps / total_steps if total_steps > 0 else 0
        neural_confidence = neural_results.get('confidence', 0.5)
        
        return (reasoning_confidence * 0.7) + (neural_confidence * 0.3)
    
    def _generate_skill_breakdown(self, reasoning_chain: List[ReasoningStep]) -> Dict:
        """Generate skill breakdown"""
        
        breakdown = {
            'expert_skills': [],
            'senior_skills': [],
            'intermediate_skills': [],
            'beginner_skills': []
        }
        
        for step in reasoning_chain:
            if step.step_type == 'skill_match':
                conclusion = step.output_conclusion.lower()
                skill = step.output_conclusion.split()[-1]
                
                if 'expert' in conclusion:
                    breakdown['expert_skills'].append(skill)
                elif 'senior' in conclusion:
                    breakdown['senior_skills'].append(skill)
                elif 'intermediate' in conclusion:
                    breakdown['intermediate_skills'].append(skill)
                else:
                    breakdown['beginner_skills'].append(skill)
        
        return breakdown

# --------------------------------------------------------
# 4. FIXED Main Execution Function
# --------------------------------------------------------

def run_advanced_neurosymbolic_system():
    """Run the complete system with dictionary-based results"""
    
    system = AdvancedNeurosymbolicRecommendationSystem()
    
    job_description = """
    Simulation/Machine Learning Engineer — 3D Simulation & Generative Modeling
    Design and deploy ML-driven 3D simulation systems that model complex real-world environments, agents, and physics. You will fuse differentiable simulation, generative 3D (NeRF/Gaussian Splatting/Mesh), and reinforcement learning to create high-fidelity, data-efficient virtual worlds for experimentation, training, and digital twin use cases.
    
    Requirements: Python, C++, PyTorch/JAX, CUDA, Isaac Sim, Unreal, NeRF, Computer Vision, Reinforcement Learning, 3D Simulation, Machine Learning, Deep Learning, GPU Programming, Distributed Computing
    """
    
    developer_profiles = {
        "Ananya Sharma": """
        Built end-to-end 3D simulation pipelines in Isaac Sim & Unreal Engine, with GPU-accelerated synthetic data generation for autonomous navigation projects.
        Extensive hands-on with NeRFs, Gaussian Splatting, PyTorch3D, Kaolin, and differentiable rendering (Mitsuba).
        Strong foundation in RL (PPO, SAC, DreamerV3) — deployed trained policies from sim-to-real on quadruped robots.
        Optimized large-scale simulation workloads using Ray + Slurm, achieving 10x faster rollouts in parallel clusters.
        Published research in differentiable physics and domain randomization for sim-to-real transfer.
        Led a team of 4 engineers developing next-generation simulation infrastructure.
        5 years experience in ML & 3D simulation, M.S. in Computer Vision & Robotics, NVIDIA Omniverse team background.
        Expert in Python, C++, CUDA programming. Senior experience with distributed computing and cloud deployment.
        """,
        "Raghav Mehta": """
        Hands-on with PyTorch, JAX, and RL frameworks — trained agents in Mujoco & PyBullet for 3 years.
        Solid understanding of numerical optimization, RL algorithms, and control theory.
        Experience with real-world robotics (ROS1, embedded deployments) and sensor integration.
        Strong Python coder, clean experimentation practices, CI/CD familiar.
        Limited exposure to photorealistic 3D simulation (has only used Mujoco/Bullet, no Unreal/Omniverse).
        No direct experience with differentiable rendering, NeRF, or synthetic data pipelines.
        GPU optimization knowledge is basic — hasn't scaled beyond single GPU workloads.
        Intermediate experience in machine learning research, published 2 conference papers.
        """,
        "Priya Nair": """
        Expert in Unreal, Unity, and GPU rendering pipelines — built large-scale simulation environments for gaming and AR/VR.
        Mastery of C++, CUDA, OpenGL, Vulkan for high-performance graphics and real-time rendering.
        Led teams of 8+ developers delivering production-quality simulation engines with photoreal realism.
        Architect of distributed rendering systems handling millions of concurrent users.
        Excellent in project leadership, architecture design, and performance optimization.
        Limited exposure to machine learning & RL — hasn't worked with PyTorch/JAX or trained ML-based agents.
        No hands-on with NeRFs, differentiable physics, or generative 3D models.
        Strong graphics expertise, but missing the ML-driven simulation integration required for this role.
        12+ years in graphics simulation, senior architect level, startup and enterprise experience.
        """
    }
    
    print("🧠 ADVANCED NEUROSYMBOLIC RECOMMENDATION SYSTEM")
    print("=" * 80)
    print("   🚨 COMPREHENSIVE EVALUATION WITH ROBUST JSON PARSING 🚨")
    
    # Initialize system
    config = system.initialize_from_job_description(job_description)
    
    print(f"\n📊 Advanced Configuration Generated:")
    metadata = config.get('generation_metadata', {})
    print(f"   -  Skills Extracted: {metadata.get('total_skills_extracted', 0)}")
    print(f"   -  Relationship Mappings: {metadata.get('relationship_mappings', 0)}")
    print(f"   -  Transferable Mappings: {metadata.get('transferable_mappings', 0)}")
    print(f"   -  Synergy Combinations: {metadata.get('synergy_combinations', 0)}")
    
    # FIXED: Evaluate candidates returning dictionary list
    results = []
    for dev_name, profile in developer_profiles.items():
        print(f"\n🔄 Comprehensive evaluation of {dev_name}...")
        try:
            evaluation = system.evaluate_candidate_comprehensive(dev_name, profile, job_description)
            results.append(evaluation)  # Already a dictionary
        except Exception as e:
            print(f"Error evaluating {dev_name}: {e}")
            continue
    
    # FIXED: Sort using dictionary access - NO MORE TUPLE ERRORS!
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    print(f"\n🏆 COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)
    
    for rank, candidate in enumerate(results, 1):
        print(f"\n#{rank} {candidate['name']}")
        print(f"   Final Score: {candidate['final_score']:.1f}/100")
        print(f"   {candidate['recommendation']}")
        print(f"   Confidence Level: {candidate['confidence_level']:.1f}/1.0")
        
        metadata = candidate['evaluation_metadata']
        print(f"   🔗 Reasoning Analysis:")
        print(f"      -  Total Steps: {metadata['total_reasoning_steps']}")
        print(f"      -  High Confidence: {metadata['high_confidence_steps']}")
        print(f"      -  Neural Confidence: {metadata['neural_confidence']:.3f}")
        
        insights = candidate['insights']
        if insights['strengths']:
            print(f"   💪 Key Strengths: {', '.join(insights['strengths'][:3])}")
        
        skill_breakdown = candidate['skill_breakdown']
        if skill_breakdown['expert_skills']:
            print(f"   🎯 Expert Skills: {', '.join(skill_breakdown['expert_skills'][:3])}")
        
        evidence = candidate['evidence_synthesis']
        component_scores = evidence['component_scores']
        print(f"   ⚖️ Component Scores:")
        print(f"      -  Technical Depth: {component_scores['technical_depth']:.3f}")
        print(f"      -  Technical Breadth: {component_scores['technical_breadth']:.3f}")
        print(f"      -  Neural Score: {component_scores['neural_score']:.3f}")
        print(f"      -  Symbolic Score: {component_scores['symbolic_score']:.3f}")
    
    # FIXED: Final recommendation with dictionary access
    if results:
        print(f"\n✨ FINAL NEUROSYMBOLIC RECOMMENDATION")
        print("=" * 60)
        best_candidate = results  # First element in sorted list
        print(f"🎯 RECOMMENDED HIRE: {best_candidate[0]}")
        print(f"   Score: {best_candidate[0]['final_score']:.1f}/100")
        print(f"   {best_candidate[0]['recommendation']}")
        print(f"   Reasoning Steps: {best_candidate[0]['evaluation_metadata']['total_reasoning_steps']}")
        print(f"   Confidence: {best_candidate[0]['confidence_level']:.1f}/1.0")

if __name__ == "__main__":
    run_advanced_neurosymbolic_system()
