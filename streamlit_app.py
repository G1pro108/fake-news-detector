import streamlit as st
import concurrent.futures
from openai import OpenAI
from duckduckgo_search import DDGS
from deep_translator import GoogleTranslator
import time

# --- Page Config ---
st.set_page_config(page_title="Global Swarm Fact-Checker", page_icon="🌍", layout="wide")
st.title("🌍 Global 350-Agent Fact-Checker")
st.markdown("Accepts any language. Deploys 350 domain-specific agents + 3 AI Judges.")

# --- YOUR ENDPOINTS (Replace with your actual URLs/keys) ---
KAGGLE_URL = "https://YOUR-RANDOM-WORDS.trycloudflare.com/v1" 
CLOUDFLARE_ACCOUNT_ID = "YOUR_CLOUDFLARE_ACCOUNT_ID" 
CLOUDFLARE_API_TOKEN = "YOUR_CLOUDFLARE_API_TOKEN" 
KEYLESS_URL = "https://keylessai.thryx.workers.dev/v1" 

text_input = st.text_area("Paste news text here (Any Language):", height=150)

if st.button("Deploy Swarm & Consult Panel", type="primary"):
    if not text_input.strip():
        st.warning("Please enter text.")
        st.stop()

    # ==========================================
    # PHASE 0: MULTILINGUAL TRANSLATION
    # ==========================================
    st.subheader("🌐 Phase 0: Language Processing...")
    try:
        # Auto-detect and translate to English for the agents and judges
        translator = GoogleTranslator(source='auto', target='en')
        translated_text = translator.translate(text_input)
        
        if translated_text.lower() != text_input.lower():
            st.info(f"📝 **Translated to English for analysis:**\n\n{translated_text[:300]}...")
    except Exception as e:
        translated_text = text_input # Fallback to original if translation fails

    # ==========================================
    # PHASE 1: THE 350-AGENT SWARM
    # ==========================================
    st.subheader("🐝 Phase 1: Deploying 350 Domain-Specific Agents...")
    
    # Extract keywords for search
    base_query = " ".join(translated_text.split()[:12])
    
    # 10 Categories x 35 Agents = 350 Agents
    domain_pools = {
        "Politics": ["site:reuters.com", "site:apnews.com", "site:thehindu.com", "election", "government"],
        "International": ["site:bbc.com", "site:aljazeera.com", "site:un.org", "diplomacy", "global"],
        "Business": ["site:bloomberg.com", "site:wsj.com", "site:livemint.com", "market", "economy"],
        "Health": ["site:who.int", "site:nih.gov", "site:medicalnewstoday.com", "medical", "health"],
        "Science": ["site:nature.com", "site:sciencemag.org", "site:space.com", "study", "research"],
        "Environment": ["site:unep.org", "site:nasa.gov", "climate", "environment", "disaster"],
        "Crime": ["site:ndtv.com", "site:indianexpress.com", "police", "court", "investigation"],
        "Sports": ["site:espn.com", "site:skysports.com", "tournament", "match"],
        "Entertainment": ["site:variety.com", "site:hollywoodreporter.com", "movie", "celebrity"],
        "FactCheck": ["site:snopes.com", "site:altnews.in", "site:pib.gov.in", "hoax", "fake"]
    }

    agent_votes = []
    progress_bar = st.progress(0)
    
    def domain_agent_task(query, domain, site_filter):
        """A specialized agent that searches a specific domain."""
        try:
            time.sleep(0.1) # Prevent IP bans
            search_query = f"{query} {site_filter}"
            results = DDGS().text(search_query, max_results=1)
            
            if not results:
                return (0, domain) # Neutral if no results

            combined_text = " ".join([r['body'] for r in results]).lower()
            
            # Fact-check specific logic
            if any(word in combined_text for word in ["fake", "hoax", "false", "misleading", "debunked"]):
                return (-1, domain) # Vote Fake
            elif any(word in combined_text for word in ["true", "confirmed", "real", "official"]):
                return (1, domain)  # Vote Real
            return (0, domain)
        except:
            return (0, domain)

    # Run 350 agents (10 at a time to keep Streamlit stable)
    total_agents = 350
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for domain, sites in domain_pools.items():
            for i in range(35): # 35 agents per domain
                site = sites[i % len(sites)]
                futures.append(executor.submit(domain_agent_task, base_query, domain, site))
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            vote, domain = future.result()
            agent_votes.append((vote, domain))
            progress_bar.progress((i + 1) / total_agents)
            
    agent_fake_votes = sum(1 for v, _ in agent_votes if v == -1)
    agent_real_votes = sum(1 for v, _ in agent_votes if v == 1)
    
    st.success(f"✅ 350 Agents deployed. Found {agent_fake_votes} Fake signals and {agent_real_votes} Real signals.")
    progress_bar.empty()

    # ==========================================
    # PHASE 2: THE 3-JUDGE PANEL
    # ==========================================
    st.subheader("🏛️ Phase 2: Consulting the 3-Judge Panel...")
    judge_votes = []
    
    # We use the translated English text for the judges
    text_for_judges = translated_text[:2000]

    # Judge 1: Kaggle
    with st.spinner("Judge 1 (Kaggle) is analyzing..."):
        try:
            client = OpenAI(base_url=KAGGLE_URL, api_key="not-needed")
            prompt = f"Is this text real or fake? Return ONLY 'REAL' or 'FAKE'. Text: {text_for_judges}"
            res = client.chat.completions.create(model="qwen", messages=[{"role": "user", "content": prompt}])
            verdict = res.choices[0].message.content.strip().upper()
            judge_votes.append(1 if "REAL" in verdict else -1)
            st.success("Judge 1 (Kaggle): Vote recorded.")
        except: st.warning("Judge 1 is offline.")

    # Judge 2: Cloudflare
    with st.spinner("Judge 2 (Cloudflare) is analyzing..."):
        try:
            cf_url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"
            client = OpenAI(base_url=cf_url, api_key=CLOUDFLARE_API_TOKEN)
            prompt = f"Is this text real or fake? Return ONLY 'REAL' or 'FAKE'. Text: {text_for_judges}"
            res = client.chat.completions.create(model="@cf/meta/llama-3.3-70b-instruct-fp8-fast", messages=[{"role": "user", "content": prompt}])
            verdict = res.choices[0].message.content.strip().upper()
            judge_votes.append(1 if "REAL" in verdict else -1)
            st.success("Judge 2 (Cloudflare): Vote recorded.")
        except: st.warning("Judge 2 is offline.")

    # Judge 3: keylessai
    with st.spinner("Judge 3 (keylessai) is analyzing..."):
        try:
            client = OpenAI(base_url=KEYLESS_URL, api_key="not-needed")
            prompt = f"Is this text real or fake? Return ONLY 'REAL' or 'FAKE'. Text: {text_for_judges}"
            res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
            verdict = res.choices[0].message.content.strip().upper()
            judge_votes.append(1 if "REAL" in verdict else -1)
            st.success("Judge 3 (keylessai): Vote recorded.")
        except: st.warning("Judge 3 is offline.")

    # ==========================================
    # PHASE 3: THE WEIGHTED CONSENSUS
    # ==========================================
    st.divider()
    st.subheader("🏆 Final Consensus Verdict")
    
    if not judge_votes:
        st.error("All judges are offline. Cannot make a final decision.")
        st.stop()

    # Judges get 30 votes each. Agents get 1 vote each.
    # This ensures the LLMs have the final say, but the 350 agents provide massive context.
    total_fake_score = (agent_fake_votes * 1) + (judge_votes.count(-1) * 30)
    total_real_score = (agent_real_votes * 1) + (judge_votes.count(1) * 30)

    # Breakdown by Category (Optional but looks amazing in UI)
    st.markdown("**Agent Consensus Breakdown:**")
    for domain in domain_pools.keys():
        d_votes = [v for v, d in agent_votes if d == domain]
        st.write(f"- *{domain}*: {d_votes.count(-1)} Fake / {d_votes.count(1)} Real")

    if total_fake_score > total_real_score:
        st.error(f"🚨 **VERDICT: FAKE**\n(Score: Fake={total_fake_score}, Real={total_real_score})")
    else:
        st.success(f"✅ **VERDICT: REAL**\n(Score: Fake={total_fake_score}, Real={total_real_score})")
