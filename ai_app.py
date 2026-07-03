import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchResults

# =====================================
# GEMINI API KEY
# =====================================


# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Search Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Search Agent")
st.write("LangChain + Gemini + DuckDuckGo Search")

# =====================================
# LLM
# =====================================
api_key = st.secrets["GOOGLE_API_KEY"]
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=api_key,
    temperature=0.3,
)

# =====================================
# TOOLS
# =====================================

search = DuckDuckGoSearchResults()

tools = [search]

# =====================================
# AGENT
# =====================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a helpful AI assistant.

Use DuckDuckGo Search whenever needed to answer the user's questions.

Give clear, concise and accurate answers.
"""
)

# =====================================
# MEMORY
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for message in st.session_state.messages:

    role = "assistant"

    if isinstance(message, HumanMessage):
        role = "user"

    with st.chat_message(role):
        st.markdown(message.content)

# =====================================
# CHAT INPUT
# =====================================

user_input = st.chat_input("Ask me anything...")

if user_input:

    human = HumanMessage(content=user_input)
    st.session_state.messages.append(human)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Searching..."):

        response = agent.invoke(
            {
                "messages": st.session_state.messages
            }
        )

        ai_message = response["messages"][-1]

        st.session_state.messages.append(ai_message)

    with st.chat_message("assistant"):
        st.markdown(ai_message.content)
