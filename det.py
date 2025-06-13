# app.py

import os
import requests
import streamlit as st
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# === Load environment variables ===
load_dotenv()
os.environ["SERPAPI_API_KEY"] = "70a10bc50d4160d064884939f68e4faf25c9324df717349abf987527edf9f01f"
os.environ["GROQ_API_KEY"] = "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"
SERPAPI_KEY = os.environ["SERPAPI_API_KEY"]
llm = ChatGroq(
    temperature=0.7,
    model_name="gemma2-9b-it"
)

# === LangGraph state ===
class TripState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "Conversation"]
    location: str
    days: int
    data: dict
    final: str

# === Google Maps-safe search ===
def search_places(query, location):
    url = f"https://serpapi.com/search.json?q={query}+in+{location}&api_key={SERPAPI_KEY}&hl=en&gl=in"
    try:
        res = requests.get(url).json()
    except Exception as e:
        return f"Error from SerpAPI: {e}"

    results = []

    for r in res.get("local_results", []):
        if isinstance(r, dict):
            name = r.get("title", "Unknown")
            safe_link = f"https://www.google.com/maps/search/?api=1&query={requests.utils.quote(name + ' ' + location)}"
            results.append(f"{name} [Maps]({safe_link})")

    for r in res.get("organic_results", [])[:3]:
        if isinstance(r, dict):
            title = r.get("title", "")
            link = r.get("link", "")
            results.append(f"{title} - {link}")

    return "\n".join(results) or "No results found"

# === LangGraph Node 1 ===
def parse_request(state: TripState) -> TripState:
    user_msg = state["messages"][-1].content
    prompt = f"Extract city and days from this input:\n'{user_msg}'\nReply like:\nCity: <city>\nDays: <number>"
    res = llm.invoke([HumanMessage(content=prompt)]).content
    try:
        lines = res.strip().splitlines()
        city = lines[0].split(":")[1].strip()
        days = int(lines[1].split(":")[1].strip())
    except:
        city, days = "Bhubaneswar", 3
    return {**state, "location": city, "days": days}

# === LangGraph Node 2 ===
def get_data(state: TripState) -> TripState:
    city = state["location"]
    data = {
        "spots": search_places("top tourist attractions", city),
        "food": search_places("best street food", city),
        "events": search_places("local events", city)
    }
    return {**state, "data": data}

# === LangGraph Node 3 ===
def generate_itinerary(state: TripState) -> TripState:
    city = state["location"]
    days = state["days"]
    d = state["data"]

    prompt = f"""
Plan a fun {days}-day trip to {city} in Hinglish with emojis 😎.

Use:
Tourist Spots:\n{d['spots']}
Street Food:\n{d['food']}
Events:\n{d['events']}

🗓️ Format:
Day 1:
🌄 Morning (7–10am): ...
🍽️ Brunch (10–12pm): ...
🏛️ Afternoon (12–5pm): ...
🌇 Evening (5–8pm): ...
🌃 Night (8–12am): ...

Add Google Maps links.
End with: 'aur chahiye toh message kardena! 😁'
"""
    result = llm.invoke([HumanMessage(content=prompt)]).content
    return {**state, "final": result}

# === LangGraph flow ===
builder = StateGraph(TripState)
builder.add_node("parse", parse_request)
builder.add_node("search", get_data)
builder.add_node("plan", generate_itinerary)

builder.set_entry_point("parse")
builder.add_edge("parse", "search")
builder.add_edge("search", "plan")
builder.add_edge("plan", END)

graph = builder.compile()

# === Run function ===
def plan_trip(input_text: str) -> str:
    state = {
        "messages": [HumanMessage(content=input_text)],
        "location": "",
        "days": 0,
        "data": {},
        "final": ""
    }
    result = graph.invoke(state)
    return result['final']

# === Streamlit UI ===
st.set_page_config(page_title="Jolly Itinerary Planner", page_icon="🌍")
st.title("🌍 Jolly Itinerary Planner (AI-Powered)")
st.markdown("Type something like **'Plan a 3 day trip to Bhubaneswar with food and events'**")

user_input = st.text_input("Your Trip Plan", "")

if user_input:
    with st.spinner("Planning your trip..."):
        output = plan_trip(user_input)
        st.markdown("---")
        st.markdown(output)
